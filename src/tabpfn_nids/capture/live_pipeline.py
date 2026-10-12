"""End-to-end coordinator for live packet capture, flow aggregation, and TabPFN inference."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from tabpfn_nids.api.event_bus import EventBus
from tabpfn_nids.capture.capture_config import CaptureConfig
from tabpfn_nids.capture.tshark_capture import TSharkCapture
from tabpfn_nids.detection.window_manager import WindowManager
from tabpfn_nids.flows.live_flow_manager import LiveFlowManager
from tabpfn_nids.inference.live_inference_engine import (
    LiveDetectionResult,
    LiveInferenceEngine,
)

logger = logging.getLogger(__name__)


@dataclass
class LivePipelineConfig:
    """Consolidated configuration for the live monitoring pipeline."""

    interface: str | None = None
    tshark_path: str = "tshark"
    packet_queue_size: int = 10_000
    bpf_filter: str | None = None
    window_size_seconds: float = 10.0
    step_seconds: float = 5.0
    flow_timeout_seconds: float = 10.0
    idle_timeout_seconds: float = 10.0
    max_workers: int = 2
    inference_queue_size: int = 100
    status_broadcast_interval: float = 1.0


class LivePipelineCoordinator:
    """Coordinates packet capture, flow aggregation, windowing, and inference."""

    def __init__(
        self,
        config: LivePipelineConfig | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.config = config or LivePipelineConfig()
        self.event_bus = event_bus or EventBus()

        # 1. Packet capture
        capture_cfg = CaptureConfig(
            interface=self.config.interface,
            tshark_path=self.config.tshark_path,
            packet_queue_size=self.config.packet_queue_size,
            bpf_filter=self.config.bpf_filter,
        )
        self.capture = TSharkCapture(capture_cfg)

        # 2. Flow manager
        self.flow_manager = LiveFlowManager(
            flow_timeout_seconds=self.config.flow_timeout_seconds,
            idle_timeout_seconds=self.config.idle_timeout_seconds,
        )

        # 3. Window manager
        self.window_manager = WindowManager(
            flow_manager=self.flow_manager,
            window_size_seconds=self.config.window_size_seconds,
            step_seconds=self.config.step_seconds,
            queue_size=self.config.inference_queue_size,
        )

        # 4. Inference engine
        self.inference_engine = LiveInferenceEngine(
            window_queue=self.window_manager.window_queue,
            max_workers=self.config.max_workers,
            result_callback=self._on_detection_result,
        )

        # Pipeline state and tasks
        self._is_running = False
        self._packet_consumer_task: asyncio.Task[None] | None = None
        self._status_broadcast_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    @property
    def is_running(self) -> bool:
        return self._is_running

    def get_status(self) -> dict[str, Any]:
        """Aggregate current real-time system metrics for the dashboard and API."""
        cap_metrics = self.capture.get_metrics()
        win_metrics = self.window_manager.metrics.to_dict()
        inf_metrics = self.inference_engine.metrics.to_dict()

        return {
            "running": self.is_running,
            "interface": cap_metrics["interface"] or (self.config.interface or "Unknown"),
            "packets_captured": cap_metrics["packets_captured"],
            "packets_processed": cap_metrics["packets_processed"],
            "packets_dropped": cap_metrics["packets_dropped"],
            "active_flows": self.flow_manager.active_flow_count,
            "windows_processed": inf_metrics["total_windows_processed"],
            "attack_count": inf_metrics["total_attacks_detected"],
            "job_queue_size": self.window_manager.window_queue.qsize(),
            "windows_dropped": win_metrics["windows_dropped"],
            "inference_jobs_failed": inf_metrics["inference_jobs_failed"],
            "uptime_seconds": cap_metrics["uptime_seconds"],
            "tshark_pid": cap_metrics["tshark_pid"],
            "websocket_subscribers": self.event_bus.subscriber_count,
            "metrics": {
                "avg_inference_latency_ms": inf_metrics["avg_inference_latency_ms"],
                "avg_feature_extraction_latency_ms": inf_metrics["avg_feature_extraction_latency_ms"],
                "avg_e2e_latency_ms": inf_metrics["avg_e2e_latency_ms"],
            },
        }

    async def start(self) -> None:
        """Start the complete live monitoring pipeline."""
        if self._is_running:
            raise RuntimeError("Live monitoring pipeline is already running.")

        logger.info("Starting TabPFN-NIDS Live Pipeline...")
        self._stop_event.clear()

        # 1. Start inference engine
        self.inference_engine.start()

        # 2. Start window manager
        self.window_manager.start()

        # 3. Start packet capture
        await self.capture.start()

        # 4. Start packet consumer task
        self._packet_consumer_task = asyncio.create_task(
            self._consume_packet_stream(), name="live-packet-consumer"
        )

        # 5. Start periodic status broadcaster
        self._status_broadcast_task = asyncio.create_task(
            self._status_broadcaster(), name="live-status-broadcaster"
        )

        self._is_running = True
        logger.info("TabPFN-NIDS Live Pipeline started successfully.")

    async def _consume_packet_stream(self) -> None:
        """Asynchronously pulls packets from TSharkCapture and routes to WindowManager."""
        packet_queue = self.capture.packet_queue

        try:
            while not self._stop_event.is_set():
                try:
                    pkt = await asyncio.wait_for(packet_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                self.window_manager.record_packet(pkt)
                self.capture.metrics.packets_processed += 1
                packet_queue.task_done()

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("Error in packet consumer stream: %s", exc)

    async def _status_broadcaster(self) -> None:
        """Periodically broadcast capture status to connected WebSockets."""
        interval = self.config.status_broadcast_interval
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(interval)
                if self._stop_event.is_set():
                    break
                status = self.get_status()
                await self.event_bus.broadcast_status(status)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("Status broadcaster stopped: %s", exc)

    async def _on_detection_result(self, result: LiveDetectionResult) -> None:
        """Callback invoked whenever an inference worker finishes a window."""
        logger.info(
            "Window %s detection: %s (confidence: %.2f%%, flows: %d, packets: %d, attacks: %d)",
            result.window_id,
            result.prediction,
            result.confidence * 100.0,
            result.flows_analyzed,
            result.packets_analyzed,
            result.attack_count,
        )
        await self.event_bus.broadcast_detection(result.to_dict())

    async def stop(self) -> dict[str, Any]:
        """Gracefully stop packet capture, drain queues, finalize inference, and close."""
        if not self._is_running:
            return self.get_status()

        logger.info("Stopping TabPFN-NIDS Live Pipeline...")
        self._stop_event.set()

        # 1. Stop TShark capture
        await self.capture.stop()

        # 2. Drain packet queue into WindowManager
        packet_queue = self.capture.packet_queue
        drained_packets = 0
        while not packet_queue.empty():
            try:
                pkt = packet_queue.get_nowait()
                self.window_manager.record_packet(pkt)
                self.capture.metrics.packets_processed += 1
                packet_queue.task_done()
                drained_packets += 1
            except asyncio.QueueEmpty:
                break
        if drained_packets > 0:
            logger.info("Drained %d pending packets into flow manager.", drained_packets)

        # Cancel packet consumer
        if self._packet_consumer_task and not self._packet_consumer_task.done():
            self._packet_consumer_task.cancel()

        # 3. Stop window manager (flushes active flows into final window)
        await self.window_manager.stop()

        # 4. Wait for remaining windows to complete inference (max 10s)
        try:
            if not self.window_manager.window_queue.empty():
                logger.info(
                    "Waiting for %d pending inference jobs to complete...",
                    self.window_manager.window_queue.qsize(),
                )
                await asyncio.wait_for(
                    self.window_manager.window_queue.join(), timeout=10.0
                )
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for inference queue to drain.")
        except Exception:
            pass

        # 5. Stop inference engine
        await self.inference_engine.stop()

        # 6. Stop status broadcaster
        if self._status_broadcast_task and not self._status_broadcast_task.done():
            self._status_broadcast_task.cancel()

        self._is_running = False
        final_status = self.get_status()
        await self.event_bus.broadcast_status(final_status)

        logger.info("TabPFN-NIDS Live Pipeline stopped cleanly.")
        return final_status
