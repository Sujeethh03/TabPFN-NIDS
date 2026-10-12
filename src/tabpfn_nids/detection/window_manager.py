"""Sliding window detection manager for real-time intrusion monitoring."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from tabpfn_nids.flows.flow_builder import FlowRecord
from tabpfn_nids.flows.live_flow_manager import LiveFlowManager
from tabpfn_nids.pcap.extractor import PacketRecord

logger = logging.getLogger(__name__)


@dataclass
class DetectionWindow:
    """A batch of flows corresponding to one detection window."""

    window_id: str
    window_start: float
    window_end: float
    flows: list[FlowRecord]
    packets_analyzed: int
    created_at: float = field(default_factory=time.time)

    @property
    def flow_count(self) -> int:
        return len(self.flows)


@dataclass
class WindowManagerMetrics:
    """Metrics tracking window generation and backpressure."""

    windows_created: int = 0
    windows_processed: int = 0
    windows_dropped: int = 0
    total_packets_routed: int = 0
    current_queue_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "windows_created": self.windows_created,
            "windows_processed": self.windows_processed,
            "windows_dropped": self.windows_dropped,
            "total_packets_routed": self.total_packets_routed,
            "current_queue_size": self.current_queue_size,
        }


class WindowManager:
    """Orchestrates sliding time-window segmentation over live flow streams.

    Features:
    - Runs a periodic timer every ``step_seconds`` (e.g. 5s) while packet capture runs continuously.
    - Collects active snapshots and completed flows from ``LiveFlowManager``.
    - Enqueues completed ``DetectionWindow`` jobs into a bounded feature/inference queue.
    - Protects the system under heavy load by dropping old windows if inference queue is full,
      logging rate-limited backpressure warnings without crashing or stopping packet capture.
    """

    def __init__(
        self,
        flow_manager: LiveFlowManager,
        window_size_seconds: float = 10.0,
        step_seconds: float = 5.0,
        queue_size: int = 100,
    ) -> None:
        self.flow_manager = flow_manager
        self.window_size_seconds = max(1.0, float(window_size_seconds))
        self.step_seconds = max(0.5, float(step_seconds))
        self.queue_size = queue_size

        self.window_queue: asyncio.Queue[DetectionWindow] = asyncio.Queue(
            maxsize=self.queue_size
        )
        self.metrics = WindowManagerMetrics()

        self._window_counter = 0
        self._loop_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._window_start_time = time.time()
        self._packets_in_window = 0

    @property
    def is_running(self) -> bool:
        return self._loop_task is not None and not self._loop_task.done()

    def record_packet(self, pkt: PacketRecord) -> None:
        """Route packet to flow manager and record window packet count."""
        self._packets_in_window += 1
        self.metrics.total_packets_routed += 1
        self.flow_manager.add_packet(pkt)

    def start(self) -> None:
        """Start the background sliding window timer loop."""
        if self.is_running:
            return

        self._stop_event.clear()
        self._window_counter = 0
        self._window_start_time = time.time()
        self._packets_in_window = 0
        self._loop_task = asyncio.create_task(
            self._window_timer_loop(), name="window-timer-loop"
        )
        logger.info(
            "WindowManager started (window=%.1fs, step=%.1fs, max_queue=%d)",
            self.window_size_seconds,
            self.step_seconds,
            self.queue_size,
        )

    async def _window_timer_loop(self) -> None:
        """Periodically trigger window finalization every step_seconds."""
        last_drop_warning = 0.0

        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(self.step_seconds)
                if self._stop_event.is_set():
                    break

                # Timeout checks on active flows
                now = time.time()
                self.flow_manager.check_timeouts(now)

                # Harvest flows for this window
                window_flows = self.flow_manager.get_window_snapshot(only_updated=True)
                packets_count = self._packets_in_window
                self._packets_in_window = 0

                # Even if 0 packets arrived, if completed flows timed out, process them
                if not window_flows and packets_count == 0:
                    continue

                self._window_counter += 1
                window_id = f"win-{self._window_counter:05d}"
                window = DetectionWindow(
                    window_id=window_id,
                    window_start=now - self.window_size_seconds,
                    window_end=now,
                    flows=window_flows,
                    packets_analyzed=packets_count,
                )
                self.metrics.windows_created += 1

                # Enqueue with bounded backpressure handling
                try:
                    self.window_queue.put_nowait(window)
                    self.metrics.current_queue_size = self.window_queue.qsize()
                    logger.debug(
                        "Window %s enqueued (%d flows, %d packets)",
                        window_id,
                        len(window_flows),
                        packets_count,
                    )
                except asyncio.QueueFull:
                    self.metrics.windows_dropped += 1
                    if now - last_drop_warning > 5.0:
                        logger.warning(
                            "Inference window queue is full (%d)! Dropping window %s. "
                            "Consider increasing inference max_workers or reducing traffic volume.",
                            self.queue_size,
                            window_id,
                        )
                        last_drop_warning = now

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("Error in WindowManager timer loop: %s", exc)

    async def stop(self) -> list[DetectionWindow]:
        """Stop window loop and flush any remaining flows into a final window."""
        self._stop_event.set()
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()

        # Flush any remaining flows
        remaining_flows = self.flow_manager.flush_all()
        flushed_windows: list[DetectionWindow] = []

        if remaining_flows:
            self._window_counter += 1
            now = time.time()
            final_win = DetectionWindow(
                window_id=f"win-final-{self._window_counter:05d}",
                window_start=now - self.window_size_seconds,
                window_end=now,
                flows=remaining_flows,
                packets_analyzed=self._packets_in_window,
            )
            flushed_windows.append(final_win)
            try:
                self.window_queue.put_nowait(final_win)
            except asyncio.QueueFull:
                self.metrics.windows_dropped += 1

        self.metrics.current_queue_size = self.window_queue.qsize()
        logger.info(
            "WindowManager stopped. Created: %d, Dropped: %d",
            self.metrics.windows_created,
            self.metrics.windows_dropped,
        )
        return flushed_windows
