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
    packet_refresh_threshold: int = 100
    packets_since_last_refresh: int = 0
    packets_until_next_refresh: int = 100
    windows_triggered_by_packet_count: int = 0
    windows_triggered_by_timer: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "windows_created": self.windows_created,
            "windows_processed": self.windows_processed,
            "windows_dropped": self.windows_dropped,
            "total_packets_routed": self.total_packets_routed,
            "current_queue_size": self.current_queue_size,
            "packet_refresh_threshold": self.packet_refresh_threshold,
            "packets_since_last_refresh": self.packets_since_last_refresh,
            "packets_until_next_refresh": self.packets_until_next_refresh,
            "windows_triggered_by_packet_count": self.windows_triggered_by_packet_count,
            "windows_triggered_by_timer": self.windows_triggered_by_timer,
        }


class WindowManager:
    """Orchestrates packet-count and sliding time-window segmentation over live flow streams.

    Features:
    - Automatically updates active flow statistics as each packet arrives.
    - Triggers processing refresh every ``packet_refresh_threshold`` packets (default: 100 packets).
    - Preserves periodic timer refresh every ``step_seconds`` (e.g. 5s) so low-volume traffic is not ignored.
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
        packet_refresh_threshold: int = 100,
    ) -> None:
        self.flow_manager = flow_manager
        self.window_size_seconds = max(1.0, float(window_size_seconds))
        self.step_seconds = max(0.5, float(step_seconds))
        self.queue_size = queue_size
        self.packet_refresh_threshold = max(1, int(packet_refresh_threshold))

        self.window_queue: asyncio.Queue[DetectionWindow] = asyncio.Queue(
            maxsize=self.queue_size
        )
        self.metrics = WindowManagerMetrics(
            packet_refresh_threshold=self.packet_refresh_threshold,
            packets_until_next_refresh=self.packet_refresh_threshold,
        )

        self._window_counter = 0
        self._loop_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._window_start_time = time.time()
        self._packets_in_window = 0
        self._packets_since_last_refresh = 0
        self._last_drop_warning = 0.0

    @property
    def is_running(self) -> bool:
        return self._loop_task is not None and not self._loop_task.done()

    @property
    def packets_until_next_refresh(self) -> int:
        return max(0, self.packet_refresh_threshold - self._packets_since_last_refresh)

    def record_packet(self, pkt: PacketRecord) -> None:
        """Route packet to flow manager, update counters, and refresh upon packet threshold."""
        self._packets_in_window += 1
        self._packets_since_last_refresh += 1
        self.metrics.total_packets_routed += 1
        self.metrics.packets_since_last_refresh = self._packets_since_last_refresh
        self.metrics.packets_until_next_refresh = self.packets_until_next_refresh

        self.flow_manager.add_packet(pkt)

        # Trigger processing refresh every packet_refresh_threshold packets (default 100)
        if self._packets_since_last_refresh >= self.packet_refresh_threshold:
            self._packets_since_last_refresh = 0
            self.metrics.packets_since_last_refresh = 0
            self.metrics.packets_until_next_refresh = self.packet_refresh_threshold
            enqueued = self._harvest_and_enqueue_window(reason="packet_count")
            if enqueued:
                self.metrics.windows_triggered_by_packet_count += 1

    def _harvest_and_enqueue_window(self, reason: str = "timer") -> bool:
        """Harvest active and completed flow snapshots and enqueue into detection queue."""
        now = time.time()
        self.flow_manager.check_timeouts(now)
        window_flows = self.flow_manager.get_window_snapshot(only_updated=True)
        packets_count = self._packets_in_window
        self._packets_in_window = 0

        if not window_flows and packets_count == 0:
            return False

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

        try:
            self.window_queue.put_nowait(window)
            self.metrics.current_queue_size = self.window_queue.qsize()
            logger.debug(
                "Window %s enqueued via %s (%d flows, %d packets)",
                window_id,
                reason,
                len(window_flows),
                packets_count,
            )
            return True
        except asyncio.QueueFull:
            self.metrics.windows_dropped += 1
            if now - self._last_drop_warning > 5.0:
                logger.warning(
                    "Inference window queue is full (%d)! Dropping window %s. "
                    "Consider increasing inference max_workers or reducing traffic volume.",
                    self.queue_size,
                    window_id,
                )
                self._last_drop_warning = now
            return False

    def start(self) -> None:
        """Start the background sliding window timer loop."""
        if self.is_running:
            return

        self._stop_event.clear()
        self._window_counter = 0
        self._window_start_time = time.time()
        self._packets_in_window = 0
        self._packets_since_last_refresh = 0
        self._loop_task = asyncio.create_task(
            self._window_timer_loop(), name="window-timer-loop"
        )
        logger.info(
            "WindowManager started (packet_threshold=%d, window=%.1fs, step=%.1fs, max_queue=%d)",
            self.packet_refresh_threshold,
            self.window_size_seconds,
            self.step_seconds,
            self.queue_size,
        )

    async def _window_timer_loop(self) -> None:
        """Periodically check flow timeouts and refresh active flows on timer interval."""
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(self.step_seconds)
                if self._stop_event.is_set():
                    break
                enqueued = self._harvest_and_enqueue_window(reason="timer")
                if enqueued:
                    self.metrics.windows_triggered_by_timer += 1
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("Error in WindowManager timer loop: %s", exc)

    async def stop(self) -> list[DetectionWindow]:
        """Stop window loop and flush any remaining flows into a final window."""
        self._stop_event.set()
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except (asyncio.CancelledError, Exception):
                pass

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
