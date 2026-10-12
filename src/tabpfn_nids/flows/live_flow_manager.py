"""Live bidirectional network flow manager for streaming packet capture."""

from __future__ import annotations

import copy
import logging
import time
from typing import Any

from tabpfn_nids.flows.flow_builder import (
    FlowBuilder,
    FlowRecord,
    _canonical_flow_key,
)
from tabpfn_nids.pcap.extractor import PacketRecord

logger = logging.getLogger(__name__)


class LiveFlowManager:
    """Aggregates streaming packet records into canonical bidirectional flows.

    Uses the exact same flow construction logic as the offline PCAP pipeline
    (canonical 5-tuple, directional statistics, TCP flag tracking, packet lengths,
    and inter-arrival times), guaranteeing feature compatibility with the trained model.
    """

    def __init__(
        self,
        flow_timeout_seconds: float = 10.0,
        idle_timeout_seconds: float = 10.0,
        max_packets_per_flow: int = 1_000_000,
    ) -> None:
        self.flow_timeout = flow_timeout_seconds
        self.idle_timeout = idle_timeout_seconds
        self.max_packets_per_flow = max_packets_per_flow

        self._builder = FlowBuilder(
            flow_timeout=self.flow_timeout,
            idle_timeout=self.idle_timeout,
            max_packets_per_flow=self.max_packets_per_flow,
        )

        # Track keys updated since last window harvest
        self._updated_in_window: set[tuple[str, int, str, int, int]] = set()
        self._total_packets_ingested = 0
        self._last_packet_timestamp: float | None = None

    @property
    def active_flow_count(self) -> int:
        """Current number of active (unfinalized) flows."""
        return self._builder.active_flow_count

    @property
    def total_packets_ingested(self) -> int:
        return self._total_packets_ingested

    def add_packet(self, pkt: PacketRecord) -> None:
        """Process a single incoming packet."""
        self._total_packets_ingested += 1
        self._last_packet_timestamp = pkt.timestamp

        key = _canonical_flow_key(
            pkt.src_ip, pkt.src_port, pkt.dst_ip, pkt.dst_port, pkt.protocol
        )
        self._updated_in_window.add(key)
        self._builder.add_packet(pkt)

    def add_packets(self, packets: list[PacketRecord]) -> None:
        """Process a batch of incoming packets."""
        for pkt in packets:
            self.add_packet(pkt)

    def check_timeouts(self, current_time: float | None = None) -> list[FlowRecord]:
        """Trigger timeout check on active flows and return newly expired flows.

        Args:
            current_time: Reference timestamp. Defaults to now or last packet timestamp.

        Returns:
            List of flows finalized due to timeout.
        """
        ts = current_time or self._last_packet_timestamp or time.time()
        self._builder._check_timeouts(ts)
        return self._builder.get_completed()

    def get_window_snapshot(self, only_updated: bool = True) -> list[FlowRecord]:
        """Harvest flow snapshots for the current detection window.

        Policy for flows spanning multiple windows:
        - Completed flows (closed by FIN/RST or timeout) are always returned and cleared.
        - Active flows that received traffic during this window interval are snapshotted
          with finalized derived stats (duration, total packets, total bytes).
        - Inactive flows (no packets in this window) are omitted to avoid duplicate evaluation.

        Args:
            only_updated: If True, only snapshot active flows updated in this window.

        Returns:
            Combined list of FlowRecord objects representing traffic in this window.
        """
        # 1. Collect flows completed during this interval
        completed_flows = self._builder.get_completed()

        # 2. Snapshot active flows
        active_snapshots: list[FlowRecord] = []
        for key, (flow, _, _) in self._builder._active.items():
            if only_updated and key not in self._updated_in_window:
                continue

            snap = copy.deepcopy(flow)
            snap.duration = max(0.0, snap.end_time - snap.start_time)
            snap.total_packets = snap.fwd_packets + snap.bwd_packets
            snap.total_bytes = snap.fwd_bytes + snap.bwd_bytes
            active_snapshots.append(snap)

        # Reset updated tracker for next window
        self._updated_in_window.clear()

        return completed_flows + active_snapshots

    def flush_all(self) -> list[FlowRecord]:
        """Force finalize all active flows (e.g. on shutdown)."""
        self._updated_in_window.clear()
        return self._builder.flush_all()
