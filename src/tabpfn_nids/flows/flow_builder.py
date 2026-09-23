"""Bidirectional network flow reconstruction from packets.

Converts a stream of PacketRecord objects into bidirectional network flows.
A flow is defined by a canonical 5-tuple:
    (min_ip, min_port, max_ip, max_port, protocol)
where "min" and "max" ensure the forward and reverse directions of the same
conversation map to the same flow key.

Flows are separated by timeouts:
    - **flow_timeout**: Maximum gap between any two packets in the flow.
    - **idle_timeout**: Maximum inactivity before the flow is closed.

The builder is streaming: packets are fed in batches, and completed flows
are flushed when their timeout expires. This keeps memory bounded even for
captures with millions of concurrent flows.

Output: each flow is a dict with metadata, packet lists, and directional
information suitable for downstream feature engineering.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tabpfn_nids.pcap.extractor import PacketRecord

logger = logging.getLogger(__name__)

# TCP flag constants
TCP_FIN = 0x01
TCP_SYN = 0x02
TCP_RST = 0x04
TCP_PSH = 0x08
TCP_ACK = 0x10
TCP_URG = 0x20


@dataclass
class FlowRecord:
    """A single bidirectional network flow.

    The flow_id is a deterministic hash of the canonical 5-tuple and start
    time, so the same flow always gets the same ID across runs.
    """

    flow_id: str
    src_ip: str               # IP that sent the first packet (initiator)
    dst_ip: str               # IP that received the first packet
    src_port: int
    dst_port: int
    protocol: int             # IP protocol number
    protocol_name: str        # "tcp", "udp", "icmp"

    start_time: float         # Epoch time of first packet
    end_time: float           # Epoch time of last packet
    duration: float           # end_time - start_time

    # Directional packet/byte counts
    fwd_packets: int = 0      # Packets in initiator → responder direction
    bwd_packets: int = 0      # Packets in responder → initiator direction
    fwd_bytes: int = 0        # Bytes initiator → responder
    bwd_bytes: int = 0        # Bytes responder → initiator
    total_packets: int = 0
    total_bytes: int = 0

    # Packet lengths (for statistical features)
    fwd_packet_lengths: list[int] = field(default_factory=list)
    bwd_packet_lengths: list[int] = field(default_factory=list)

    # Packet timestamps (for IAT features)
    fwd_timestamps: list[float] = field(default_factory=list)
    bwd_timestamps: list[float] = field(default_factory=list)

    # TCP flags (aggregate counts)
    fwd_syn: int = 0
    fwd_ack: int = 0
    fwd_fin: int = 0
    fwd_rst: int = 0
    fwd_psh: int = 0
    fwd_urg: int = 0
    bwd_syn: int = 0
    bwd_ack: int = 0
    bwd_fin: int = 0
    bwd_rst: int = 0
    bwd_psh: int = 0
    bwd_urg: int = 0

    # Payload
    fwd_payload_bytes: int = 0
    bwd_payload_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to a flat dict for DataFrame construction.

        Packet-level lists (lengths, timestamps) are excluded from the flat
        representation — they are consumed by the feature engineering stage
        and not stored in the intermediate parquet.
        """
        return {
            "flow_id": self.flow_id,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
            "protocol_name": self.protocol_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "fwd_packets": self.fwd_packets,
            "bwd_packets": self.bwd_packets,
            "fwd_bytes": self.fwd_bytes,
            "bwd_bytes": self.bwd_bytes,
            "total_packets": self.total_packets,
            "total_bytes": self.total_bytes,
            "fwd_syn": self.fwd_syn,
            "fwd_ack": self.fwd_ack,
            "fwd_fin": self.fwd_fin,
            "fwd_rst": self.fwd_rst,
            "fwd_psh": self.fwd_psh,
            "fwd_urg": self.fwd_urg,
            "bwd_syn": self.bwd_syn,
            "bwd_ack": self.bwd_ack,
            "bwd_fin": self.bwd_fin,
            "bwd_rst": self.bwd_rst,
            "bwd_psh": self.bwd_psh,
            "bwd_urg": self.bwd_urg,
            "fwd_payload_bytes": self.fwd_payload_bytes,
            "bwd_payload_bytes": self.bwd_payload_bytes,
            # Encode packet-level lists as strings for parquet
            "fwd_packet_lengths": ",".join(str(x) for x in self.fwd_packet_lengths),
            "bwd_packet_lengths": ",".join(str(x) for x in self.bwd_packet_lengths),
            "fwd_timestamps": ",".join(f"{x:.6f}" for x in self.fwd_timestamps),
            "bwd_timestamps": ",".join(f"{x:.6f}" for x in self.bwd_timestamps),
        }


def _canonical_flow_key(
    src_ip: str, src_port: int, dst_ip: str, dst_port: int, protocol: int
) -> tuple[str, int, str, int, int]:
    """Create a canonical (direction-independent) flow key.

    The key is ordered so that (A→B) and (B→A) produce the same tuple.
    Ordering is by IP string, then by port.

    Args:
        src_ip: Source IP.
        src_port: Source port.
        dst_ip: Destination IP.
        dst_port: Destination port.
        protocol: IP protocol number.

    Returns:
        A 5-tuple (ip_a, port_a, ip_b, port_b, protocol) where ip_a ≤ ip_b
        (lexicographically), with ports following their respective IP.
    """
    if (src_ip, src_port) <= (dst_ip, dst_port):
        return (src_ip, src_port, dst_ip, dst_port, protocol)
    return (dst_ip, dst_port, src_ip, src_port, protocol)


def _generate_flow_id(
    key: tuple[str, int, str, int, int], start_time: float
) -> str:
    """Generate a deterministic flow ID from the canonical key + start time.

    Args:
        key: The canonical 5-tuple.
        start_time: Epoch time of the first packet in the flow.

    Returns:
        A short hex hash suitable for use as a flow identifier.
    """
    raw = f"{key[0]}:{key[1]}-{key[2]}:{key[3]}-{key[4]}-{start_time:.6f}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class FlowBuilder:
    """Reconstruct bidirectional network flows from a packet stream.

    The builder maintains a table of active flows keyed by canonical 5-tuple.
    Each incoming packet either extends an existing flow or starts a new one.
    Flows are closed when:
        - The idle timeout expires (no packet for ``idle_timeout`` seconds)
        - A TCP FIN or RST is seen
        - ``flush_all()`` is called at the end of a PCAP

    Args:
        flow_timeout: Maximum total duration of a flow before forced close.
        idle_timeout: Maximum gap between consecutive packets.
        max_packets_per_flow: Safety cap on packets per flow.
    """

    def __init__(
        self,
        flow_timeout: float = 120.0,
        idle_timeout: float = 60.0,
        max_packets_per_flow: int = 1_000_000,
    ) -> None:
        self.flow_timeout = flow_timeout
        self.idle_timeout = idle_timeout
        self.max_packets_per_flow = max_packets_per_flow

        # Active flows: key → (FlowRecord, last_packet_time, initiator_key)
        self._active: dict[
            tuple[str, int, str, int, int],
            tuple[FlowRecord, float, tuple[str, int]],
        ] = {}
        self._completed: list[FlowRecord] = []
        self._total_packets = 0

    def _is_forward(
        self, pkt: PacketRecord, initiator_key: tuple[str, int]
    ) -> bool:
        """Check if a packet is in the forward (initiator→responder) direction.

        Args:
            pkt: The packet record.
            initiator_key: (src_ip, src_port) of the flow initiator.

        Returns:
            True if the packet's source matches the initiator.
        """
        return (pkt.src_ip, pkt.src_port) == initiator_key

    def _close_flow(self, flow: FlowRecord) -> None:
        """Finalise a flow and move it to the completed list."""
        flow.duration = flow.end_time - flow.start_time
        flow.total_packets = flow.fwd_packets + flow.bwd_packets
        flow.total_bytes = flow.fwd_bytes + flow.bwd_bytes
        self._completed.append(flow)

    def _check_timeouts(self, current_time: float) -> None:
        """Close all flows that have exceeded their timeout.

        Args:
            current_time: The timestamp of the current packet.
        """
        expired_keys = []
        for key, (flow, last_time, _) in self._active.items():
            idle_expired = (current_time - last_time) > self.idle_timeout
            flow_expired = (current_time - flow.start_time) > self.flow_timeout
            if idle_expired or flow_expired:
                expired_keys.append(key)

        for key in expired_keys:
            flow, _, _ = self._active.pop(key)
            self._close_flow(flow)

    def add_packet(self, pkt: PacketRecord) -> None:
        """Add a single packet to the flow table.

        Args:
            pkt: A normalised packet record.
        """
        self._total_packets += 1
        key = _canonical_flow_key(
            pkt.src_ip, pkt.src_port, pkt.dst_ip, pkt.dst_port, pkt.protocol
        )

        # Check timeouts periodically (every 10000 packets for performance)
        if self._total_packets % 10000 == 0:
            self._check_timeouts(pkt.timestamp)

        if key in self._active:
            flow, last_time, initiator_key = self._active[key]

            # Check if this packet exceeds the idle/flow timeout
            if (pkt.timestamp - last_time > self.idle_timeout or
                    pkt.timestamp - flow.start_time > self.flow_timeout):
                # Close old flow, start new one
                self._close_flow(flow)
                del self._active[key]
                # Fall through to create new flow
            elif flow.total_packets >= self.max_packets_per_flow:
                # Safety cap reached
                self._close_flow(flow)
                del self._active[key]
            else:
                # Add packet to existing flow
                flow.end_time = pkt.timestamp
                is_fwd = self._is_forward(pkt, initiator_key)

                if is_fwd:
                    flow.fwd_packets += 1
                    flow.fwd_bytes += pkt.length
                    flow.fwd_packet_lengths.append(pkt.length)
                    flow.fwd_timestamps.append(pkt.timestamp)
                    flow.fwd_payload_bytes += pkt.payload_length
                    if pkt.protocol_name == "tcp":
                        flow.fwd_syn += bool(pkt.tcp_flags & TCP_SYN)
                        flow.fwd_ack += bool(pkt.tcp_flags & TCP_ACK)
                        flow.fwd_fin += bool(pkt.tcp_flags & TCP_FIN)
                        flow.fwd_rst += bool(pkt.tcp_flags & TCP_RST)
                        flow.fwd_psh += bool(pkt.tcp_flags & TCP_PSH)
                        flow.fwd_urg += bool(pkt.tcp_flags & TCP_URG)
                else:
                    flow.bwd_packets += 1
                    flow.bwd_bytes += pkt.length
                    flow.bwd_packet_lengths.append(pkt.length)
                    flow.bwd_timestamps.append(pkt.timestamp)
                    flow.bwd_payload_bytes += pkt.payload_length
                    if pkt.protocol_name == "tcp":
                        flow.bwd_syn += bool(pkt.tcp_flags & TCP_SYN)
                        flow.bwd_ack += bool(pkt.tcp_flags & TCP_ACK)
                        flow.bwd_fin += bool(pkt.tcp_flags & TCP_FIN)
                        flow.bwd_rst += bool(pkt.tcp_flags & TCP_RST)
                        flow.bwd_psh += bool(pkt.tcp_flags & TCP_PSH)
                        flow.bwd_urg += bool(pkt.tcp_flags & TCP_URG)

                flow.total_packets = flow.fwd_packets + flow.bwd_packets
                flow.total_bytes = flow.fwd_bytes + flow.bwd_bytes
                self._active[key] = (flow, pkt.timestamp, initiator_key)

                # Check for TCP FIN/RST — close the flow
                if pkt.protocol_name == "tcp" and (pkt.tcp_flags & (TCP_FIN | TCP_RST)):
                    self._close_flow(flow)
                    del self._active[key]

                return

        # Create new flow — this packet is the initiator
        initiator_key = (pkt.src_ip, pkt.src_port)
        flow_id = _generate_flow_id(key, pkt.timestamp)

        flow = FlowRecord(
            flow_id=flow_id,
            src_ip=pkt.src_ip,
            dst_ip=pkt.dst_ip,
            src_port=pkt.src_port,
            dst_port=pkt.dst_port,
            protocol=pkt.protocol,
            protocol_name=pkt.protocol_name,
            start_time=pkt.timestamp,
            end_time=pkt.timestamp,
            duration=0.0,
            fwd_packets=1,
            fwd_bytes=pkt.length,
            fwd_packet_lengths=[pkt.length],
            fwd_timestamps=[pkt.timestamp],
            fwd_payload_bytes=pkt.payload_length,
        )

        # TCP flags for first packet
        if pkt.protocol_name == "tcp":
            flow.fwd_syn = bool(pkt.tcp_flags & TCP_SYN)
            flow.fwd_ack = bool(pkt.tcp_flags & TCP_ACK)
            flow.fwd_fin = bool(pkt.tcp_flags & TCP_FIN)
            flow.fwd_rst = bool(pkt.tcp_flags & TCP_RST)
            flow.fwd_psh = bool(pkt.tcp_flags & TCP_PSH)
            flow.fwd_urg = bool(pkt.tcp_flags & TCP_URG)

        self._active[key] = (flow, pkt.timestamp, initiator_key)

    def add_packets(self, packets: list[PacketRecord]) -> None:
        """Add a batch of packets.

        Args:
            packets: List of PacketRecord to process.
        """
        for pkt in packets:
            self.add_packet(pkt)

    def flush_all(self) -> list[FlowRecord]:
        """Close all active flows and return all completed flows.

        This should be called after the last packet batch to ensure no
        flows remain open.

        Returns:
            All completed flows (including those closed by timeout).
        """
        for key in list(self._active.keys()):
            flow, _, _ = self._active.pop(key)
            self._close_flow(flow)

        result = self._completed
        self._completed = []
        return result

    def get_completed(self) -> list[FlowRecord]:
        """Return flows completed so far without closing active ones.

        Returns:
            A list of completed flows. The internal list is cleared.
        """
        result = self._completed
        self._completed = []
        return result

    @property
    def active_flow_count(self) -> int:
        """Number of currently active (open) flows."""
        return len(self._active)

    @property
    def total_packets_processed(self) -> int:
        """Total packets fed to this builder."""
        return self._total_packets


def flows_to_dataframe(flows: list[FlowRecord]) -> pd.DataFrame:
    """Convert a list of FlowRecord to a pandas DataFrame.

    Args:
        flows: Completed flow records.

    Returns:
        A DataFrame with one row per flow.
    """
    if not flows:
        return pd.DataFrame()
    return pd.DataFrame([f.to_dict() for f in flows])


def save_flows_parquet(
    flows: list[FlowRecord],
    output_path: Path | str,
    append: bool = False,
) -> int:
    """Save flow records to a Parquet file.

    Args:
        flows: Completed flow records.
        output_path: Destination path.
        append: If True, append to existing file. If False, overwrite.

    Returns:
        Number of flows written.
    """
    if not flows:
        logger.warning("No flows to save")
        return 0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = flows_to_dataframe(flows)
    table = pa.Table.from_pandas(df)

    if append and output_path.exists():
        existing = pq.read_table(output_path)
        table = pa.concat_tables([existing, table])

    pq.write_table(table, output_path, compression="snappy")
    logger.info("Saved %d flows to %s", len(df), output_path)
    return len(df)


def extract_flows_from_pcap(
    pcap_path: Path | str,
    output_path: Path | str,
    backend: str = "scapy",
    flow_timeout: float = 120.0,
    idle_timeout: float = 60.0,
    batch_size: int = 50_000,
    tshark_path: str = "tshark",
) -> dict[str, Any]:
    """End-to-end: read a PCAP and produce a Parquet of flows.

    This is the primary entry point for PCAP → flow extraction. It:
    1. Reads packets in streaming batches
    2. Reconstructs bidirectional flows with timeout handling
    3. Saves the result as a Parquet file

    Args:
        pcap_path: Input PCAP file.
        output_path: Output Parquet file for flows.
        backend: "scapy" or "tshark".
        flow_timeout: Maximum flow duration.
        idle_timeout: Maximum inactivity gap.
        batch_size: Packets per extraction batch.
        tshark_path: Path to tshark binary.

    Returns:
        A summary dict with extraction statistics.
    """
    from tabpfn_nids.pcap.extractor import extract_packets

    pcap_path = Path(pcap_path)
    started = time.time()

    builder = FlowBuilder(
        flow_timeout=flow_timeout,
        idle_timeout=idle_timeout,
    )

    total_packets = 0
    for batch in extract_packets(pcap_path, backend=backend,
                                  tshark_path=tshark_path, batch_size=batch_size):
        builder.add_packets(batch)
        total_packets += len(batch)
        logger.debug(
            "Processed %d packets, %d active flows",
            total_packets, builder.active_flow_count,
        )

    all_flows = builder.flush_all()
    flow_count = save_flows_parquet(all_flows, output_path)

    elapsed = time.time() - started
    summary = {
        "pcap_file": pcap_path.name,
        "total_packets": total_packets,
        "total_flows": flow_count,
        "active_flows_at_end": builder.active_flow_count,
        "elapsed_seconds": round(elapsed, 2),
        "packets_per_second": round(total_packets / max(elapsed, 0.001), 1),
    }

    logger.info(
        "Extracted %d flows from %d packets in %.1fs (%.0f pkt/s)",
        flow_count, total_packets, elapsed,
        total_packets / max(elapsed, 0.001),
    )
    return summary
