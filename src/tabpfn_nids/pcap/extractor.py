"""Packet-level extraction from PCAP files.

Reads raw PCAP files and yields normalised packet records suitable for the
flow builder. This is a streaming interface — packets are yielded one at a
time or in configurable batches, so multi-GB PCAPs never need to be loaded
entirely into RAM.

Two backends are supported:
    - **scapy**: Pure Python. Reliable, pip-installable, slower on large files.
    - **tshark**: CLI-based. Fast C parsing, requires Wireshark installation.

The extractor produces ``PacketRecord`` namedtuples with a fixed schema
regardless of backend, so the flow builder does not need to know which
tool produced the packets.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PacketRecord:
    """A normalised packet record extracted from a PCAP.

    This is the interface between the extractor and the flow builder.
    Every field must be populated regardless of which backend produced it.
    """

    timestamp: float          # Epoch seconds (float for sub-second precision)
    src_ip: str               # Source IP address (v4 or v6)
    dst_ip: str               # Destination IP address
    src_port: int             # Source port (0 for ICMP)
    dst_port: int             # Destination port (0 for ICMP)
    protocol: int             # IP protocol number (6=TCP, 17=UDP, 1=ICMP)
    protocol_name: str        # "tcp", "udp", "icmp", or "other"
    length: int               # Total packet length in bytes
    tcp_flags: int            # TCP flags byte (0 if not TCP)
    payload_length: int       # Application-layer payload length


def _extract_with_scapy(
    path: Path,
    batch_size: int = 50_000,
) -> Generator[list[PacketRecord], None, None]:
    """Extract packets from a PCAP using scapy.

    Yields batches of PacketRecord lists. Each batch contains at most
    ``batch_size`` packets.

    Args:
        path: Path to the PCAP file.
        batch_size: Maximum packets per yielded batch.

    Yields:
        Lists of PacketRecord, each at most ``batch_size`` long.
    """
    from scapy.all import PcapReader, IP, IPv6, TCP, UDP, ICMP

    batch: list[PacketRecord] = []
    total = 0
    errors = 0

    with PcapReader(str(path)) as reader:
        for pkt in reader:
            try:
                ts = float(pkt.time)
                length = len(pkt)

                # Extract IP layer
                if pkt.haslayer(IP):
                    ip_layer = pkt[IP]
                    src_ip = ip_layer.src
                    dst_ip = ip_layer.dst
                    proto_num = ip_layer.proto
                elif pkt.haslayer(IPv6):
                    ip_layer = pkt[IPv6]
                    src_ip = ip_layer.src
                    dst_ip = ip_layer.dst
                    proto_num = ip_layer.nh
                else:
                    # Non-IP packet (ARP, etc.) — skip for flow construction
                    continue

                # Extract transport layer
                src_port = 0
                dst_port = 0
                tcp_flags = 0
                payload_len = 0
                proto_name = "other"

                if pkt.haslayer(TCP):
                    tcp_layer = pkt[TCP]
                    src_port = tcp_layer.sport
                    dst_port = tcp_layer.dport
                    tcp_flags = int(tcp_layer.flags)
                    payload_len = len(tcp_layer.payload) if tcp_layer.payload else 0
                    proto_name = "tcp"
                elif pkt.haslayer(UDP):
                    udp_layer = pkt[UDP]
                    src_port = udp_layer.sport
                    dst_port = udp_layer.dport
                    payload_len = len(udp_layer.payload) if udp_layer.payload else 0
                    proto_name = "udp"
                elif pkt.haslayer(ICMP):
                    # ICMP: use type/code as pseudo-ports for flow keying
                    icmp_layer = pkt[ICMP]
                    src_port = icmp_layer.type
                    dst_port = icmp_layer.code
                    proto_name = "icmp"

                record = PacketRecord(
                    timestamp=ts,
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    src_port=src_port,
                    dst_port=dst_port,
                    protocol=proto_num,
                    protocol_name=proto_name,
                    length=length,
                    tcp_flags=tcp_flags,
                    payload_length=payload_len,
                )
                batch.append(record)
                total += 1

                if len(batch) >= batch_size:
                    logger.debug("Yielding batch of %d packets (total: %d)", len(batch), total)
                    yield batch
                    batch = []

            except Exception as exc:
                errors += 1
                if errors <= 10:
                    logger.warning("Malformed packet #%d: %s", total + errors, exc)

    if batch:
        yield batch

    logger.info(
        "Extracted %d packets from %s (%d errors skipped)",
        total, path.name, errors,
    )


def _extract_with_tshark(
    path: Path,
    tshark_path: str = "tshark",
    batch_size: int = 50_000,
) -> Generator[list[PacketRecord], None, None]:
    """Extract packets from a PCAP using tshark.

    Uses tshark's ``-T fields`` output for fast structured extraction.

    Args:
        path: Path to the PCAP file.
        tshark_path: Path to tshark binary.
        batch_size: Maximum packets per yielded batch.

    Yields:
        Lists of PacketRecord.
    """
    fields = [
        "frame.time_epoch",
        "ip.src", "ip.dst",
        "tcp.srcport", "tcp.dstport",
        "udp.srcport", "udp.dstport",
        "ip.proto",
        "frame.len",
        "tcp.flags",
        "tcp.len",
        "udp.length",
    ]

    cmd = [
        tshark_path, "-r", str(path),
        "-T", "fields",
        *[arg for f in fields for arg in ("-e", f)],
        "-E", "separator=|",
        "-E", "occurrence=f",
    ]

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    except FileNotFoundError:
        logger.error("tshark not found at '%s'", tshark_path)
        raise

    batch: list[PacketRecord] = []
    total = 0
    errors = 0

    for line in proc.stdout:
        try:
            parts = line.strip().split("|")
            if len(parts) < len(fields):
                continue

            ts = float(parts[0]) if parts[0] else 0.0
            src_ip = parts[1] or ""
            dst_ip = parts[2] or ""

            # Determine protocol and ports
            if parts[3]:  # TCP source port present
                src_port = int(parts[3])
                dst_port = int(parts[4]) if parts[4] else 0
                proto_num = 6
                proto_name = "tcp"
                tcp_flags = int(parts[9], 16) if parts[9] else 0
                payload_len = int(parts[10]) if parts[10] else 0
            elif parts[5]:  # UDP source port present
                src_port = int(parts[5])
                dst_port = int(parts[6]) if parts[6] else 0
                proto_num = 17
                proto_name = "udp"
                tcp_flags = 0
                payload_len = int(parts[11]) if parts[11] else 0
            else:
                proto_num = int(parts[7]) if parts[7] else 0
                proto_name = "icmp" if proto_num == 1 else "other"
                src_port = 0
                dst_port = 0
                tcp_flags = 0
                payload_len = 0

            if not src_ip or not dst_ip:
                continue

            length = int(parts[8]) if parts[8] else 0

            record = PacketRecord(
                timestamp=ts,
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=proto_num,
                protocol_name=proto_name,
                length=length,
                tcp_flags=tcp_flags,
                payload_length=payload_len,
            )
            batch.append(record)
            total += 1

            if len(batch) >= batch_size:
                yield batch
                batch = []

        except Exception:
            errors += 1

    proc.wait()

    if batch:
        yield batch

    logger.info(
        "tshark extracted %d packets from %s (%d errors)",
        total, path.name, errors,
    )


def extract_packets(
    path: Path | str,
    backend: str = "scapy",
    tshark_path: str = "tshark",
    batch_size: int = 50_000,
) -> Generator[list[PacketRecord], None, None]:
    """Extract packets from a PCAP file using the specified backend.

    This is the primary entry point. It yields batches of normalised
    PacketRecord objects suitable for the flow builder.

    Args:
        path: Path to the PCAP file.
        backend: "scapy" or "tshark".
        tshark_path: Path to tshark binary.
        batch_size: Maximum packets per batch.

    Yields:
        Lists of PacketRecord.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PCAP file not found: {path}")

    logger.info("Extracting packets from %s (backend=%s)", path.name, backend)

    if backend == "tshark":
        yield from _extract_with_tshark(path, tshark_path, batch_size)
    else:
        yield from _extract_with_scapy(path, batch_size)
