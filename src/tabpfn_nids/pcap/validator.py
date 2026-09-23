"""PCAP file validation and metadata extraction.

Before processing any PCAP file, this module validates its integrity and
collects metadata (packet count, timestamps, protocols, checksums). Corrupted,
empty, or unreadable PCAPs are reported — never silently skipped.

Supports two backends:
    - **scapy**: Pure Python, pip-installable, no external tools required.
    - **tshark**: Faster C-based parsing via the Wireshark CLI tool.

The backend is selected automatically based on availability, or can be forced
via the pipeline config.

Every validated PCAP produces a JSON metadata file in
``data/intermediate/packet_metadata/<pcap_name>.json``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Minimum file size for a valid PCAP (24-byte global header)
MIN_PCAP_SIZE = 24

# PCAP magic numbers (big-endian and little-endian)
PCAP_MAGIC_LE = b"\xd4\xc3\xb2\xa1"
PCAP_MAGIC_BE = b"\xa1\xb2\xc3\xd4"
PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"


@dataclass
class PcapMetadata:
    """Metadata collected from validating a single PCAP file.

    Every field is serialisable to JSON so the result can be written to
    ``data/intermediate/packet_metadata/<name>.json``.
    """

    filename: str
    absolute_path: str
    file_size_bytes: int
    sha256: str
    readable: bool
    valid: bool
    error_message: str | None = None

    packet_count: int = 0
    first_timestamp: float | None = None
    last_timestamp: float | None = None
    duration_seconds: float | None = None

    tcp_packets: int = 0
    udp_packets: int = 0
    icmp_packets: int = 0
    other_packets: int = 0

    protocols: list[str] = field(default_factory=list)
    link_type: str | None = None
    malformed_packets: int = 0

    validation_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-compatible dict."""
        return asdict(self)


def _compute_sha256(path: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file in streaming fashion.

    Args:
        path: Path to the file.
        chunk_size: Read buffer size in bytes.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _check_magic_bytes(path: Path) -> tuple[bool, str | None]:
    """Check if the file starts with a valid PCAP/PCAPNG magic number.

    Args:
        path: Path to the file.

    Returns:
        (is_valid, format_name) — e.g. (True, "pcap") or (False, None).
    """
    with open(path, "rb") as f:
        magic = f.read(4)

    if magic in (PCAP_MAGIC_LE, PCAP_MAGIC_BE):
        return True, "pcap"
    if magic == PCAPNG_MAGIC:
        return True, "pcapng"
    return False, None


def _validate_with_scapy(path: Path, meta: PcapMetadata) -> PcapMetadata:
    """Validate a PCAP using scapy's packet reader.

    Reads packets one by one to stay memory-efficient. Counts protocols,
    extracts timestamps, and detects malformed packets.

    Args:
        path: Path to the PCAP file.
        meta: Pre-initialised metadata object to populate.

    Returns:
        The populated metadata object.
    """
    try:
        from scapy.all import PcapReader, TCP, UDP, ICMP, IP, IPv6
    except ImportError:
        meta.valid = False
        meta.error_message = "scapy is not installed (pip install scapy)"
        return meta

    protocols_seen: set[str] = set()
    timestamps: list[float] = []

    try:
        with PcapReader(str(path)) as reader:
            for pkt in reader:
                meta.packet_count += 1
                ts = float(pkt.time)
                timestamps.append(ts)

                try:
                    if pkt.haslayer(TCP):
                        meta.tcp_packets += 1
                        protocols_seen.add("tcp")
                    elif pkt.haslayer(UDP):
                        meta.udp_packets += 1
                        protocols_seen.add("udp")
                    elif pkt.haslayer(ICMP):
                        meta.icmp_packets += 1
                        protocols_seen.add("icmp")
                    else:
                        meta.other_packets += 1

                    if pkt.haslayer(IP):
                        proto_num = pkt[IP].proto
                        protocols_seen.add(str(proto_num))
                    elif pkt.haslayer(IPv6):
                        proto_num = pkt[IPv6].nh
                        protocols_seen.add(str(proto_num))
                except Exception:
                    meta.malformed_packets += 1

    except Exception as exc:
        meta.valid = False
        meta.error_message = f"scapy read error: {type(exc).__name__}: {exc}"
        return meta

    if meta.packet_count == 0:
        meta.valid = False
        meta.error_message = "PCAP contains zero packets"
        return meta

    meta.first_timestamp = min(timestamps)
    meta.last_timestamp = max(timestamps)
    meta.duration_seconds = meta.last_timestamp - meta.first_timestamp
    meta.protocols = sorted(protocols_seen)
    meta.valid = True
    return meta


def _validate_with_tshark(
    path: Path, meta: PcapMetadata, tshark_path: str = "tshark"
) -> PcapMetadata:
    """Validate a PCAP using tshark's capinfos output.

    Falls back to scapy if tshark is not available.

    Args:
        path: Path to the PCAP file.
        meta: Pre-initialised metadata object to populate.
        tshark_path: Path to the tshark binary.

    Returns:
        The populated metadata object.
    """
    try:
        # Use capinfos for summary statistics (ships with Wireshark)
        result = subprocess.run(
            [tshark_path, "-r", str(path), "-q", "-z", "io,stat,0"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            logger.warning("tshark failed, falling back to scapy: %s", result.stderr)
            return _validate_with_scapy(path, meta)

        # Count packets by protocol using tshark filters
        for proto, display_filter in [("tcp", "tcp"), ("udp", "udp"), ("icmp", "icmp")]:
            count_result = subprocess.run(
                [tshark_path, "-r", str(path), "-Y", display_filter, "-T", "fields",
                 "-e", "frame.number"],
                capture_output=True, text=True, timeout=300,
            )
            count = len(count_result.stdout.strip().split("\n")) if count_result.stdout.strip() else 0
            setattr(meta, f"{proto}_packets", count)

        # Get total packet count and timestamps
        ts_result = subprocess.run(
            [tshark_path, "-r", str(path), "-T", "fields",
             "-e", "frame.time_epoch"],
            capture_output=True, text=True, timeout=600,
        )
        if ts_result.stdout.strip():
            timestamps = [float(t) for t in ts_result.stdout.strip().split("\n") if t.strip()]
            meta.packet_count = len(timestamps)
            meta.first_timestamp = min(timestamps)
            meta.last_timestamp = max(timestamps)
            meta.duration_seconds = meta.last_timestamp - meta.first_timestamp
            meta.other_packets = meta.packet_count - meta.tcp_packets - meta.udp_packets - meta.icmp_packets
            meta.protocols = sorted({
                p for p in ["tcp", "udp", "icmp"]
                if getattr(meta, f"{p}_packets") > 0
            })
            meta.valid = True
        else:
            meta.valid = False
            meta.error_message = "tshark returned no packets"

    except FileNotFoundError:
        logger.info("tshark not found at '%s', falling back to scapy", tshark_path)
        return _validate_with_scapy(path, meta)
    except subprocess.TimeoutExpired:
        meta.valid = False
        meta.error_message = "tshark timed out"
    except Exception as exc:
        meta.valid = False
        meta.error_message = f"tshark error: {type(exc).__name__}: {exc}"

    return meta


def validate_pcap(
    path: Path | str,
    backend: str = "scapy",
    tshark_path: str = "tshark",
) -> PcapMetadata:
    """Validate a single PCAP file and extract its metadata.

    This is the primary entry point. It performs:
    1. File existence and size check
    2. Magic-byte validation (PCAP/PCAPNG header)
    3. SHA-256 checksum computation
    4. Full packet scan for protocol counts, timestamps, and corruption

    Args:
        path: Path to the PCAP file.
        backend: "scapy" or "tshark".
        tshark_path: Path to tshark binary (only used when backend="tshark").

    Returns:
        A PcapMetadata with all fields populated. Check ``.valid`` to see
        if the file passed validation.

    Raises:
        FileNotFoundError: If the file does not exist (this is an error,
            not a validation failure, because the caller asked for a
            specific file).
    """
    path = Path(path)
    started = time.time()

    if not path.is_file():
        raise FileNotFoundError(f"PCAP file not found: {path}")

    meta = PcapMetadata(
        filename=path.name,
        absolute_path=str(path.resolve()),
        file_size_bytes=path.stat().st_size,
        sha256="",
        readable=False,
        valid=False,
    )

    logger.info("Validating %s (%d bytes)...", path.name, meta.file_size_bytes)

    # --- size check -------------------------------------------------------
    if meta.file_size_bytes < MIN_PCAP_SIZE:
        meta.error_message = (
            f"File too small ({meta.file_size_bytes} bytes); "
            f"minimum PCAP size is {MIN_PCAP_SIZE} bytes."
        )
        meta.validation_time_seconds = time.time() - started
        return meta

    # --- magic bytes ------------------------------------------------------
    is_valid_magic, pcap_format = _check_magic_bytes(path)
    if not is_valid_magic:
        meta.error_message = "Invalid PCAP magic bytes — not a valid PCAP/PCAPNG file."
        meta.validation_time_seconds = time.time() - started
        return meta

    meta.link_type = pcap_format
    meta.readable = True

    # --- checksum ---------------------------------------------------------
    meta.sha256 = _compute_sha256(path)

    # --- packet-level validation ------------------------------------------
    if backend == "tshark":
        meta = _validate_with_tshark(path, meta, tshark_path)
    else:
        meta = _validate_with_scapy(path, meta)

    meta.validation_time_seconds = time.time() - started
    logger.info(
        "Validated %s: %s — %d packets, %.1fs duration, %.1fs elapsed",
        path.name,
        "VALID" if meta.valid else f"INVALID ({meta.error_message})",
        meta.packet_count,
        meta.duration_seconds or 0,
        meta.validation_time_seconds,
    )
    return meta


def validate_pcap_directory(
    directory: Path | str,
    output_dir: Path | str | None = None,
    backend: str = "scapy",
    tshark_path: str = "tshark",
) -> list[PcapMetadata]:
    """Validate every PCAP file in a directory.

    Results are written as individual JSON files to ``output_dir`` and also
    returned as a list.

    Args:
        directory: Directory containing .pcap files.
        output_dir: Where to write per-PCAP JSON metadata. Defaults to
            ``data/intermediate/packet_metadata/``.
        backend: "scapy" or "tshark".
        tshark_path: Path to tshark binary.

    Returns:
        A list of PcapMetadata, one per file.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"PCAP directory not found: {directory}")

    pcap_files = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in (".pcap", ".pcapng", ".cap")
    )

    if not pcap_files:
        logger.warning("No PCAP files found in %s", directory)
        return []

    logger.info("Found %d PCAP files in %s", len(pcap_files), directory)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    results: list[PcapMetadata] = []
    for pcap_path in pcap_files:
        meta = validate_pcap(pcap_path, backend=backend, tshark_path=tshark_path)
        results.append(meta)

        if output_dir:
            json_path = output_dir / f"{pcap_path.stem}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(meta.to_dict(), f, indent=2, default=str)
            logger.info("Metadata written to %s", json_path)

    return results


def generate_validation_report(
    results: list[PcapMetadata],
    output_path: Path | str,
) -> dict[str, Any]:
    """Generate a summary validation report from a list of PCAP metadata.

    Args:
        results: Metadata from ``validate_pcap_directory``.
        output_path: Where to write the JSON report.

    Returns:
        The report as a dict.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_packets = sum(r.packet_count for r in results)
    total_bytes = sum(r.file_size_bytes for r in results)
    valid_count = sum(1 for r in results if r.valid)
    invalid_count = len(results) - valid_count

    report = {
        "total_files": len(results),
        "valid_files": valid_count,
        "invalid_files": invalid_count,
        "total_size_bytes": total_bytes,
        "total_packets": total_packets,
        "total_tcp_packets": sum(r.tcp_packets for r in results),
        "total_udp_packets": sum(r.udp_packets for r in results),
        "total_icmp_packets": sum(r.icmp_packets for r in results),
        "total_other_packets": sum(r.other_packets for r in results),
        "total_malformed_packets": sum(r.malformed_packets for r in results),
        "files": [r.to_dict() for r in results],
    }

    if invalid_count > 0:
        report["invalid_details"] = [
            {"file": r.filename, "error": r.error_message}
            for r in results if not r.valid
        ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(
        "Validation report: %d/%d valid, %d total packets → %s",
        valid_count, len(results), total_packets, output_path,
    )
    return report
