"""Configuration for live packet capture."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CaptureConfig:
    """Configuration settings for live packet capture."""

    interface: str | None = None
    tshark_path: str = "tshark"
    packet_queue_size: int = 10_000
    bpf_filter: str | None = None
    packet_batch_size: int = 500
    snaplen: int = 65535
