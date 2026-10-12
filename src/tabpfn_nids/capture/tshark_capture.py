"""Asynchronous live packet capture engine using TShark on Windows."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator

from tabpfn_nids.capture.capture_config import CaptureConfig
from tabpfn_nids.capture.interface_manager import (
    NetworkInterface,
    find_tshark_binary,
    list_interfaces,
    resolve_interface,
)
from tabpfn_nids.pcap.extractor import PacketRecord

logger = logging.getLogger(__name__)

# Structured TShark extraction fields
TSHARK_FIELDS = [
    "frame.time_epoch",
    "ip.src",
    "ip.dst",
    "tcp.srcport",
    "tcp.dstport",
    "udp.srcport",
    "udp.dstport",
    "ip.proto",
    "frame.len",
    "tcp.flags",
    "tcp.len",
    "udp.length",
]


@dataclass
class CaptureMetrics:
    """Live capture telemetry metrics."""

    packets_captured: int = 0
    packets_processed: int = 0
    packets_dropped: int = 0
    malformed_packets: int = 0
    queue_size: int = 0
    is_running: bool = False
    interface: str = ""
    tshark_pid: int | None = None
    start_time: float | None = None
    uptime_seconds: float = 0.0
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        uptime = 0.0
        if self.start_time and self.is_running:
            uptime = round(time.time() - self.start_time, 1)
        elif self.uptime_seconds > 0:
            uptime = self.uptime_seconds

        return {
            "running": self.is_running,
            "interface": self.interface,
            "packets_captured": self.packets_captured,
            "packets_processed": self.packets_processed,
            "packets_dropped": self.packets_dropped,
            "malformed_packets": self.malformed_packets,
            "queue_size": self.queue_size,
            "tshark_pid": self.tshark_pid,
            "uptime_seconds": uptime,
            "last_error": self.last_error,
        }


def parse_tshark_line(line: str) -> PacketRecord | None:
    """Safely parse a single pipe-delimited TShark line into a PacketRecord.

    Expected fields (12 elements):
    0: frame.time_epoch
    1: ip.src
    2: ip.dst
    3: tcp.srcport
    4: tcp.dstport
    5: udp.srcport
    6: udp.dstport
    7: ip.proto
    8: frame.len
    9: tcp.flags
    10: tcp.len
    11: udp.length
    """
    if not line:
        return None

    parts = line.strip().split("|")
    if len(parts) < len(TSHARK_FIELDS):
        return None

    try:
        ts_str = parts[0]
        ts = float(ts_str) if ts_str else time.time()
        src_ip = parts[1]
        dst_ip = parts[2]

        if not src_ip or not dst_ip:
            return None

        # TCP
        if parts[3]:
            src_port = int(parts[3])
            dst_port = int(parts[4]) if parts[4] else 0
            proto_num = 6
            proto_name = "tcp"
            # tcp.flags in TShark can be hex like 0x0002 or 0x018
            flags_str = parts[9]
            if flags_str:
                tcp_flags = int(flags_str, 16) if flags_str.startswith(("0x", "0X")) else int(flags_str)
            else:
                tcp_flags = 0
            payload_len = int(parts[10]) if parts[10] else 0

        # UDP
        elif parts[5]:
            src_port = int(parts[5])
            dst_port = int(parts[6]) if parts[6] else 0
            proto_num = 17
            proto_name = "udp"
            tcp_flags = 0
            payload_len = int(parts[11]) if parts[11] else 0

        # Other IP (e.g. ICMP = 1)
        else:
            proto_num = int(parts[7]) if parts[7] else 0
            proto_name = "icmp" if proto_num == 1 else "other"
            src_port = 0
            dst_port = 0
            tcp_flags = 0
            payload_len = 0

        length_str = parts[8]
        length = int(length_str) if length_str else 0

        return PacketRecord(
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

    except Exception:
        return None


class TSharkCapture:
    """Asynchronous TShark capture manager for live packet streams."""

    def __init__(self, config: CaptureConfig | None = None) -> None:
        self.config = config or CaptureConfig()
        self.metrics = CaptureMetrics()
        self.packet_queue: asyncio.Queue[PacketRecord] = asyncio.Queue(
            maxsize=self.config.packet_queue_size
        )
        self._process: asyncio.subprocess.Process | None = None
        self._read_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._resolved_interface: NetworkInterface | None = None
        self._tshark_bin: Path | None = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    def get_metrics(self) -> dict[str, Any]:
        """Return a snapshot of capture metrics."""
        self.metrics.is_running = self.is_running
        self.metrics.queue_size = self.packet_queue.qsize()
        return self.metrics.to_dict()

    async def start(self) -> None:
        """Initialize and start the live capture process."""
        if self.is_running:
            logger.warning("Capture already running (PID=%s)", self.metrics.tshark_pid)
            return

        self._tshark_bin = find_tshark_binary(self.config.tshark_path)

        # Discover & resolve interface
        available_ifaces = list_interfaces(self._tshark_bin)
        if not available_ifaces:
            raise RuntimeError("No network interfaces detected by TShark/Npcap.")

        if self.config.interface is None:
            # Default to first interface
            self._resolved_interface = available_ifaces[0]
        else:
            self._resolved_interface = resolve_interface(
                self.config.interface, available_ifaces
            )

        self._stop_event.clear()

        # Build tshark command
        # On Windows, we can use the device string or interface index
        iface_arg = str(self._resolved_interface.index)
        if sys.platform == "win32" and self._resolved_interface.device:
            iface_arg = self._resolved_interface.device

        cmd = [
            str(self._tshark_bin),
            "-i", iface_arg,
            "-l",               # Line-buffered output
            "-n",               # Disable network object name resolution
            "-T", "fields",
            *[arg for f in TSHARK_FIELDS for arg in ("-e", f)],
            "-E", "separator=|",
            "-E", "occurrence=f",
        ]

        if self.config.bpf_filter:
            cmd.extend(["-f", self.config.bpf_filter])

        logger.info(
            "Starting TShark live capture on '%s' (Device: %s)",
            self._resolved_interface.name,
            iface_arg,
        )

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:
            self.metrics.last_error = str(exc)
            logger.error("Failed to spawn TShark process: %s", exc)
            raise

        self.metrics.is_running = True
        self.metrics.interface = self._resolved_interface.name
        self.metrics.tshark_pid = self._process.pid
        self.metrics.start_time = time.time()
        self.metrics.last_error = None

        logger.info("TShark capture process started (PID=%d)", self._process.pid)

        # Start asynchronous readers
        self._read_task = asyncio.create_task(
            self._read_stdout_stream(), name="tshark-stdout-reader"
        )
        self._stderr_task = asyncio.create_task(
            self._read_stderr_stream(), name="tshark-stderr-reader"
        )

    async def _read_stdout_stream(self) -> None:
        """Asynchronously stream and parse lines from TShark stdout."""
        assert self._process is not None
        assert self._process.stdout is not None

        stdout = self._process.stdout
        last_drop_warning = 0.0

        try:
            while not self._stop_event.is_set():
                line_bytes = await stdout.readline()
                if not line_bytes:
                    break

                try:
                    line_str = line_bytes.decode("utf-8", errors="replace").strip()
                except Exception:
                    self.metrics.malformed_packets += 1
                    continue

                if not line_str:
                    continue

                pkt = parse_tshark_line(line_str)
                if pkt is None:
                    self.metrics.malformed_packets += 1
                    continue

                self.metrics.packets_captured += 1

                # Enqueue with bounded backpressure handling
                try:
                    self.packet_queue.put_nowait(pkt)
                except asyncio.QueueFull:
                    self.metrics.packets_dropped += 1
                    now = time.time()
                    if now - last_drop_warning > 5.0:
                        logger.warning(
                            "Packet queue is full (size=%d)! Dropping packets "
                            "(total dropped: %d). Increase packet_queue_size or consumer throughput.",
                            self.config.packet_queue_size,
                            self.metrics.packets_dropped,
                        )
                        last_drop_warning = now

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.metrics.last_error = f"Error reading TShark stdout: {exc}"
            logger.error("Error reading TShark stdout: %s", exc)
        finally:
            logger.debug("TShark stdout reader finished.")

    async def _read_stderr_stream(self) -> None:
        """Asynchronously consume TShark stderr messages for diagnostics."""
        assert self._process is not None
        assert self._process.stderr is not None

        stderr = self._process.stderr
        try:
            while not self._stop_event.is_set():
                line_bytes = await stderr.readline()
                if not line_bytes:
                    break
                msg = line_bytes.decode("utf-8", errors="replace").strip()
                if msg:
                    # TShark outputs informational messages like "Capturing on 'Wi-Fi'"
                    if "Capturing on" in msg:
                        logger.info("TShark: %s", msg)
                    elif "error" in msg.lower() or "failed" in msg.lower():
                        logger.warning("TShark warning/error: %s", msg)
                        self.metrics.last_error = msg
                    else:
                        logger.debug("TShark stderr: %s", msg)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def stop(self) -> CaptureMetrics:
        """Gracefully terminate TShark capture process and release resources."""
        self._stop_event.set()

        if self._process is not None:
            pid = self._process.pid
            logger.info("Stopping TShark capture process (PID=%s)...", pid)

            if self._process.returncode is None:
                try:
                    self._process.terminate()
                except ProcessLookupError:
                    pass
                except Exception as exc:
                    logger.debug("Terminate call error: %s", exc)

                try:
                    await asyncio.wait_for(self._process.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    logger.warning("TShark did not exit gracefully within 3s. Force killing (PID=%s)...", pid)
                    try:
                        self._process.kill()
                        await self._process.wait()
                    except Exception as exc:
                        logger.error("Failed to kill TShark process: %s", exc)

        # Cancel readers
        if self._read_task and not self._read_task.done():
            self._read_task.cancel()
        if self._stderr_task and not self._stderr_task.done():
            self._stderr_task.cancel()

        if self.metrics.start_time:
            self.metrics.uptime_seconds = round(time.time() - self.metrics.start_time, 1)

        self.metrics.is_running = False
        self.metrics.tshark_pid = None
        self._process = None

        logger.info(
            "TShark capture stopped. Captured: %d, Processed: %d, Dropped: %d",
            self.metrics.packets_captured,
            self.metrics.packets_processed,
            self.metrics.packets_dropped,
        )
        return self.metrics
