"""Unit tests for live capture, interface manager, live flow manager, and window manager."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tabpfn_nids.capture.capture_config import CaptureConfig
from tabpfn_nids.capture.interface_manager import (
    InterfaceNotFoundError,
    NetworkInterface,
    NpcapNotFoundError,
    TSharkNotFoundError,
    find_tshark_binary,
    list_interfaces,
    resolve_interface,
)
from tabpfn_nids.capture.tshark_capture import (
    CaptureMetrics,
    TSharkCapture,
    parse_tshark_line,
)
from tabpfn_nids.detection.window_manager import DetectionWindow, WindowManager
from tabpfn_nids.flows.flow_builder import FlowRecord, _canonical_flow_key
from tabpfn_nids.flows.live_flow_manager import LiveFlowManager
from tabpfn_nids.inference.live_inference_engine import LiveDetectionResult
from tabpfn_nids.pcap.extractor import PacketRecord


# ─── 1. TShark Line Parser Tests ───────────────────────────────────────────────

def test_parse_tshark_line_tcp():
    line = "1727764800.123456|192.168.1.10|192.168.1.1|54321|443|||6|120|0x0018|66|"
    pkt = parse_tshark_line(line)
    assert pkt is not None
    assert pkt.timestamp == 1727764800.123456
    assert pkt.src_ip == "192.168.1.10"
    assert pkt.dst_ip == "192.168.1.1"
    assert pkt.src_port == 54321
    assert pkt.dst_port == 443
    assert pkt.protocol == 6
    assert pkt.protocol_name == "tcp"
    assert pkt.length == 120
    assert pkt.tcp_flags == 0x18
    assert pkt.payload_length == 66


def test_parse_tshark_line_udp():
    line = "1727764801.000000|10.0.0.5|8.8.8.8|||5353|53|17|80|||40"
    pkt = parse_tshark_line(line)
    assert pkt is not None
    assert pkt.protocol == 17
    assert pkt.protocol_name == "udp"
    assert pkt.src_port == 5353
    assert pkt.dst_port == 53
    assert pkt.length == 80
    assert pkt.payload_length == 40
    assert pkt.tcp_flags == 0


def test_parse_tshark_line_icmp():
    line = "1727764802.000000|10.0.0.1|10.0.0.2|||||1|64|||"
    pkt = parse_tshark_line(line)
    assert pkt is not None
    assert pkt.protocol == 1
    assert pkt.protocol_name == "icmp"
    assert pkt.src_port == 0
    assert pkt.dst_port == 0
    assert pkt.length == 64


def test_parse_tshark_line_malformed():
    assert parse_tshark_line("") is None
    assert parse_tshark_line("random invalid data") is None
    assert parse_tshark_line("1727764800|||||||||||") is None


# ─── 2. Interface Manager Tests ───────────────────────────────────────────────

def test_resolve_interface_by_index():
    ifaces = [
        NetworkInterface(index=1, device=r"\Device\NPF_1", name="Ethernet", description="Ethernet"),
        NetworkInterface(index=4, device=r"\Device\NPF_4", name="Wi-Fi", description="Wi-Fi Adapter"),
    ]
    resolved = resolve_interface(4, ifaces)
    assert resolved.name == "Wi-Fi"
    assert resolved.index == 4

    resolved_str = resolve_interface("1", ifaces)
    assert resolved_str.name == "Ethernet"


def test_resolve_interface_by_name():
    ifaces = [
        NetworkInterface(index=1, device=r"\Device\NPF_1", name="Ethernet", description="Intel Ethernet"),
        NetworkInterface(index=2, device=r"\Device\NPF_2", name="Wi-Fi", description="Intel Wi-Fi 6"),
    ]
    resolved = resolve_interface("wi-fi", ifaces)
    assert resolved.index == 2
    assert resolved.name == "Wi-Fi"


def test_resolve_interface_not_found():
    ifaces = [
        NetworkInterface(index=1, device=r"\Device\NPF_1", name="Ethernet", description="Ethernet"),
    ]
    with pytest.raises(InterfaceNotFoundError):
        resolve_interface("NonExistentAdapter", ifaces)


def test_find_tshark_binary_not_found():
    with patch("shutil.which", return_value=None), \
         patch("pathlib.Path.is_file", return_value=False):
        with pytest.raises(TSharkNotFoundError) as exc_info:
            find_tshark_binary(configured_path="non_existent_tshark_path_xyz")
        assert "TShark executable was not found" in str(exc_info.value)


# ─── 3. Canonical Flow Key & Bidirectional Matching ────────────────────────────

def test_canonical_flow_key_bidirectional():
    # Forward packet
    key_fwd = _canonical_flow_key("192.168.1.10", 12345, "10.0.0.1", 80, 6)
    # Reverse packet
    key_rev = _canonical_flow_key("10.0.0.1", 80, "192.168.1.10", 12345, 6)
    assert key_fwd == key_rev


# ─── 4. Live Flow Manager Tests ────────────────────────────────────────────────

def test_live_flow_manager_aggregation():
    mgr = LiveFlowManager(flow_timeout_seconds=10.0, idle_timeout_seconds=5.0)

    p1 = PacketRecord(
        timestamp=100.0,
        src_ip="192.168.1.5",
        dst_ip="1.1.1.1",
        src_port=50000,
        dst_port=443,
        protocol=6,
        protocol_name="tcp",
        length=60,
        tcp_flags=0x02,  # SYN
        payload_length=0,
    )
    p2 = PacketRecord(
        timestamp=100.1,
        src_ip="1.1.1.1",
        dst_ip="192.168.1.5",
        src_port=443,
        dst_port=50000,
        protocol=6,
        protocol_name="tcp",
        length=60,
        tcp_flags=0x12,  # SYN-ACK
        payload_length=0,
    )

    mgr.add_packet(p1)
    mgr.add_packet(p2)

    assert mgr.active_flow_count == 1
    assert mgr.total_packets_ingested == 2

    # Get snapshot
    flows = mgr.get_window_snapshot(only_updated=True)
    assert len(flows) == 1
    flow = flows[0]
    assert flow.total_packets == 2
    assert flow.fwd_packets == 1
    assert flow.bwd_packets == 1
    assert flow.fwd_bytes == 60
    assert flow.bwd_bytes == 60
    assert flow.fwd_syn == 1
    assert flow.bwd_syn == 1
    assert flow.bwd_ack == 1


def test_live_flow_manager_timeout():
    mgr = LiveFlowManager(flow_timeout_seconds=5.0, idle_timeout_seconds=2.0)

    p1 = PacketRecord(
        timestamp=100.0,
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        src_port=1111,
        dst_port=2222,
        protocol=6,
        protocol_name="tcp",
        length=100,
        tcp_flags=0x18,
        payload_length=50,
    )
    mgr.add_packet(p1)
    assert mgr.active_flow_count == 1

    # At t=101.0, idle timeout (2.0s) has not expired
    expired = mgr.check_timeouts(current_time=101.0)
    assert len(expired) == 0
    assert mgr.active_flow_count == 1

    # At t=103.5, idle timeout (2.0s) has expired
    expired = mgr.check_timeouts(current_time=103.5)
    assert len(expired) == 1
    assert mgr.active_flow_count == 0
    assert expired[0].src_ip == "10.0.0.1"


# ─── 5. Window Manager & Backpressure Tests ────────────────────────────────────

@pytest.mark.asyncio
async def test_window_manager_enqueue_and_backpressure():
    flow_mgr = LiveFlowManager()
    win_mgr = WindowManager(
        flow_manager=flow_mgr,
        window_size_seconds=2.0,
        step_seconds=0.2,
        queue_size=2,  # Intentionally small queue to test backpressure
    )

    p = PacketRecord(
        timestamp=100.0,
        src_ip="192.168.1.1",
        dst_ip="8.8.8.8",
        src_port=1000,
        dst_port=53,
        protocol=17,
        protocol_name="udp",
        length=50,
        tcp_flags=0,
        payload_length=20,
    )

    win_mgr.start()
    win_mgr.record_packet(p)

    # Allow window timer loop to run for a couple of cycles
    await asyncio.sleep(0.6)

    # Queue should contain windows, and if full, dropped count tracks backpressure
    assert win_mgr.window_queue.qsize() > 0 or win_mgr.metrics.windows_created > 0

    await win_mgr.stop()
    assert not win_mgr.is_running


# ─── 6. Live Detection Result Formatting ──────────────────────────────────────

def test_live_detection_result_serialization():
    res = LiveDetectionResult(
        window_id="win-00042",
        timestamp="2026-10-01T15:00:00Z",
        prediction="BENIGN",
        confidence=0.975432,
        flows_analyzed=15,
        packets_analyzed=120,
        attack_count=0,
        normal_count=15,
        latencies={"inference_latency": 0.01234, "total_latency": 0.02345},
    )

    data = res.to_dict()
    assert data["window_id"] == "win-00042"
    assert data["prediction"] == "BENIGN"
    assert data["confidence"] == 0.9754
    assert data["flows_analyzed"] == 15
    assert data["packets_analyzed"] == 120
    assert data["attack_count"] == 0
    assert data["latencies"]["inference_latency"] == 0.0123
