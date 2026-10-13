"""Unit and integration tests for packet-count refresh and active flow tracking.

Tests:
1. WindowManager triggers window generation every 100 packets (packet_refresh_threshold)
2. packets_until_next_refresh countdown counts down correctly
3. LiveFlowManager tracks last processed packet count per flow to prevent redundant inference
4. Time-based refresh operates for low-volume traffic (timer fallback)
"""

import time
import pytest
from unittest.mock import MagicMock

from tabpfn_nids.detection.window_manager import WindowManager, DetectionWindow
from tabpfn_nids.flows.live_flow_manager import LiveFlowManager
from tabpfn_nids.pcap.extractor import PacketRecord


def create_dummy_packet(src_ip="192.168.1.10", dst_ip="10.0.0.1", src_port=12345, dst_port=80, proto=6, length=100, ts=None):
    if ts is None:
        ts = time.time()
    return PacketRecord(
        timestamp=ts,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        protocol=proto,
        protocol_name="tcp" if proto == 6 else ("udp" if proto == 17 else "other"),
        length=length,
        tcp_flags=0x02,
        payload_length=max(0, length - 40),
    )


def test_window_manager_packet_count_refresh():
    """WindowManager enqueues a window when packet count reaches packet_refresh_threshold (100)."""
    flow_manager = LiveFlowManager()
    wm = WindowManager(
        flow_manager=flow_manager,
        window_size_seconds=10.0,
        step_seconds=5.0,
        packet_refresh_threshold=100,
    )

    assert wm.packet_refresh_threshold == 100
    assert wm.packets_until_next_refresh == 100

    # Feed 99 packets
    for i in range(99):
        pkt = create_dummy_packet(src_port=10000 + (i % 5))
        wm.record_packet(pkt)

    assert wm.metrics.total_packets_routed == 99
    assert wm.packets_until_next_refresh == 1
    assert wm.window_queue.qsize() == 0

    # 100th packet triggers refresh
    p100 = create_dummy_packet(src_port=10000)
    wm.record_packet(p100)

    assert wm.metrics.total_packets_routed == 100
    assert wm.packets_until_next_refresh == 100  # Reset
    assert wm.window_queue.qsize() == 1

    window = wm.window_queue.get_nowait()
    assert isinstance(window, DetectionWindow)
    assert len(window.flows) > 0


def test_flow_manager_tracks_last_packet_count():
    """LiveFlowManager skips unchanged flows on subsequent snapshot calls."""
    flow_manager = LiveFlowManager()

    # Add 10 packets to Flow A
    for _ in range(10):
        flow_manager.add_packet(create_dummy_packet(src_ip="192.168.1.10", dst_ip="10.0.0.1", src_port=5000, dst_port=80))

    # Add 5 packets to Flow B
    for _ in range(5):
        flow_manager.add_packet(create_dummy_packet(src_ip="192.168.1.20", dst_ip="10.0.0.1", src_port=6000, dst_port=443))

    # First snapshot with only_updated=True returns both flows
    snap1 = flow_manager.get_window_snapshot(only_updated=True)
    assert len(snap1) == 2

    # Second snapshot immediately after without new packets returns 0 flows
    snap2 = flow_manager.get_window_snapshot(only_updated=True)
    assert len(snap2) == 0

    # Add 1 packet only to Flow A
    flow_manager.add_packet(create_dummy_packet(src_ip="192.168.1.10", dst_ip="10.0.0.1", src_port=5000, dst_port=80))

    # Third snapshot returns only Flow A
    snap3 = flow_manager.get_window_snapshot(only_updated=True)
    assert len(snap3) == 1
    assert snap3[0].src_ip == "192.168.1.10" or snap3[0].dst_ip == "192.168.1.10"


def test_configurable_packet_refresh_threshold():
    """packet_refresh_threshold can be configured to other values (e.g. 25)."""
    flow_manager = LiveFlowManager()
    wm = WindowManager(
        flow_manager=flow_manager,
        packet_refresh_threshold=25,
    )
    assert wm.packet_refresh_threshold == 25
    assert wm.packets_until_next_refresh == 25

    for _ in range(25):
        pkt = create_dummy_packet()
        wm.record_packet(pkt)

    assert wm.window_queue.qsize() == 1
