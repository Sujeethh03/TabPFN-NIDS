"""Integration tests for FastAPI endpoints and WebSocket interface."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tabpfn_nids.api.event_bus import EventBus
from tabpfn_nids.api.main import app, coordinator_lock
from tabpfn_nids.capture.interface_manager import NetworkInterface


@pytest.fixture
def client():
    return TestClient(app)


def test_get_dashboard(client: TestClient):
    resp = client.get("/")
    assert resp.status_code == 200
    # Dashboard now serves the PCAP analysis interface
    assert "TabPFN" in resp.text
    assert "NIDS" in resp.text


def test_get_capture_status_idle(client: TestClient):
    resp = client.get("/capture/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["running"] is False
    assert data["packets_captured"] == 0
    assert "uptime_seconds" in data


def test_get_capture_statistics_idle(client: TestClient):
    resp = client.get("/capture/statistics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["running"] is False


def test_get_capture_interfaces(client: TestClient):
    mock_ifaces = [
        NetworkInterface(index=1, device=r"\Device\NPF_1", name="Wi-Fi", description="Intel Wi-Fi"),
        NetworkInterface(index=2, device=r"\Device\NPF_2", name="Ethernet", description="Realtek Ethernet"),
    ]
    with patch("tabpfn_nids.api.main.list_interfaces", return_value=mock_ifaces):
        resp = client.get("/capture/interfaces")
        assert resp.status_code == 200
        data = resp.json()
        assert "interfaces" in data
        assert len(data["interfaces"]) == 2
        assert data["interfaces"][0]["name"] == "Wi-Fi"


def test_start_capture_validation(client: TestClient):
    # Invalid window_size (0)
    resp = client.post("/capture/start", json={
        "interface": "Wi-Fi",
        "window_size_seconds": 0,
        "step_seconds": 5,
        "flow_timeout_seconds": 10,
    })
    assert resp.status_code == 422  # Pydantic validation error

    # Missing interface
    resp = client.post("/capture/start", json={
        "window_size_seconds": 10,
        "step_seconds": 5,
    })
    assert resp.status_code == 422


def test_stop_capture_when_not_running(client: TestClient):
    resp = client.post("/capture/stop")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "stopped"


def test_websocket_connection_ack(client: TestClient):
    with client.websocket_connect("/ws/detections") as websocket:
        data = websocket.receive_json()
        assert data["type"] == "connection_ack"
        assert data["data"]["status"] == "connected"
        assert "recent_history" in data["data"]
