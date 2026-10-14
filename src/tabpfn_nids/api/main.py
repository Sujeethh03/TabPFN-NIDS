"""FastAPI REST and WebSocket application for TabPFN-NIDS.

Provides two operational modes:
  1. PCAP File Analysis  — upload a .pcap file and receive an intrusion-detection report.
  2. Live Network Monitoring — real-time packet capture and detection (existing behaviour).
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from tabpfn_nids.api.event_bus import EventBus
from tabpfn_nids.api.database_client import (
    get_analysis_by_id,
    get_analysis_history,
    get_analysis_predictions,
)
from tabpfn_nids.api.pcap_router import router as pcap_router
from tabpfn_nids.capture.interface_manager import (
    InterfaceNotFoundError,
    NpcapNotFoundError,
    TSharkNotFoundError,
    list_interfaces,
)
from tabpfn_nids.capture.live_pipeline import (
    LivePipelineConfig,
    LivePipelineCoordinator,
)
from tabpfn_nids.config import PROJECT_ROOT

from tabpfn_nids.firewall.firewall_manager import (
    FirewallConfig,
    FirewallManager,
    FirewallMode,
)

logger = logging.getLogger(__name__)

DASHBOARD_FILE = PROJECT_ROOT / "dashboard" / "index.html"

# Global state
event_bus = EventBus(history_limit=50)
global_firewall_manager = FirewallManager(FirewallConfig())
coordinator: LivePipelineCoordinator | None = None
coordinator_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global coordinator
    logger.info("TabPFN-NIDS API service starting up.")
    yield
    # Graceful shutdown
    logger.info("TabPFN-NIDS API service shutting down.")
    async with coordinator_lock:
        if coordinator and coordinator.is_running:
            try:
                await coordinator.stop()
            except Exception as exc:
                logger.error("Error during shutdown cleanup: %s", exc)


app = FastAPI(
    title="TabPFN-NIDS API",
    description=(
        "Network Intrusion Detection System powered by TabPFN.\n\n"
        "**Modes**: PCAP file analysis (offline) and live network monitoring (real-time)."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# Register PCAP analysis router (prefix: /api)
app.include_router(pcap_router, prefix="/api")


@app.get("/api/analyses", summary="List saved PCAP analyses")
async def list_saved_analyses(limit: int = 20):
    """Return recent analysis records stored by the Prisma database service."""
    if not 1 <= limit <= 100:
        raise HTTPException(
            status_code=422,
            detail="limit must be between 1 and 100",
        )

    try:
        return await asyncio.to_thread(get_analysis_history, limit)
    except RuntimeError as exc:
        logger.exception("Could not load saved analyses from the database")
        raise HTTPException(
            status_code=503,
            detail="Saved analysis history is temporarily unavailable.",
        ) from exc


@app.get(
    "/api/analyses/{analysis_id}",
    summary="Get a saved PCAP analysis by ID",
)
async def get_saved_analysis(analysis_id: str):
    """Return a single saved analysis record from the Prisma database service."""
    try:
        return await asyncio.to_thread(get_analysis_by_id, analysis_id)
    except RuntimeError as exc:
        if "HTTP 404" in str(exc):
            raise HTTPException(
                status_code=404,
                detail=f"Analysis '{analysis_id}' not found.",
            ) from exc
        logger.exception(
            "Could not load saved analysis %s from the database",
            analysis_id,
        )
        raise HTTPException(
            status_code=503,
            detail="Saved analysis is temporarily unavailable.",
        ) from exc


@app.get(
    "/api/analyses/{analysis_id}/predictions",
    summary="List saved predictions for an analysis",
)
async def list_saved_predictions(
    analysis_id: str,
    skip: int = 0,
    limit: int = 100,
):
    """Return a page of flow predictions from the Prisma database service."""
    if skip < 0 or not 1 <= limit <= 500:
        raise HTTPException(
            status_code=422,
            detail="skip must be non-negative and limit must be between 1 and 500",
        )

    try:
        return await asyncio.to_thread(
            get_analysis_predictions,
            analysis_id,
            skip,
            limit,
        )
    except RuntimeError as exc:
        logger.exception(
            "Could not load saved predictions for analysis %s",
            analysis_id,
        )
        raise HTTPException(
            status_code=503,
            detail="Saved predictions are temporarily unavailable.",
        ) from exc


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request Models
class StartCaptureRequest(BaseModel):
    interface: str = Field(
        ...,
        description="Network interface name, index, or device string (e.g. 'Wi-Fi' or '4')",
    )
    packet_refresh_threshold: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Packet count counter threshold to trigger flow refresh and inference",
    )
    window_size_seconds: float = Field(
        default=10.0,
        gt=0.0,
        le=300.0,
        description="Sliding detection window size in seconds",
    )
    step_seconds: float = Field(
        default=5.0,
        gt=0.0,
        le=300.0,
        description="Window evaluation step interval in seconds",
    )
    flow_timeout_seconds: float = Field(
        default=10.0,
        gt=0.0,
        le=600.0,
        description="Maximum flow timeout in seconds",
    )
    max_workers: int = Field(
        default=2,
        ge=1,
        le=16,
        description="Maximum parallel inference workers",
    )
    tshark_path: str = Field(
        default="tshark",
        description="Path to tshark executable or 'tshark' if on PATH",
    )
    bpf_filter: str | None = Field(
        default=None,
        description="Optional BPF capture filter (e.g. 'ip and not port 22')",
    )
    firewall_mode: str = Field(
        default="monitor_only",
        description="Firewall response mode: 'monitor_only' or 'block_attacker'",
    )
    admin_authorized: bool = Field(
        default=False,
        description="Explicit administrator authorization to provision firewall rules",
    )
    min_confidence_to_block: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum detection confidence required before blocking",
    )
    rule_expiration_seconds: float = Field(
        default=300.0,
        ge=0.0,
        description="Firewall rule expiration in seconds (0 = permanent)",
    )
    trusted_allowlist: list[str] = Field(
        default_factory=list,
        description="List of IP addresses or CIDRs protected from firewall blocks",
    )


class UpdateFirewallConfigRequest(BaseModel):
    mode: str | None = Field(default=None, description="'monitor_only' or 'block_attacker'")
    admin_authorized: bool | None = Field(default=None, description="Explicit administrator authorization")
    min_confidence_to_block: float | None = Field(default=None, ge=0.0, le=1.0)
    rule_expiration_seconds: float | None = Field(default=None, ge=0.0)
    trusted_allowlist: list[str] | None = None


class UnblockIpRequest(BaseModel):
    source_ip: str = Field(..., description="Source IP address to unblock")


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", summary="Dashboard UI")
async def get_dashboard():
    """Serve the single-page live monitoring dashboard."""
    if not DASHBOARD_FILE.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dashboard file not found at {DASHBOARD_FILE}",
        )
    return FileResponse(DASHBOARD_FILE, media_type="text/html")


@app.get("/capture/interfaces", summary="List available network interfaces")
async def get_interfaces():
    """Discover and list all network interfaces detected by TShark and Npcap."""
    try:
        ifaces = await asyncio.to_thread(list_interfaces)
        return {"interfaces": [i.to_dict() for i in ifaces]}
    except TSharkNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "TShark executable was not found",
                "suggestion": "Install Wireshark with TShark and Npcap, or configure tshark_path in settings.",
                "details": str(exc),
            },
        )
    except NpcapNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Npcap packet capture driver is not available",
                "suggestion": "Install Npcap from https://npcap.com/#download with WinPcap compatibility.",
                "details": str(exc),
            },
        )
    except Exception as exc:
        logger.error("Failed to list interfaces: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to discover interfaces", "details": str(exc)},
        )


@app.post("/capture/start", summary="Start live capture session")
async def start_capture(req: StartCaptureRequest):
    """Start continuous live packet capture and intrusion detection."""
    global coordinator

    async with coordinator_lock:
        if coordinator and coordinator.is_running:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Capture session is already running",
                    "suggestion": "Stop the current capture session before starting a new one.",
                },
            )

        pipeline_config = LivePipelineConfig(
            interface=req.interface,
            tshark_path=req.tshark_path,
            packet_refresh_threshold=req.packet_refresh_threshold,
            window_size_seconds=req.window_size_seconds,
            step_seconds=req.step_seconds,
            flow_timeout_seconds=req.flow_timeout_seconds,
            idle_timeout_seconds=req.flow_timeout_seconds,
            max_workers=req.max_workers,
            bpf_filter=req.bpf_filter,
            firewall_mode=req.firewall_mode,
            firewall_admin_authorized=req.admin_authorized,
            firewall_min_confidence=req.min_confidence_to_block,
            firewall_rule_expiration=req.rule_expiration_seconds,
            firewall_trusted_allowlist=req.trusted_allowlist,
        )

        coordinator = LivePipelineCoordinator(
            config=pipeline_config, event_bus=event_bus
        )

        try:
            await coordinator.start()
        except TSharkNotFoundError as exc:
            coordinator = None
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "TShark executable was not found",
                    "suggestion": "Install Wireshark with TShark and Npcap or specify tshark_path.",
                    "details": str(exc),
                },
            )
        except InterfaceNotFoundError as exc:
            coordinator = None
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Selected interface was not found",
                    "suggestion": "Query GET /capture/interfaces to select a valid interface name or index.",
                    "details": str(exc),
                },
            )
        except Exception as exc:
            coordinator = None
            logger.error("Failed to start live capture: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Failed to start capture pipeline",
                    "details": str(exc),
                },
            )

        return {
            "status": "started",
            "interface": req.interface,
            "packet_refresh_threshold": req.packet_refresh_threshold,
            "window_size_seconds": req.window_size_seconds,
            "step_seconds": req.step_seconds,
            "firewall_mode": req.firewall_mode,
            "admin_authorized": req.admin_authorized,
        }


@app.post("/capture/stop", summary="Stop live capture session")
async def stop_capture():
    """Gracefully stop live packet capture, drain queues, and finalize results."""
    global coordinator

    async with coordinator_lock:
        if not coordinator or not coordinator.is_running:
            return {"status": "stopped", "message": "No capture session was running"}

        try:
            final_stats = await coordinator.stop()
            return {"status": "stopped", "statistics": final_stats}
        except Exception as exc:
            logger.error("Error stopping live capture: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Failed to stop capture session cleanly", "details": str(exc)},
            )


@app.get("/capture/status", summary="Get current capture status")
async def get_status():
    """Retrieve real-time metrics, queue statistics, and detection counts."""
    if coordinator:
        return coordinator.get_status()

    return {
        "running": False,
        "interface": None,
        "packets_captured": 0,
        "packets_processed": 0,
        "packets_dropped": 0,
        "packet_refresh_threshold": 100,
        "packets_until_next_refresh": 100,
        "active_flows": 0,
        "windows_processed": 0,
        "attack_count": 0,
        "job_queue_size": 0,
        "windows_dropped": 0,
        "windows_triggered_by_packet_count": 0,
        "windows_triggered_by_timer": 0,
        "inference_jobs_failed": 0,
        "uptime_seconds": 0.0,
        "tshark_pid": None,
        "websocket_subscribers": event_bus.subscriber_count,
        "firewall": global_firewall_manager.get_status(),
    }


# ─── Firewall Endpoints ────────────────────────────────────────────────────────

@app.get("/firewall/status", summary="Get host firewall status and rules")
async def get_firewall_status():
    """Retrieve current firewall response mode, active block rules, and capabilities."""
    active_mgr = coordinator.firewall_manager if (coordinator and coordinator.is_running) else global_firewall_manager
    status_data = active_mgr.get_status()
    status_data["active_blocks"] = active_mgr.get_active_blocks()
    return status_data


@app.post("/firewall/config", summary="Configure firewall response settings")
async def update_firewall_config(req: UpdateFirewallConfigRequest):
    """Update firewall response mode, administrator authorization, or allowlist."""
    active_mgr = coordinator.firewall_manager if (coordinator and coordinator.is_running) else global_firewall_manager
    try:
        active_mgr.update_config(
            mode=req.mode,
            admin_authorized=req.admin_authorized,
            min_confidence_to_block=req.min_confidence_to_block,
            rule_expiration_seconds=req.rule_expiration_seconds,
            trusted_allowlist=req.trusted_allowlist,
        )
        # Also sync global manager if running coordinator
        if coordinator and coordinator.firewall_manager != global_firewall_manager:
            global_firewall_manager.update_config(
                mode=req.mode,
                admin_authorized=req.admin_authorized,
                min_confidence_to_block=req.min_confidence_to_block,
                rule_expiration_seconds=req.rule_expiration_seconds,
                trusted_allowlist=req.trusted_allowlist,
            )
        return {"status": "updated", "config": active_mgr.get_status()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/firewall/unblock", summary="Manually unblock an IP address")
async def unblock_firewall_ip(req: UnblockIpRequest):
    """Manually revoke a firewall block rule for a source IP."""
    active_mgr = coordinator.firewall_manager if (coordinator and coordinator.is_running) else global_firewall_manager
    event = active_mgr.unblock_ip(req.source_ip)
    return event.to_dict()


@app.get("/firewall/logs", summary="List firewall audit logs")
async def get_firewall_logs(limit: int = 50):
    """Retrieve recent firewall defense actions and evaluation results."""
    active_mgr = coordinator.firewall_manager if (coordinator and coordinator.is_running) else global_firewall_manager
    return {"logs": active_mgr.get_audit_logs(limit=limit)}


@app.get("/capture/statistics", summary="Get detailed performance statistics")
async def get_statistics():
    """Return detailed latency breakdown and pipeline throughput."""
    if not coordinator:
        return {"running": False, "statistics": None}

    status_data = coordinator.get_status()
    return {
        "running": coordinator.is_running,
        "interface": status_data["interface"],
        "uptime_seconds": status_data["uptime_seconds"],
        "packets": {
            "captured": status_data["packets_captured"],
            "processed": status_data["packets_processed"],
            "dropped": status_data["packets_dropped"],
        },
        "flows": {
            "active": status_data["active_flows"],
        },
        "windows": {
            "processed": status_data["windows_processed"],
            "dropped": status_data["windows_dropped"],
            "queue_size": status_data["job_queue_size"],
        },
        "detections": {
            "attacks": status_data["attack_count"],
            "failed_jobs": status_data["inference_jobs_failed"],
        },
        "latencies": status_data.get("metrics", {}),
    }


@app.websocket("/ws/detections")
async def websocket_detections(websocket: WebSocket):
    """WebSocket endpoint streaming live detection events and capture status updates."""
    await websocket.accept()

    # Register subscriber queue
    subscriber_queue = event_bus.subscribe()

    # Send initial connection ack with recent detection history
    ack_payload = {
        "type": "connection_ack",
        "data": {
            "status": "connected",
            "recent_history": event_bus.get_recent_history(),
        },
    }
    await websocket.send_json(ack_payload)

    # If pipeline is already running, send immediate status update
    if coordinator:
        await websocket.send_json({
            "type": "capture_status",
            "data": coordinator.get_status(),
        })

    try:
        while True:
            # We listen for client messages (or ping/keepalive) and stream events
            event = await subscriber_queue.get()
            await websocket.send_json(event)
            subscriber_queue.task_done()
    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected.")
    except Exception as exc:
        logger.debug("WebSocket connection terminated: %s", exc)
    finally:
        event_bus.unsubscribe(subscriber_queue)
