"""
PCAP Upload and Analysis API Router

Provides REST endpoints for:
  POST /api/pcap/upload              — upload a .pcap file
  POST /api/pcap/analyze/{job_id}    — start analysis job
  GET  /api/pcap/status/{job_id}     — poll job progress
  GET  /api/pcap/report/{job_id}     — retrieve full analysis report
  GET  /api/pcap/download/{job_id}   — download prediction CSV
  GET  /api/pcap/flows/{job_id}      — paginated flow-level predictions
  GET  /api/pcap/info                — backend/model information

The pipeline reuses existing logic from scripts/analyze_pcap.py without
duplicating model loading, preprocessing, or inference code.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import pickle
import tempfile
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pcap", tags=["PCAP Analysis"])

# ---------------------------------------------------------------------------
# Paths (mirrors scripts/analyze_pcap.py)
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[3]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "models" / "tabpfn_binary_model.pkl"
FEATURE_SCHEMA_PATH = ARTIFACTS_DIR / "models" / "model_feature_schema.json"
SCALER_PATH = ARTIFACTS_DIR / "preprocessing" / "fitted_scaler.pkl"
UPLOAD_DIR = PROJECT_ROOT / "results" / "pcap_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Global in-process artifact cache (load once, reuse)
# ---------------------------------------------------------------------------

_artifact_cache: dict[str, Any] = {}


def _load_artifacts() -> tuple:
    """Load trained model, scaler, and feature schema (cached)."""
    if "model" not in _artifact_cache:
        logger.info("Loading TabPFN model artifacts...")
        with open(MODEL_PATH, "rb") as f:
            _artifact_cache["model"] = pickle.load(f)
        with open(FEATURE_SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)
        _artifact_cache["feature_names"] = schema["feature_names"]
        with open(SCALER_PATH, "rb") as f:
            _artifact_cache["scaler"] = pickle.load(f)
        logger.info(
            "Artifacts loaded. Model expects %d features.",
            len(_artifact_cache["feature_names"]),
        )
    return (
        _artifact_cache["model"],
        _artifact_cache["scaler"],
        _artifact_cache["feature_names"],
    )


def _get_model_info() -> dict[str, Any]:
    """Return static model metadata."""
    feature_names = _artifact_cache.get("feature_names", [])
    return {
        "model_type": "TabPFN",
        "task": "Binary Network Intrusion Detection",
        "feature_count": len(feature_names) if feature_names else 67,
        "artifacts_dir": str(ARTIFACTS_DIR),
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
    }


# ---------------------------------------------------------------------------
# Job state store
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    PENDING = "pending"
    UPLOADING = "uploading"
    EXTRACTING = "extracting"
    BUILDING_FEATURES = "building_features"
    PREPROCESSING = "preprocessing"
    INFERRING = "inferring"
    GENERATING_REPORT = "generating_report"
    COMPLETE = "complete"
    FAILED = "failed"


_STATUS_LABELS: dict[str, str] = {
    JobStatus.PENDING: "Waiting to start",
    JobStatus.UPLOADING: "PCAP uploaded",
    JobStatus.EXTRACTING: "Extracting packets & building flows",
    JobStatus.BUILDING_FEATURES: "Extracting 67 ML features",
    JobStatus.PREPROCESSING: "Scaling features",
    JobStatus.INFERRING: "Running TabPFN inference",
    JobStatus.GENERATING_REPORT: "Generating analysis report",
    JobStatus.COMPLETE: "Analysis complete",
    JobStatus.FAILED: "Analysis failed",
}

# stage ordering for progress percentage
_STAGE_ORDER = [
    JobStatus.PENDING,
    JobStatus.UPLOADING,
    JobStatus.EXTRACTING,
    JobStatus.BUILDING_FEATURES,
    JobStatus.PREPROCESSING,
    JobStatus.INFERRING,
    JobStatus.GENERATING_REPORT,
    JobStatus.COMPLETE,
]

jobs: dict[str, dict[str, Any]] = {}  # job_id -> job record


def _new_job(pcap_filename: str, pcap_size: int) -> str:
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "pcap_filename": pcap_filename,
        "pcap_size": pcap_size,
        "status": JobStatus.PENDING,
        "status_label": _STATUS_LABELS[JobStatus.PENDING],
        "progress": 0,
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "error": None,
        "pcap_path": None,
        "report_path": None,
        "csv_path": None,
        "report_data": None,
        "flows_df": None,  # in-memory DataFrame for API queries
    }
    return job_id


def _set_stage(job_id: str, stage: JobStatus) -> None:
    job = jobs[job_id]
    job["status"] = stage
    job["status_label"] = _STATUS_LABELS[stage]
    try:
        idx = _STAGE_ORDER.index(stage)
        job["progress"] = round(idx / (len(_STAGE_ORDER) - 1) * 100)
    except ValueError:
        pass


def _fail_job(job_id: str, error: str) -> None:
    job = jobs[job_id]
    job["status"] = JobStatus.FAILED
    job["status_label"] = _STATUS_LABELS[JobStatus.FAILED]
    job["error"] = error
    job["completed_at"] = datetime.now().isoformat()
    job["progress"] = 0


# ---------------------------------------------------------------------------
# Core analysis (runs in a thread via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _run_analysis(job_id: str) -> None:
    """Full PCAP → prediction pipeline, executed in a background thread."""
    job = jobs[job_id]
    pcap_path = Path(job["pcap_path"])
    work_dir = UPLOAD_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        from tabpfn_nids.flows.flow_builder import extract_flows_from_pcap
        from tabpfn_nids.features.feature_pipeline import compute_all_features
        from tabpfn_nids.inference import DynamicInferenceManager, ModelRegistry, ModelInfo
        from tabpfn_nids.pcap.analysis_report import build_report_data, save_json_report
        from tabpfn_nids.pipeline_config import load_config, InferenceConfig

        analysis_start = time.perf_counter()
        job["started_at"] = datetime.now().isoformat()

        # ------------------------------------------------------------------
        # 1. Load artifacts
        # ------------------------------------------------------------------
        model, scaler, feature_names = _load_artifacts()

        # ------------------------------------------------------------------
        # 2. Extract flows from PCAP
        # ------------------------------------------------------------------
        _set_stage(job_id, JobStatus.EXTRACTING)
        flows_path = work_dir / "flows.parquet"
        extraction_summary = extract_flows_from_pcap(
            pcap_path=pcap_path,
            output_path=flows_path,
            backend="scapy",
            flow_timeout=120.0,
            idle_timeout=60.0,
            batch_size=50_000,
        )
        logger.info(
            "Job %s: extracted %d flows from %d packets",
            job_id,
            extraction_summary["total_flows"],
            extraction_summary["total_packets"],
        )
        flows = pd.read_parquet(flows_path)

        # ------------------------------------------------------------------
        # 3. Build ML features
        # ------------------------------------------------------------------
        _set_stage(job_id, JobStatus.BUILDING_FEATURES)
        features = compute_all_features(flows)
        logger.info("Job %s: %d rows × %d cols", job_id, len(features), len(features.columns))

        # ------------------------------------------------------------------
        # 4. Prepare model input
        # ------------------------------------------------------------------
        _set_stage(job_id, JobStatus.PREPROCESSING)
        missing = [c for c in feature_names if c not in features.columns]
        if missing:
            raise ValueError(f"Missing model features: {', '.join(missing)}")

        metadata_cols = [c for c in features.columns if c not in feature_names]
        metadata = features[metadata_cols].copy()
        original_features = features[feature_names].copy()

        X = original_features.copy()
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = scaler.transform(X)

        # ------------------------------------------------------------------
        # 5. Inference
        # ------------------------------------------------------------------
        _set_stage(job_id, JobStatus.INFERRING)

        try:
            cfg = load_config()
            inf_cfg = cfg.inference
        except Exception:
            inf_cfg = InferenceConfig()

        registry = ModelRegistry(
            models_dir=ARTIFACTS_DIR / "models",
            default_schema_path=FEATURE_SCHEMA_PATH,
        )
        registry.register_model(
            ModelInfo(
                model_id="tabpfn_binary_model",
                model_path=MODEL_PATH,
                schema_path=FEATURE_SCHEMA_PATH,
                feature_count=len(feature_names),
                feature_names=feature_names,
            ),
            model_instance=model,
        )

        inference_manager = DynamicInferenceManager(
            max_rows_per_worker=inf_cfg.max_rows_per_worker,
            max_workers=inf_cfg.max_workers,
            executor_type=inf_cfg.executor_type,
            probability_aggregation=inf_cfg.probability_aggregation,
            prediction_threshold=inf_cfg.prediction_threshold,
            enable_ensemble=inf_cfg.enable_ensemble,
            include_per_model_probabilities=inf_cfg.include_per_model_probabilities,
            weights=inf_cfg.weights,
            model_registry=registry,
        )

        inf_result = inference_manager.predict(X_scaled)
        predictions = inf_result.predictions
        probabilities = inf_result.probabilities
        inference_meta = inf_result.metadata

        # ------------------------------------------------------------------
        # 6. Build output DataFrame
        # ------------------------------------------------------------------
        output = metadata.copy()
        for col in original_features.columns:
            output[col] = original_features[col].values
        output["prediction"] = predictions
        output["prediction_label"] = ["Normal" if int(v) == 0 else "Attack" for v in predictions]
        if probabilities is not None and probabilities.shape[1] >= 2:
            output["normal_probability"] = probabilities[:, 0]
            output["attack_probability"] = probabilities[:, 1]

        # ------------------------------------------------------------------
        # 7. Save CSV
        # ------------------------------------------------------------------
        csv_path = work_dir / f"{pcap_path.stem}_predictions.csv"
        output.to_csv(csv_path, index=False)

        # ------------------------------------------------------------------
        # 8. Build report data (reuse existing builder)
        # ------------------------------------------------------------------
        _set_stage(job_id, JobStatus.GENERATING_REPORT)
        total_analysis_seconds = time.perf_counter() - analysis_start

        report_data = build_report_data(
            pcap_path=pcap_path,
            extraction_summary=extraction_summary,
            features=features,
            feature_names=feature_names,
            predictions=predictions,
            probabilities=probabilities,
            total_analysis_seconds=total_analysis_seconds,
            inference_meta=inference_meta,
        )

        # Enrich report with additional UI statistics
        _enrich_report(report_data, output, features, inference_meta)

        json_report_path = work_dir / "analysis_report.json"
        save_json_report(report_data, json_report_path)

        # ------------------------------------------------------------------
        # 9. Finalise
        # ------------------------------------------------------------------
        job["report_path"] = str(json_report_path)
        job["csv_path"] = str(csv_path)
        job["report_data"] = report_data
        # Store flows for paginated table endpoint (keep only needed columns)
        job["flows_df"] = _prepare_flows_table(output)
        job["completed_at"] = datetime.now().isoformat()
        _set_stage(job_id, JobStatus.COMPLETE)
        logger.info("Job %s completed in %.1fs", job_id, total_analysis_seconds)

    except Exception as exc:
        logger.error("Job %s failed: %s", job_id, exc, exc_info=True)
        _fail_job(job_id, str(exc))


def _enrich_report(
    report_data: dict,
    output: pd.DataFrame,
    features: pd.DataFrame,
    inference_meta: dict,
) -> None:
    """Add extra statistics used by the web UI that the base builder omits."""
    total = len(output)
    normal = int((output["prediction"] == 0).sum())
    attack = int((output["prediction"] == 1).sum())

    # Network statistics
    net_stats: dict[str, Any] = {"total_flows": total}

    # Unique IPs
    for col in ("src_ip", "source_ip", "ip_src", "IP_src"):
        if col in output.columns:
            net_stats["unique_src_ips"] = int(output[col].nunique())
            break
    for col in ("dst_ip", "dest_ip", "destination_ip", "ip_dst", "IP_dst"):
        if col in output.columns:
            net_stats["unique_dst_ips"] = int(output[col].nunique())
            break

    # Protocol distribution
    proto_dist: dict[str, int] = {}
    for col in ("protocol", "Protocol", "proto"):
        if col in output.columns:
            vc = output[col].value_counts()
            proto_dist = {str(k): int(v) for k, v in vc.items()}
            break
    if proto_dist:
        net_stats["protocol_distribution"] = proto_dist

    # Flow duration stats
    for col in ("duration", "Duration", "flow_duration"):
        if col in output.columns:
            net_stats["avg_flow_duration"] = round(float(output[col].mean()), 4)
            net_stats["total_duration"] = round(float(output[col].sum()), 4)
            break

    # Byte / packet stats
    fwd_pkt_col = next((c for c in output.columns if "fwd" in c.lower() and "pkt" in c.lower()), None)
    bwd_pkt_col = next((c for c in output.columns if "bwd" in c.lower() and "pkt" in c.lower()), None)
    fwd_byte_col = next((c for c in output.columns if "fwd" in c.lower() and "byte" in c.lower()), None)
    bwd_byte_col = next((c for c in output.columns if "bwd" in c.lower() and "byte" in c.lower()), None)

    for col, key in [
        (fwd_pkt_col, "total_fwd_packets"),
        (bwd_pkt_col, "total_bwd_packets"),
        (fwd_byte_col, "total_fwd_bytes"),
        (bwd_byte_col, "total_bwd_bytes"),
    ]:
        if col:
            net_stats[key] = int(output[col].sum())

    # avg normal / attack probability
    if "normal_probability" in output.columns:
        net_stats["avg_normal_probability"] = round(float(output["normal_probability"].mean()), 4)
    if "attack_probability" in output.columns:
        net_stats["avg_attack_probability"] = round(float(output["attack_probability"].mean()), 4)

    report_data["network_stats"] = net_stats

    # Inference performance card
    if inference_meta:
        report_data["processing_performance"] = {
            "inference_mode": inference_meta.get("inference_mode", "Single model"),
            "workers": inference_meta.get("num_workers", 1),
            "chunks": inference_meta.get("num_chunks", 1),
            "inference_time_seconds": inference_meta.get("total_inference_seconds", 0.0),
            "rows_per_second": inference_meta.get("rows_per_second", 0.0),
            "max_rows_per_worker": inference_meta.get("max_rows_per_worker", 10000),
            "models_used": inference_meta.get("models_used", []),
        }


def _prepare_flows_table(output: pd.DataFrame) -> pd.DataFrame:
    """Select and rename columns for the paginated flow table."""
    col_map = {
        "src_ip": "src_ip",
        "source_ip": "src_ip",
        "ip_src": "src_ip",
        "dst_ip": "dst_ip",
        "dest_ip": "dst_ip",
        "destination_ip": "dst_ip",
        "ip_dst": "dst_ip",
        "src_port": "src_port",
        "source_port": "src_port",
        "sport": "src_port",
        "dst_port": "dst_port",
        "dest_port": "dst_port",
        "destination_port": "dst_port",
        "dport": "dst_port",
        "protocol": "protocol",
        "Protocol": "protocol",
        "proto": "protocol",
        "duration": "duration",
        "Duration": "duration",
        "flow_duration": "duration",
    }

    keep: dict[str, str] = {}
    for src, dst in col_map.items():
        if src in output.columns and dst not in keep.values():
            keep[src] = dst

    # Always include prediction columns
    for col in ["prediction", "prediction_label", "normal_probability", "attack_probability"]:
        if col in output.columns:
            keep[col] = col

    # Add forward/backward packet/byte columns if present
    for col in output.columns:
        cl = col.lower()
        if any(x in cl for x in ["fwd_pkt", "bwd_pkt", "fwd_byte", "bwd_byte"]) and col not in keep:
            keep[col] = col

    result = output[[c for c in keep.keys() if c in output.columns]].copy()
    result = result.rename(columns={k: v for k, v in keep.items() if k in result.columns})
    result.insert(0, "flow_id", range(1, len(result) + 1))

    # Make JSON-serialisable
    for col in result.select_dtypes(include=[np.floating]).columns:
        result[col] = result[col].round(4)
    result = result.replace([np.inf, -np.inf], None)

    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/info", summary="Backend and model information")
async def get_info():
    """Return static model and backend configuration."""
    try:
        _load_artifacts()
        info = _get_model_info()
        info["artifacts_available"] = True
    except Exception as exc:
        info = {
            "model_type": "TabPFN",
            "task": "Binary Network Intrusion Detection",
            "feature_count": 67,
            "artifacts_available": False,
            "error": str(exc),
        }
    return info


@router.post("/upload", summary="Upload a PCAP file", status_code=status.HTTP_201_CREATED)
async def upload_pcap(file: UploadFile = File(...)):
    """Accept a .pcap upload and return a job_id for subsequent analysis."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    fname_lower = file.filename.lower()
    if not (fname_lower.endswith(".pcap") or fname_lower.endswith(".pcapng")):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid file type",
                "detail": "Only .pcap and .pcapng files are accepted.",
            },
        )

    # Read uploaded bytes
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail={"error": "Uploaded file is empty."})

    # Validate PCAP magic bytes
    PCAP_MAGIC = {b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4", b"\x0a\x0d\x0d\x0a"}
    if len(content) < 4 or content[:4] not in PCAP_MAGIC:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid PCAP file",
                "detail": "File does not appear to be a valid PCAP or PCAPNG capture.",
            },
        )

    job_id = _new_job(file.filename, len(content))
    job = jobs[job_id]

    # Save uploaded file
    dest = UPLOAD_DIR / job_id
    dest.mkdir(parents=True, exist_ok=True)
    pcap_path = dest / file.filename
    pcap_path.write_bytes(content)
    job["pcap_path"] = str(pcap_path)
    _set_stage(job_id, JobStatus.UPLOADING)

    return {
        "job_id": job_id,
        "filename": file.filename,
        "size_bytes": len(content),
        "status": job["status"],
    }


@router.post("/analyze/{job_id}", summary="Start PCAP analysis")
async def start_analysis(job_id: str, background_tasks: BackgroundTasks):
    """Trigger analysis for an uploaded PCAP. Returns immediately; poll /status/{job_id}."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = jobs[job_id]
    if job["status"] == JobStatus.COMPLETE:
        raise HTTPException(status_code=400, detail="Analysis already complete.")
    if job["status"] not in (JobStatus.PENDING, JobStatus.UPLOADING):
        raise HTTPException(status_code=400, detail=f"Job is in state '{job['status']}', cannot restart.")
    if not job.get("pcap_path"):
        raise HTTPException(status_code=400, detail="No PCAP file associated with this job.")

    background_tasks.add_task(_run_analysis_async, job_id)
    return {"job_id": job_id, "status": job["status"], "message": "Analysis started."}


async def _run_analysis_async(job_id: str) -> None:
    """Offload blocking analysis to a thread pool."""
    await asyncio.to_thread(_run_analysis, job_id)


@router.get("/status/{job_id}", summary="Poll analysis job status")
async def get_job_status(job_id: str):
    """Return current status and progress percentage for a job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "status_label": job["status_label"],
        "progress": job["progress"],
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
        "error": job["error"],
        "pcap_filename": job["pcap_filename"],
        "pcap_size": job["pcap_size"],
    }


@router.get("/report/{job_id}", summary="Get full analysis report")
async def get_report(job_id: str):
    """Return the complete JSON analysis report for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = jobs[job_id]
    if job["status"] != JobStatus.COMPLETE:
        raise HTTPException(
            status_code=400,
            detail=f"Report not ready. Current status: {job['status']}",
        )
    report_data = job.get("report_data")
    if report_data is None and job.get("report_path"):
        with open(job["report_path"], "r", encoding="utf-8") as f:
            report_data = json.load(f)
    if report_data is None:
        raise HTTPException(status_code=500, detail="Report data unavailable.")
    return JSONResponse(content=report_data)


@router.get("/download/{job_id}", summary="Download prediction CSV")
async def download_csv(job_id: str):
    """Stream the flow-level prediction CSV for download."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = jobs[job_id]
    if job["status"] != JobStatus.COMPLETE:
        raise HTTPException(status_code=400, detail="Analysis not complete.")
    csv_path = job.get("csv_path")
    if not csv_path or not Path(csv_path).exists():
        raise HTTPException(status_code=404, detail="CSV file not found.")
    filename = f"{Path(csv_path).name}"
    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/flows/{job_id}", summary="Paginated flow-level predictions")
async def get_flows(
    job_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    search: str = Query(default=""),
    filter_label: str = Query(default="all", pattern="^(all|normal|attack)$"),
    sort_by: str = Query(default="flow_id"),
    sort_desc: bool = Query(default=False),
):
    """Return a paginated, filterable, sortable slice of flow-level predictions."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = jobs[job_id]
    if job["status"] != JobStatus.COMPLETE:
        raise HTTPException(status_code=400, detail="Analysis not complete.")

    df: pd.DataFrame | None = job.get("flows_df")
    if df is None:
        raise HTTPException(status_code=500, detail="Flow data unavailable.")

    # Filter by label
    if filter_label == "attack":
        df = df[df["prediction_label"] == "Attack"]
    elif filter_label == "normal":
        df = df[df["prediction_label"] == "Normal"]

    # Search across string columns
    if search:
        mask = pd.Series([False] * len(df), index=df.index)
        for col in df.select_dtypes(include="object").columns:
            mask |= df[col].astype(str).str.contains(search, case=False, na=False)
        df = df[mask]

    # Sort
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=not sort_desc)

    total = len(df)
    total_pages = max(1, math.ceil(total / page_size))
    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end]

    return {
        "total": total,
        "total_pages": total_pages,
        "page": page,
        "page_size": page_size,
        "rows": page_df.where(page_df.notna(), None).to_dict(orient="records"),
    }
