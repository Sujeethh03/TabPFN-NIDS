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
MULTICLASS_MODEL_PATH = ARTIFACTS_DIR / "models" / "tabpfn_multiclass_model.pkl"
MULTICLASS_SCHEMA_PATH = ARTIFACTS_DIR / "models" / "multiclass_feature_schema.json"
MULTICLASS_SCALER_PATH = ARTIFACTS_DIR / "preprocessing" / "multiclass_scaler.pkl"
MULTICLASS_LABEL_ENCODER_PATH = ARTIFACTS_DIR / "models" / "multiclass_label_encoder.json"

UPLOAD_DIR = PROJECT_ROOT / "results" / "pcap_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Global in-process artifact cache (load once, reuse)
# ---------------------------------------------------------------------------

_artifact_cache: dict[str, Any] = {}


def _load_artifacts() -> tuple:
    """Load trained binary and multiclass models, scalers, and schemas (cached)."""
    if "model" not in _artifact_cache:
        logger.info("Loading TabPFN binary model artifacts...")
        with open(MODEL_PATH, "rb") as f:
            _artifact_cache["model"] = pickle.load(f)
        with open(FEATURE_SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)
        _artifact_cache["feature_names"] = schema["feature_names"]
        with open(SCALER_PATH, "rb") as f:
            _artifact_cache["scaler"] = pickle.load(f)

        # Load multiclass artifacts if present
        if MULTICLASS_MODEL_PATH.exists():
            logger.info("Loading TabPFN multiclass model artifacts...")
            with open(MULTICLASS_MODEL_PATH, "rb") as f:
                _artifact_cache["multiclass_model"] = pickle.load(f)
            if MULTICLASS_SCALER_PATH.exists():
                with open(MULTICLASS_SCALER_PATH, "rb") as f:
                    _artifact_cache["multiclass_scaler"] = pickle.load(f)
            else:
                _artifact_cache["multiclass_scaler"] = _artifact_cache["scaler"]

            if MULTICLASS_LABEL_ENCODER_PATH.exists():
                with open(MULTICLASS_LABEL_ENCODER_PATH, "r", encoding="utf-8") as f:
                    le = json.load(f)
                _artifact_cache["multiclass_label_encoder"] = le
                _artifact_cache["multiclass_classes"] = le.get("classes", [])

        logger.info(
            "Artifacts loaded. Model expects %d features.",
            len(_artifact_cache["feature_names"]),
        )
    return (
        _artifact_cache["model"],
        _artifact_cache["scaler"],
        _artifact_cache["feature_names"],
        _artifact_cache.get("multiclass_model"),
        _artifact_cache.get("multiclass_scaler"),
        _artifact_cache.get("multiclass_label_encoder"),
    )


def _get_model_info() -> dict[str, Any]:
    """Return model metadata for binary and multiclass classifiers."""
    feature_names = _artifact_cache.get("feature_names", [])
    mc_classes = _artifact_cache.get("multiclass_classes", [])
    return {
        "model_type": "TabPFN",
        "task": "Dual Binary & Multiclass Network Intrusion Detection",
        "feature_count": len(feature_names) if feature_names else 67,
        "artifacts_dir": str(ARTIFACTS_DIR),
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
        "multiclass_model_path": str(MULTICLASS_MODEL_PATH) if MULTICLASS_MODEL_PATH.exists() else None,
        "multiclass_classes": mc_classes,
        "num_multiclass_classes": len(mc_classes),
        "multiclass_available": _artifact_cache.get("multiclass_model") is not None,
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
        "html_path": None,
        "docx_path": None,
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
        from tabpfn_nids.pcap.analysis_report import (
            build_report_data,
            save_json_report,
            save_html_report,
            save_docx_report,
        )
        from tabpfn_nids.evaluation import (
            decode_attack_category,
            get_attack_category_mapping,
        )
        from tabpfn_nids.pipeline_config import load_config, InferenceConfig

        analysis_start = time.perf_counter()
        job["started_at"] = datetime.now().isoformat()

        # ------------------------------------------------------------------
        # 1. Load artifacts
        # ------------------------------------------------------------------
        model, scaler, feature_names, mc_model, mc_scaler, mc_le = _load_artifacts()

        # ------------------------------------------------------------------
        # 2. Extract flows from PCAP
        # ------------------------------------------------------------------
        _set_stage(job_id, JobStatus.EXTRACTING)
        t_ext_start = time.perf_counter()
        flows_path = work_dir / "flows.parquet"
        extraction_summary = extract_flows_from_pcap(
            pcap_path=pcap_path,
            output_path=flows_path,
            backend="scapy",
            flow_timeout=120.0,
            idle_timeout=60.0,
            batch_size=50_000,
        )
        extraction_seconds = time.perf_counter() - t_ext_start
        logger.info(
            "Job %s: extracted %d flows from %d packets in %.2fs",
            job_id,
            extraction_summary["total_flows"],
            extraction_summary["total_packets"],
            extraction_seconds,
        )
        flows = pd.read_parquet(flows_path)

        # ------------------------------------------------------------------
        # 3. Build ML features
        # ------------------------------------------------------------------
        _set_stage(job_id, JobStatus.BUILDING_FEATURES)
        t_feat_start = time.perf_counter()
        features = compute_all_features(flows)
        feature_engineering_seconds = time.perf_counter() - t_feat_start
        logger.info("Job %s: %d rows × %d cols in %.2fs", job_id, len(features), len(features.columns), feature_engineering_seconds)

        # ------------------------------------------------------------------
        # 4. Prepare model input
        # ------------------------------------------------------------------
        _set_stage(job_id, JobStatus.PREPROCESSING)
        t_prep_start = time.perf_counter()
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
        X_mc_scaled = mc_scaler.transform(X) if mc_scaler is not None else X_scaled
        preprocessing_seconds = time.perf_counter() - t_prep_start

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

        t_bin_start = time.perf_counter()
        inf_result = inference_manager.predict(X_scaled, model_ids=["tabpfn_binary_model"])
        binary_inference_seconds = time.perf_counter() - t_bin_start
        predictions = inf_result.predictions
        probabilities = inf_result.probabilities
        inference_meta = inf_result.metadata

        # Multiclass inference
        multi_preds = None
        multi_probs = None
        predicted_categories = None
        category_confidences = None
        prediction_statuses = None
        is_conflicts = None
        conflict_reasons = None
        classes = None
        multiclass_inference_seconds = 0.0

        if mc_model is not None and mc_le is not None:
            logger.info("Running TabPFN multiclass inference on %d rows...", len(X_mc_scaled))
            classes = mc_le.get("classes", [])
            mapping = get_attack_category_mapping(mc_le)

            t_mc_start = time.perf_counter()
            mc_batch_size = 200
            mc_probs_list = []
            for b_idx in range(0, len(X_mc_scaled), mc_batch_size):
                b_batch = X_mc_scaled[b_idx : b_idx + mc_batch_size]
                b_probs = mc_model.predict_proba(b_batch)
                mc_probs_list.append(b_probs)
            multi_probs = np.vstack(mc_probs_list)
            multi_preds = np.argmax(multi_probs, axis=1)
            multiclass_inference_seconds = time.perf_counter() - t_mc_start

            predicted_categories = [
                mapping.get(int(p), decode_attack_category(p, mc_le)) for p in multi_preds
            ]
            category_confidences = np.max(multi_probs, axis=1)
            uncertainty_threshold = float(mc_le.get("confidence_threshold", 0.50))

            prediction_statuses = []
            is_conflicts = []
            conflict_reasons = []
            for i, p_cat in enumerate(predicted_categories):
                bin_lbl = "Normal" if int(predictions[i]) == 0 else "Attack"
                conf = float(category_confidences[i])
                has_conflict = False
                reason = None
                if bin_lbl == "Attack" and str(p_cat).lower() == "normal":
                    has_conflict = True
                    reason = "Binary model predicted Attack, but multiclass model classified flow as Normal."
                elif bin_lbl == "Normal" and str(p_cat).lower() != "normal":
                    has_conflict = True
                    reason = f"Binary model predicted Normal, but multiclass model detected {p_cat}."

                is_conflicts.append(has_conflict)
                conflict_reasons.append(reason)

                if has_conflict:
                    status = "Uncertain"
                elif conf >= uncertainty_threshold:
                    status = "Classified"
                else:
                    status = "Uncertain"
                prediction_statuses.append(status)

        # ------------------------------------------------------------------
        # 6. Build output DataFrame
        # ------------------------------------------------------------------
        output = metadata.copy()
        for col in original_features.columns:
            output[col] = original_features[col].values
        output["prediction"] = predictions
        output["prediction_label"] = ["Normal" if int(v) == 0 else "Attack" for v in predictions]
        output["binary_prediction"] = output["prediction_label"]
        if probabilities is not None and probabilities.shape[1] >= 2:
            output["normal_probability"] = probabilities[:, 0]
            output["attack_probability"] = probabilities[:, 1]
            output["binary_attack_probability"] = probabilities[:, 1]

        if multi_preds is not None:
            output["predicted_attack_category_id"] = [int(p) for p in multi_preds]

        if predicted_categories is not None:
            output["predicted_attack_category"] = predicted_categories
            output["attack_category_confidence"] = category_confidences
            output["category_confidence"] = category_confidences
            output["prediction_status"] = prediction_statuses
            output["classification_status"] = prediction_statuses
            if is_conflicts is not None:
                output["is_model_conflict"] = is_conflicts
                output["conflict_reason"] = conflict_reasons

        if multi_probs is not None and classes is not None:
            for idx, cls_name in enumerate(classes):
                clean_cls = f"prob_{cls_name.lower().replace(' ', '_')}"
                output[clean_cls] = multi_probs[:, idx]

        # ------------------------------------------------------------------
        # 7. Save CSV
        # ------------------------------------------------------------------
        csv_path = work_dir / f"{pcap_path.stem}_predictions.csv"
        output.to_csv(csv_path, index=False)

        # ------------------------------------------------------------------
        # 8. Build report data (reuse existing builder)
        # ------------------------------------------------------------------
        _set_stage(job_id, JobStatus.GENERATING_REPORT)
        t_rep_start = time.perf_counter()
        total_analysis_seconds = time.perf_counter() - analysis_start

        perf_breakdown = {
            "extraction_seconds": round(extraction_seconds, 3),
            "feature_engineering_seconds": round(feature_engineering_seconds, 3),
            "preprocessing_seconds": round(preprocessing_seconds, 3),
            "binary_inference_seconds": round(binary_inference_seconds, 3),
            "multiclass_inference_seconds": round(multiclass_inference_seconds, 3),
            "total_inference_seconds": round(binary_inference_seconds + multiclass_inference_seconds, 3),
            "total_analysis_seconds": round(total_analysis_seconds, 3),
            "flows_processed": len(output),
            "flows_per_second": round(len(output) / total_analysis_seconds, 2) if total_analysis_seconds > 0 else 0.0,
            "inference_throughput_flows_per_sec": round(len(output) / max(1e-6, binary_inference_seconds + multiclass_inference_seconds), 2),
        }

        report_data = build_report_data(
            pcap_path=pcap_path,
            extraction_summary=extraction_summary,
            features=features,
            feature_names=feature_names,
            predictions=predictions,
            probabilities=probabilities,
            total_analysis_seconds=total_analysis_seconds,
            inference_meta=inference_meta,
            multiclass_predictions=multi_preds,
            multiclass_probabilities=multi_probs,
            class_names=classes,
            predicted_categories=predicted_categories,
            category_confidences=category_confidences,
            prediction_statuses=prediction_statuses,
            uncertainty_threshold=0.50,
            performance_breakdown=perf_breakdown,
        )

        # Enrich report with additional UI statistics
        _enrich_report(report_data, output, features, inference_meta, perf_breakdown=perf_breakdown)

        json_report_path = work_dir / "analysis_report.json"
        save_json_report(report_data, json_report_path)

        html_report_path = work_dir / f"{pcap_path.stem}_analysis_report.html"
        save_html_report(report_data, html_report_path)

        docx_report_path = work_dir / f"{pcap_path.stem}_analysis_report.docx"
        save_docx_report(report_data, docx_report_path)

        # ------------------------------------------------------------------
        # 9. Finalise
        # ------------------------------------------------------------------
        reporting_seconds = time.perf_counter() - t_rep_start
        report_data["performance"]["reporting_seconds"] = round(reporting_seconds, 3)

        job["report_path"] = str(json_report_path)
        job["html_path"] = str(html_report_path)
        job["docx_path"] = str(docx_report_path)
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
    inference_meta: dict | None,
    perf_breakdown: dict | None = None,
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

    # Multiclass counts
    if "predicted_attack_category" in output.columns:
        cat_counts = output["predicted_attack_category"].value_counts().to_dict()
        net_stats["attack_category_counts"] = {str(k): int(v) for k, v in cat_counts.items()}
    if "prediction_status" in output.columns:
        stat_counts = output["prediction_status"].value_counts().to_dict()
        net_stats["prediction_status_counts"] = {str(k): int(v) for k, v in stat_counts.items()}
    if "is_model_conflict" in output.columns:
        net_stats["model_conflicts_count"] = int(output["is_model_conflict"].sum())

    report_data["network_stats"] = net_stats

    # Inference performance card
    tot_sec = report_data.get("performance", {}).get("total_analysis_seconds", 0.0)
    inf_sec = inference_meta.get("total_inference_seconds", 0.0) if inference_meta else 0.0
    if perf_breakdown and "total_inference_seconds" in perf_breakdown:
        inf_sec = perf_breakdown["total_inference_seconds"]
    inf_rps = inference_meta.get("rows_per_second", 0.0) if inference_meta else 0.0
    if (inf_rps == 0.0 or inf_rps < 1.0) and inf_sec > 0:
        inf_rps = round(len(output) / inf_sec, 2)
    overall_rps = round(len(output) / tot_sec, 2) if tot_sec > 0 else inf_rps

    perf_info = {
        "inference_mode": inference_meta.get("inference_mode", "Single model") if inference_meta else "Single model",
        "workers": inference_meta.get("num_workers", 1) if inference_meta else 1,
        "chunks": inference_meta.get("num_chunks", 1) if inference_meta else 1,
        "inference_time_seconds": round(inf_sec, 2),
        "rows_per_second": inf_rps,
        "overall_rows_per_second": overall_rps,
        "max_rows_per_worker": inference_meta.get("max_rows_per_worker", 10000) if inference_meta else 10000,
        "models_used": inference_meta.get("models_used", ["tabpfn_binary_model", "tabpfn_multiclass_model"]) if inference_meta else ["tabpfn_binary_model"],
    }
    if perf_breakdown:
        perf_info.update(perf_breakdown)
    report_data["processing_performance"] = perf_info


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
    for col in [
        "prediction",
        "prediction_label",
        "binary_prediction",
        "normal_probability",
        "attack_probability",
        "binary_attack_probability",
        "predicted_attack_category_id",
        "predicted_attack_category",
        "attack_category_confidence",
        "category_confidence",
        "prediction_status",
        "classification_status",
        "is_model_conflict",
        "conflict_reason",
    ]:
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


@router.get("/download/{job_id}", summary="Download prediction results or reports")
async def download_file(
    job_id: str,
    format: str = Query(default="csv", pattern="^(csv|json|html|docx)$"),
):
    """Download analysis results in CSV, JSON, HTML, or Word DOCX format."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = jobs[job_id]
    if job["status"] != JobStatus.COMPLETE:
        raise HTTPException(status_code=400, detail="Analysis not complete.")

    fmt = format.lower()
    if fmt == "csv":
        path = job.get("csv_path")
        media = "text/csv"
    elif fmt == "json":
        path = job.get("report_path")
        media = "application/json"
    elif fmt == "html":
        path = job.get("html_path")
        media = "text/html"
    elif fmt == "docx":
        path = job.get("docx_path")
        media = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        raise HTTPException(status_code=400, detail="Invalid format requested.")

    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail=f"{format.upper()} report file not found.")
    filename = Path(path).name
    return FileResponse(
        path,
        media_type=media,
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
    filter_category: str = Query(default="all"),
    filter_status: str = Query(default="all"),
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

    # Filter by binary label
    if filter_label == "attack":
        df = df[df["prediction_label"] == "Attack"]
    elif filter_label == "normal":
        df = df[df["prediction_label"] == "Normal"]

    # Filter by multiclass category
    if filter_category.lower() != "all" and "predicted_attack_category" in df.columns:
        df = df[df["predicted_attack_category"].astype(str).str.lower() == filter_category.lower()]

    # Filter by prediction status (Classified, Uncertain)
    if filter_status.lower() != "all" and "prediction_status" in df.columns:
        df = df[df["prediction_status"].astype(str).str.lower() == filter_status.lower()]

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
