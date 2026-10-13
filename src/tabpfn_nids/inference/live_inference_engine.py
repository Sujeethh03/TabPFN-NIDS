"""Asynchronous inference engine and worker pool for live network intrusion detection."""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from tabpfn_nids.config import PROJECT_ROOT
from tabpfn_nids.detection.window_manager import DetectionWindow
from tabpfn_nids.evaluation.metrics import (
    decode_attack_category,
    get_attack_category_mapping,
)
from tabpfn_nids.features.feature_pipeline import compute_all_features
from tabpfn_nids.firewall.firewall_manager import FirewallManager
from tabpfn_nids.flows.flow_builder import FlowRecord
from tabpfn_nids.inference import DynamicInferenceManager, ModelInfo, ModelRegistry

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACTS_DIR / "models" / "tabpfn_binary_model.pkl"
DEFAULT_SCHEMA_PATH = DEFAULT_ARTIFACTS_DIR / "models" / "model_feature_schema.json"
DEFAULT_SCALER_PATH = DEFAULT_ARTIFACTS_DIR / "preprocessing" / "fitted_scaler.pkl"

DEFAULT_MULTICLASS_MODEL_PATH = DEFAULT_ARTIFACTS_DIR / "models" / "tabpfn_multiclass_model.pkl"
DEFAULT_MULTICLASS_SCHEMA_PATH = DEFAULT_ARTIFACTS_DIR / "models" / "multiclass_feature_schema.json"
DEFAULT_MULTICLASS_SCALER_PATH = DEFAULT_ARTIFACTS_DIR / "preprocessing" / "multiclass_scaler.pkl"
DEFAULT_MULTICLASS_LABEL_ENCODER_PATH = DEFAULT_ARTIFACTS_DIR / "models" / "multiclass_label_encoder.json"


@dataclass
class LiveDetectionResult:
    """Detection result for a single traffic refresh window."""

    window_id: str
    timestamp: str
    prediction: str                      # "BENIGN" or "ATTACK"
    confidence: float                    # 0.0 - 1.0
    flows_analyzed: int
    packets_analyzed: int
    attack_count: int
    normal_count: int
    latencies: dict[str, float] = field(default_factory=dict)
    flow_predictions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "timestamp": self.timestamp,
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "flows_analyzed": self.flows_analyzed,
            "packets_analyzed": self.packets_analyzed,
            "attack_count": self.attack_count,
            "normal_count": self.normal_count,
            "latencies": {k: round(v, 4) for k, v in self.latencies.items()},
            "flow_predictions": self.flow_predictions,
        }


@dataclass
class LiveInferenceMetrics:
    """Performance telemetry for the live inference engine."""

    total_windows_processed: int = 0
    total_flows_processed: int = 0
    total_attacks_detected: int = 0
    inference_jobs_failed: int = 0
    avg_inference_latency_ms: float = 0.0
    avg_feature_extraction_latency_ms: float = 0.0
    avg_e2e_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_windows_processed": self.total_windows_processed,
            "total_flows_processed": self.total_flows_processed,
            "total_attacks_detected": self.total_attacks_detected,
            "inference_jobs_failed": self.inference_jobs_failed,
            "avg_inference_latency_ms": round(self.avg_inference_latency_ms, 2),
            "avg_feature_extraction_latency_ms": round(self.avg_feature_extraction_latency_ms, 2),
            "avg_e2e_latency_ms": round(self.avg_e2e_latency_ms, 2),
        }


def extract_and_preprocess_window(
    flows: list[FlowRecord],
    feature_names: list[str],
    scaler: Any,
    multiclass_scaler: Any | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Extract features from flow records and preprocess using the fitted scalers.

    Ensures 100% schema alignment with the offline PCAP pipeline.
    Port numbers and IP addresses are preserved solely as metadata and are strictly
    excluded from model inputs to ensure port-independent behavioral detection.
    """
    if not flows:
        empty_meta = pd.DataFrame()
        empty_x = np.empty((0, len(feature_names)))
        return empty_meta, empty_x, empty_x

    # Convert FlowRecords to DataFrame
    flow_dicts = [f.to_dict() for f in flows]
    raw_df = pd.DataFrame(flow_dicts)

    # Compute all feature groups
    engineered_df = compute_all_features(raw_df)

    # Validate feature columns
    missing = [c for c in feature_names if c not in engineered_df.columns]
    if missing:
        raise ValueError(f"Missing required model features in live extraction: {missing}")

    X = engineered_df[feature_names].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Apply existing fitted scaler for binary model
    X_scaled = scaler.transform(X)

    # Apply multiclass scaler if available, otherwise reuse binary scaler
    if multiclass_scaler is not None:
        X_mc_scaled = multiclass_scaler.transform(X)
    else:
        X_mc_scaled = X_scaled

    metadata_cols = [
        c for c in [
            "flow_id", "src_ip", "dst_ip", "src_port", "dst_port",
            "protocol", "protocol_name", "duration", "total_packets", "total_bytes"
        ] if c in engineered_df.columns
    ]
    metadata_df = engineered_df[metadata_cols].copy()

    return metadata_df, X_scaled, X_mc_scaled


class LiveInferenceEngine:
    """Producer-consumer inference engine that processes detection windows concurrently.

    Supports dual TabPFN models:
    - TabPFN Binary Classifier (Normal vs Attack)
    - TabPFN Multiclass Classifier (9 attack categories from learned flow characteristics, independent of ports)
    - Integrated Firewall defense hook for confirmed malicious source IPs.
    """

    def __init__(
        self,
        window_queue: asyncio.Queue[DetectionWindow],
        max_workers: int = 2,
        model_path: Path | str | None = None,
        schema_path: Path | str | None = None,
        scaler_path: Path | str | None = None,
        multiclass_model_path: Path | str | None = None,
        multiclass_scaler_path: Path | str | None = None,
        multiclass_label_encoder_path: Path | str | None = None,
        firewall_manager: FirewallManager | None = None,
        result_callback: Callable[[LiveDetectionResult], Any] | None = None,
    ) -> None:
        self.window_queue = window_queue
        self.max_workers = max(1, int(max_workers))
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.schema_path = Path(schema_path) if schema_path else DEFAULT_SCHEMA_PATH
        self.scaler_path = Path(scaler_path) if scaler_path else DEFAULT_SCALER_PATH

        self.multiclass_model_path = (
            Path(multiclass_model_path) if multiclass_model_path else DEFAULT_MULTICLASS_MODEL_PATH
        )
        self.multiclass_scaler_path = (
            Path(multiclass_scaler_path) if multiclass_scaler_path else DEFAULT_MULTICLASS_SCALER_PATH
        )
        self.multiclass_label_encoder_path = (
            Path(multiclass_label_encoder_path)
            if multiclass_label_encoder_path
            else DEFAULT_MULTICLASS_LABEL_ENCODER_PATH
        )

        self.firewall_manager = firewall_manager
        self.result_callback = result_callback

        self.metrics = LiveInferenceMetrics()
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._stop_event = asyncio.Event()

        # Shared preloaded artifacts
        self.model: Any = None
        self.scaler: Any = None
        self.feature_names: list[str] = []
        self.inference_manager: DynamicInferenceManager | None = None

        self.multiclass_model: Any = None
        self.multiclass_scaler: Any = None
        self.multiclass_label_encoder: dict[str, Any] | None = None
        self.multiclass_classes: list[str] = []
        self._artifacts_loaded = False

        # Deduplication tracker for detected flow events (flow_id -> last_status)
        self._processed_flow_events: dict[str, str] = {}

    def load_artifacts(self) -> None:
        """Load trained TabPFN models, schemas, and scalers into memory once."""
        if self._artifacts_loaded:
            return

        logger.info("Loading model artifacts for live inference...")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Binary model file not found: {self.model_path}")
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Feature schema not found: {self.schema_path}")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")

        # 1. Binary model artifacts
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(self.schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        self.feature_names = schema["feature_names"]

        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        registry = ModelRegistry()
        registry.register_model(
            ModelInfo(
                model_id="tabpfn_binary_model",
                model_path=self.model_path,
                schema_path=self.schema_path,
                feature_count=len(self.feature_names),
                feature_names=self.feature_names,
            ),
            model_instance=self.model,
        )

        self.inference_manager = DynamicInferenceManager(
            max_rows_per_worker=10_000,
            max_workers=self.max_workers,
            enable_ensemble=False,
            prediction_threshold=0.5,
            model_registry=registry,
        )

        # 2. Multiclass model artifacts (if present)
        if self.multiclass_model_path.exists():
            try:
                logger.info("Loading multiclass TabPFN model from %s...", self.multiclass_model_path)
                with open(self.multiclass_model_path, "rb") as f:
                    self.multiclass_model = pickle.load(f)

                if self.multiclass_scaler_path.exists():
                    with open(self.multiclass_scaler_path, "rb") as f:
                        self.multiclass_scaler = pickle.load(f)
                else:
                    self.multiclass_scaler = self.scaler

                if self.multiclass_label_encoder_path.exists():
                    with open(self.multiclass_label_encoder_path, "r", encoding="utf-8") as f:
                        self.multiclass_label_encoder = json.load(f)
                    self.multiclass_classes = self.multiclass_label_encoder.get("classes", [])

                logger.info(
                    "Multiclass TabPFN model loaded with %d classes: %s",
                    len(self.multiclass_classes),
                    self.multiclass_classes,
                )
            except Exception as exc:
                logger.warning("Could not load multiclass model: %s. Using binary model only.", exc)
                self.multiclass_model = None

        self._artifacts_loaded = True
        logger.info(
            "Live inference artifacts loaded successfully (%d features, multiclass=%s)",
            len(self.feature_names),
            self.multiclass_model is not None,
        )

    def start(self) -> None:
        """Start worker tasks to consume windows from the queue."""
        if self._worker_tasks:
            return

        self.load_artifacts()
        self._stop_event.clear()

        for i in range(self.max_workers):
            task = asyncio.create_task(
                self._worker_loop(i + 1), name=f"inference-worker-{i+1}"
            )
            self._worker_tasks.append(task)

        logger.info("LiveInferenceEngine started with %d workers.", self.max_workers)

    async def _worker_loop(self, worker_id: int) -> None:
        """Continuously pulls detection windows and performs inference."""
        logger.debug("Inference worker %d started.", worker_id)

        while not self._stop_event.is_set():
            try:
                try:
                    window = await asyncio.wait_for(
                        self.window_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                queue_latency = time.time() - window.created_at

                try:
                    result = await asyncio.to_thread(
                        self._process_window_sync, window, queue_latency
                    )
                    self.metrics.total_windows_processed += 1
                    self.metrics.total_flows_processed += result.flows_analyzed
                    self.metrics.total_attacks_detected += result.attack_count

                    if self.result_callback:
                        if asyncio.iscoroutinefunction(self.result_callback):
                            await self.result_callback(result)
                        else:
                            self.result_callback(result)

                except Exception as exc:
                    self.metrics.inference_jobs_failed += 1
                    logger.error(
                        "Worker %d failed processing window %s: %s",
                        worker_id,
                        window.window_id,
                        exc,
                        exc_info=True,
                    )
                finally:
                    self.window_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Unexpected error in worker %d: %s", worker_id, exc)

        logger.debug("Inference worker %d stopped.", worker_id)

    def _process_window_sync(
        self, window: DetectionWindow, queue_latency: float
    ) -> LiveDetectionResult:
        """Synchronous feature extraction, binary & multiclass model inference, and firewall response."""
        t_start = time.perf_counter()
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if not window.flows:
            return LiveDetectionResult(
                window_id=window.window_id,
                timestamp=now_iso,
                prediction="BENIGN",
                confidence=1.0,
                flows_analyzed=0,
                packets_analyzed=window.packets_analyzed,
                attack_count=0,
                normal_count=0,
                latencies={
                    "queue_latency": queue_latency,
                    "feature_extraction_latency": 0.0,
                    "inference_latency": 0.0,
                    "total_latency": time.perf_counter() - t_start,
                },
                flow_predictions=[],
            )

        # 1. Feature extraction & scaling
        t_feat_start = time.perf_counter()
        metadata_df, X_scaled, X_mc_scaled = extract_and_preprocess_window(
            window.flows, self.feature_names, self.scaler, self.multiclass_scaler
        )
        feat_latency = time.perf_counter() - t_feat_start

        # 2. Binary Model Inference
        t_inf_start = time.perf_counter()
        assert self.inference_manager is not None
        inf_result = self.inference_manager.predict(X_scaled)
        bin_inf_latency = time.perf_counter() - t_inf_start
        predictions = inf_result.predictions
        probabilities = inf_result.probabilities

        # 3. Multiclass Model Inference (behavioral classification without port bias)
        multi_preds = None
        multi_probs = None
        predicted_categories = None
        category_confidences = None
        prediction_statuses = None

        if self.multiclass_model is not None and self.multiclass_label_encoder is not None:
            t_mc_start = time.perf_counter()
            mapping = get_attack_category_mapping(self.multiclass_label_encoder)

            mc_batch_size = 200
            mc_probs_list = []
            for b_idx in range(0, len(X_mc_scaled), mc_batch_size):
                b_batch = X_mc_scaled[b_idx : b_idx + mc_batch_size]
                b_probs = self.multiclass_model.predict_proba(b_batch)
                mc_probs_list.append(b_probs)

            multi_probs = np.vstack(mc_probs_list)
            multi_preds = np.argmax(multi_probs, axis=1)
            category_confidences = np.max(multi_probs, axis=1)
            predicted_categories = [
                mapping.get(int(p), decode_attack_category(p, self.multiclass_label_encoder))
                for p in multi_preds
            ]
            mc_threshold = float(self.multiclass_label_encoder.get("confidence_threshold", 0.50))

            prediction_statuses = []
            for i, p_cat in enumerate(predicted_categories):
                bin_lbl = "Normal" if int(predictions[i]) == 0 else "Attack"
                conf = float(category_confidences[i])
                has_conflict = (bin_lbl == "Attack" and str(p_cat).lower() == "normal") or (
                    bin_lbl == "Normal" and str(p_cat).lower() != "normal"
                )
                if has_conflict or conf < mc_threshold:
                    prediction_statuses.append("Uncertain")
                else:
                    prediction_statuses.append("Classified")
        else:
            # Fallback if multiclass model is not available: Do not invent specific attack types
            predicted_categories = [
                "Normal" if int(p) == 0 else "Attack (type unknown)" for p in predictions
            ]
            category_confidences = [
                float(probabilities[i, 1]) if (probabilities is not None and probabilities.shape[1] >= 2) else 0.5
                for i in range(len(predictions))
            ]
            prediction_statuses = ["Classified" if int(p) == 0 else "Uncertain" for p in predictions]

        # 4. Construct flow-level predictions and evaluate Firewall Response
        flow_predictions: list[dict[str, Any]] = []
        attack_count = 0
        normal_count = 0

        for i in range(len(predictions)):
            bin_pred = int(predictions[i])
            bin_label = "Attack" if bin_pred == 1 else "Normal"
            attack_prob = (
                float(probabilities[i, 1])
                if (probabilities is not None and probabilities.shape[1] >= 2)
                else (1.0 if bin_pred == 1 else 0.0)
            )
            cat_name = predicted_categories[i]
            cat_conf = float(category_confidences[i])
            status_str = prediction_statuses[i]

            flow_id = str(metadata_df.iloc[i].get("flow_id", f"flow-{i+1}"))
            src_ip = str(metadata_df.iloc[i].get("src_ip", "0.0.0.0"))
            dst_ip = str(metadata_df.iloc[i].get("dst_ip", "0.0.0.0"))
            src_port = int(metadata_df.iloc[i].get("src_port", 0) or 0)
            dst_port = int(metadata_df.iloc[i].get("dst_port", 0) or 0)
            protocol = str(metadata_df.iloc[i].get("protocol_name", metadata_df.iloc[i].get("protocol", "TCP")))
            duration = float(metadata_df.iloc[i].get("duration", 0.0) or 0.0)
            total_pkts = int(metadata_df.iloc[i].get("total_packets", 0) or 0)
            total_bytes = int(metadata_df.iloc[i].get("total_bytes", 0) or 0)

            if bin_pred == 1:
                attack_count += 1
            else:
                normal_count += 1

            # Firewall Response Evaluation
            firewall_action = "NONE"
            firewall_result = "Normal flow; no action required."
            if self.firewall_manager is not None:
                # Deduplication: do not re-run firewall action for same flow event if already evaluated
                flow_event_key = f"{flow_id}:{bin_label}:{cat_name}"
                is_attack = (bin_pred == 1)
                eff_conf = max(attack_prob, cat_conf)

                fw_event = self.firewall_manager.process_attack_event(
                    source_ip=src_ip,
                    attack_classification=cat_name,
                    confidence=eff_conf,
                    flow_id=flow_id,
                    is_attack=is_attack,
                )
                firewall_action = fw_event.firewall_action
                firewall_result = fw_event.action_result

            flow_predictions.append({
                "flow_id": flow_id,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": src_port,
                "dst_port": dst_port,
                "protocol": protocol,
                "duration": round(duration, 4),
                "total_packets": total_pkts,
                "total_bytes": total_bytes,
                "prediction": bin_pred,
                "prediction_label": bin_label,
                "binary_attack_probability": round(attack_prob, 4),
                "predicted_attack_category": cat_name,
                "attack_category_confidence": round(cat_conf, 4),
                "prediction_status": status_str,
                "firewall_action": firewall_action,
                "firewall_result": firewall_result,
            })

        # Window-level summary
        if attack_count > 0:
            window_prediction = "ATTACK"
            if probabilities is not None and probabilities.shape[1] >= 2:
                confidence = float(np.max(probabilities[predictions == 1, 1]))
            else:
                confidence = float(attack_count / len(predictions))
        else:
            window_prediction = "BENIGN"
            if probabilities is not None and probabilities.shape[1] >= 2:
                confidence = float(np.mean(probabilities[:, 0]))
            else:
                confidence = 1.0

        total_latency = time.perf_counter() - t_start

        # Update running rolling latency averages
        n = self.metrics.total_windows_processed + 1
        self.metrics.avg_inference_latency_ms = (
            (self.metrics.avg_inference_latency_ms * (n - 1) + bin_inf_latency * 1000.0) / n
        )
        self.metrics.avg_feature_extraction_latency_ms = (
            (self.metrics.avg_feature_extraction_latency_ms * (n - 1) + feat_latency * 1000.0) / n
        )
        self.metrics.avg_e2e_latency_ms = (
            (self.metrics.avg_e2e_latency_ms * (n - 1) + total_latency * 1000.0) / n
        )

        return LiveDetectionResult(
            window_id=window.window_id,
            timestamp=now_iso,
            prediction=window_prediction,
            confidence=confidence,
            flows_analyzed=len(window.flows),
            packets_analyzed=window.packets_analyzed,
            attack_count=attack_count,
            normal_count=normal_count,
            latencies={
                "queue_latency": queue_latency,
                "feature_extraction_latency": feat_latency,
                "inference_latency": bin_inf_latency,
                "total_latency": total_latency,
            },
            flow_predictions=flow_predictions,
        )

    async def stop(self) -> None:
        """Stop worker tasks and drain remaining queue jobs."""
        self._stop_event.set()
        for task in self._worker_tasks:
            if not task.done():
                task.cancel()
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        logger.info("LiveInferenceEngine stopped.")
