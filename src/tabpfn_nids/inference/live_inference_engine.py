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
from tabpfn_nids.features.feature_pipeline import compute_all_features
from tabpfn_nids.flows.flow_builder import FlowRecord
from tabpfn_nids.inference import DynamicInferenceManager, ModelInfo, ModelRegistry

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACTS_DIR / "models" / "tabpfn_binary_model.pkl"
DEFAULT_SCHEMA_PATH = DEFAULT_ARTIFACTS_DIR / "models" / "model_feature_schema.json"
DEFAULT_SCALER_PATH = DEFAULT_ARTIFACTS_DIR / "preprocessing" / "fitted_scaler.pkl"


@dataclass
class LiveDetectionResult:
    """Detection result for a single sliding traffic window."""

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
) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract features from flow records and preprocess using the fitted scaler.

    Ensures 100% schema alignment with the offline PCAP pipeline.
    """
    if not flows:
        return pd.DataFrame(), np.empty((0, len(feature_names)))

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

    # Apply existing fitted scaler
    X_scaled = scaler.transform(X)

    metadata_cols = [c for c in ["flow_id", "src_ip", "dst_ip", "src_port", "dst_port", "protocol_name"] if c in engineered_df.columns]
    metadata_df = engineered_df[metadata_cols].copy()

    return metadata_df, X_scaled


class LiveInferenceEngine:
    """Producer-consumer inference engine that processes detection windows concurrently."""

    def __init__(
        self,
        window_queue: asyncio.Queue[DetectionWindow],
        max_workers: int = 2,
        model_path: Path | str | None = None,
        schema_path: Path | str | None = None,
        scaler_path: Path | str | None = None,
        result_callback: Callable[[LiveDetectionResult], Any] | None = None,
    ) -> None:
        self.window_queue = window_queue
        self.max_workers = max(1, int(max_workers))
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.schema_path = Path(schema_path) if schema_path else DEFAULT_SCHEMA_PATH
        self.scaler_path = Path(scaler_path) if scaler_path else DEFAULT_SCALER_PATH
        self.result_callback = result_callback

        self.metrics = LiveInferenceMetrics()
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._stop_event = asyncio.Event()

        # Shared preloaded artifacts
        self.model: Any = None
        self.scaler: Any = None
        self.feature_names: list[str] = []
        self.inference_manager: DynamicInferenceManager | None = None
        self._artifacts_loaded = False

    def load_artifacts(self) -> None:
        """Load trained TabPFN model, schema, and scaler into memory once."""
        if self._artifacts_loaded:
            return

        logger.info("Loading model artifacts for live inference...")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Feature schema not found: {self.schema_path}")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")

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

        self._artifacts_loaded = True
        logger.info(
            "Live inference artifacts loaded successfully (%d features)",
            len(self.feature_names),
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
                # Wait for next window job
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
        """Synchronous feature extraction and model inference executed in thread pool."""
        t_start = time.perf_counter()

        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # 0 flows case (quiet network)
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
            )

        # 1. Feature extraction & scaling
        t_feat_start = time.perf_counter()
        metadata_df, X_scaled = extract_and_preprocess_window(
            window.flows, self.feature_names, self.scaler
        )
        feat_latency = time.perf_counter() - t_feat_start

        # 2. Model inference
        t_inf_start = time.perf_counter()
        assert self.inference_manager is not None
        t_inf_start = time.perf_counter()
        inf_result = self.inference_manager.predict(X_scaled)
        inf_latency = time.perf_counter() - t_inf_start
        predictions = inf_result.predictions
        probabilities = inf_result.probabilities

        logger.info(
            "LIVE MODEL DEBUG: flows=%d predictions=%s",
            len(predictions),
            predictions.tolist(),
        )

        if probabilities is not None:
            logger.info(
                "LIVE MODEL DEBUG: probabilities=%s",
                probabilities.tolist(),
            )

        logger.info(
            "LIVE MODEL DEBUG: scaled feature range min=%.4f max=%.4f mean=%.4f",
            float(np.min(X_scaled)),
            float(np.max(X_scaled)),
            float(np.mean(X_scaled)),
        )

        attack_mask = predictions == 1
        attack_count = int(attack_mask.sum())
        normal_count = int((predictions == 0).sum())

        # Determine window prediction & confidence
        if attack_count > 0:
            window_prediction = "ATTACK"
            if probabilities is not None and probabilities.shape[1] >= 2:
                # Highest attack probability among detected attack flows
                confidence = float(np.max(probabilities[attack_mask, 1]))
            else:
                confidence = float(attack_count / len(predictions))
        else:
            window_prediction = "BENIGN"
            if probabilities is not None and probabilities.shape[1] >= 2:
                # Average benign probability
                confidence = float(np.mean(probabilities[:, 0]))
            else:
                confidence = 1.0

        total_latency = time.perf_counter() - t_start

        # Update running rolling latency averages
        n = self.metrics.total_windows_processed + 1
        self.metrics.avg_inference_latency_ms = (
            (self.metrics.avg_inference_latency_ms * (n - 1) + inf_latency * 1000.0) / n
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
                "inference_latency": inf_latency,
                "total_latency": total_latency,
            },
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
