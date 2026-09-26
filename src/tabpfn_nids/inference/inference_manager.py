"""Dynamic Parallel Inference Manager for TabPFN NIDS.

Provides dynamic, parallel inference across large datasets (data parallelism)
and true model ensembling across multiple independently trained TabPFN models.
"""

from __future__ import annotations

import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tabpfn_nids.inference.model_registry import (
    IncompatibleModelError,
    ModelInfo,
    ModelRegistry,
)
from tabpfn_nids.inference.model_worker import (
    ChunkResult,
    InferenceChunk,
    ModelWorker,
)
from tabpfn_nids.inference.prediction_aggregator import (
    AggregatedPredictions,
    PredictionAggregator,
)

logger = logging.getLogger(__name__)


def calculate_worker_count(
    num_rows: int,
    max_rows_per_worker: int = 10_000,
    max_workers: int | None = None,
) -> int:
    """Calculate the number of inference workers based on row count and resource bounds.

    Rules:
        Rows <= 10,000      -> 1 worker
        10,001–20,000       -> 2 workers
        20,001–30,000       -> 3 workers
        30,001–40,000       -> 4 workers
        ...
        num_workers = ceil(num_rows / max_rows_per_worker)

    Borders and caps:
        - Must be at least 1.
        - If max_workers is specified, worker count is clamped to max_workers.
        - If max_workers is None, a sensible upper bound based on CPU cores is applied.
    """
    if num_rows <= 0:
        return 1

    if max_rows_per_worker <= 0:
        raise ValueError(f"max_rows_per_worker must be positive, got {max_rows_per_worker}")

    required_workers = math.ceil(num_rows / max_rows_per_worker)

    if max_workers is not None:
        required_workers = min(required_workers, max_workers)
    else:
        # Sensible system ceiling based on available CPU cores
        cpu_cores = os.cpu_count() or 4
        # Allow at most max(4, cpu_cores) workers unless explicitly configured
        required_workers = min(required_workers, max(4, cpu_cores))

    return max(1, required_workers)


@dataclass
class InferenceResult:
    """Result returned by DynamicInferenceManager."""

    predictions: np.ndarray
    prediction_labels: list[str]
    probabilities: np.ndarray
    normal_probabilities: np.ndarray
    attack_probabilities: np.ndarray
    per_model_probabilities: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return a structured summary of the inference execution and findings."""
        return {
            "total_samples": len(self.predictions),
            "normal_count": int((self.predictions == 0).sum()),
            "attack_count": int((self.predictions == 1).sum()),
            "inference_mode": self.metadata.get("inference_mode", "Unknown"),
            "models_used": self.metadata.get("models_used", []),
            "num_workers": self.metadata.get("num_workers", 1),
            "total_inference_seconds": self.metadata.get("total_inference_seconds", 0.0),
        }


class DynamicInferenceManager:
    """Manages parallel inference across data chunks and true model ensembling."""

    def __init__(
        self,
        max_rows_per_worker: int = 10_000,
        max_workers: int | None = None,
        executor_type: str = "thread",
        probability_aggregation: str = "mean",
        prediction_threshold: float = 0.5,
        enable_ensemble: bool = True,
        include_per_model_probabilities: bool = False,
        weights: dict[str, float] | None = None,
        model_registry: ModelRegistry | None = None,
    ) -> None:
        self.max_rows_per_worker = max_rows_per_worker
        self.max_workers = max_workers
        self.executor_type = executor_type.lower()
        self.probability_aggregation = probability_aggregation
        self.prediction_threshold = float(prediction_threshold)
        self.enable_ensemble = enable_ensemble
        self.include_per_model_probabilities = include_per_model_probabilities
        self.weights = weights
        self.registry = model_registry or ModelRegistry()

        self.aggregator = PredictionAggregator(
            method=self.probability_aggregation,
            threshold=self.prediction_threshold,
            weights=self.weights,
            include_per_model=self.include_per_model_probabilities,
        )

    def create_chunks(self, num_rows: int) -> list[InferenceChunk]:
        """Split num_rows into contiguous disjoint chunk slices."""
        if num_rows <= 0:
            return []

        chunks: list[InferenceChunk] = []
        chunk_id = 0
        for start_idx in range(0, num_rows, self.max_rows_per_worker):
            end_idx = min(start_idx + self.max_rows_per_worker, num_rows)
            chunks.append(
                InferenceChunk(
                    chunk_id=chunk_id,
                    start_idx=start_idx,
                    end_idx=end_idx,
                )
            )
            chunk_id += 1
        return chunks

    def determine_inference_mode(
        self,
        num_models: int,
        num_workers: int,
        num_chunks: int,
    ) -> str:
        """Return human-readable inference mode label."""
        if num_models > 1:
            if num_chunks > 1 and num_workers > 1:
                return "Parallel Ensemble"
            return "Ensemble"
        else:
            if num_chunks > 1 and num_workers > 1:
                return "Parallel"
            return "Single model"

    def predict(
        self,
        X: np.ndarray | pd.DataFrame,
        models: dict[str, Any] | list[Any] | None = None,
        model_ids: list[str] | None = None,
    ) -> InferenceResult:
        """Run parallel or ensemble inference on preprocessed feature data.

        Args:
            X: Preprocessed feature matrix (scaled and cleaned).
            models: Optional dict or list of pre-loaded model instances.
            model_ids: Optional list of model IDs to load from the registry.

        Returns:
            InferenceResult containing predictions, probabilities, and execution metadata.
        """
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        num_rows = len(X_arr)
        if num_rows == 0:
            raise ValueError("Input feature matrix X contains 0 rows.")

        inference_start = time.perf_counter()

        # -------------------------------------------------------------
        # 1. Resolve and validate models
        # -------------------------------------------------------------
        active_models: dict[str, Any] = {}

        if models is not None:
            if isinstance(models, dict):
                active_models = dict(models)
            elif isinstance(models, list):
                active_models = {f"model_{i + 1}": m for i, m in enumerate(models)}
            else:
                active_models = {"model_1": models}
        elif model_ids is not None:
            for m_id in model_ids:
                active_models[m_id] = self.registry.get_model(m_id)
        else:
            # Discover from registry
            discovered = self.registry.discover_models()
            if not discovered:
                raise ValueError("No compatible TabPFN model available in registry.")

            if len(discovered) > 1 and self.enable_ensemble:
                # Multi-model mode: validate compatibility before loading
                self.registry.validate_compatibility(discovered)
                active_models = self.registry.load_all([m.model_id for m in discovered])
            else:
                # Single-model mode: use first available model
                primary = discovered[0]
                active_models = {primary.model_id: self.registry.get_model(primary.model_id)}

        num_models = len(active_models)
        if num_models == 0:
            raise ValueError("No compatible TabPFN model available.")

        # -------------------------------------------------------------
        # 2. Compute chunks and workers
        # -------------------------------------------------------------
        chunks = self.create_chunks(num_rows)
        num_chunks = len(chunks)
        worker_count = calculate_worker_count(
            num_rows=num_rows,
            max_rows_per_worker=self.max_rows_per_worker,
            max_workers=self.max_workers,
        )

        mode_name = self.determine_inference_mode(
            num_models=num_models,
            num_workers=worker_count,
            num_chunks=num_chunks,
        )

        # -------------------------------------------------------------
        # 3. Structured Logging
        # -------------------------------------------------------------
        logger.info("Input flows: %d", num_rows)
        logger.info("Available models: %d (%s)", num_models, list(active_models.keys()))
        logger.info("Max rows per worker: %d", self.max_rows_per_worker)
        logger.info("Inference chunks: %d", num_chunks)
        logger.info("Selected workers: %d", worker_count)
        logger.info("Starting %s inference", mode_name.lower())

        # -------------------------------------------------------------
        # 4. Execute inference across chunks
        # -------------------------------------------------------------
        chunk_results: list[ChunkResult] = []

        if num_chunks == 1 or worker_count == 1 or self.executor_type == "sequential":
            # Sequential single-worker execution (optimal for small data or 1 worker)
            worker = ModelWorker(worker_id=0)
            for chunk in chunks:
                X_chunk = X_arr[chunk.start_idx : chunk.end_idx]
                result = worker.run_chunk(X_chunk, chunk, active_models)
                chunk_results.append(result)
        else:
            # Parallel execution across chunks using ThreadPoolExecutor
            # Threads reuse the preloaded in-memory model instances safely.
            max_th = min(worker_count, num_chunks)
            with ThreadPoolExecutor(max_workers=max_th) as executor:
                futures = {}
                for w_idx, chunk in enumerate(chunks):
                    worker = ModelWorker(worker_id=w_idx % max_th)
                    X_chunk = X_arr[chunk.start_idx : chunk.end_idx]
                    future = executor.submit(
                        worker.run_chunk,
                        X_chunk,
                        chunk,
                        active_models,
                    )
                    futures[future] = chunk.chunk_id

                for future in as_completed(futures):
                    cid = futures[future]
                    try:
                        c_res = future.result()
                        chunk_results.append(c_res)
                    except Exception as exc:
                        logger.error("Inference execution failed on chunk %d: %s", cid, exc)
                        raise

            # Sort chunk results by chunk_id to maintain exact original row ordering
            chunk_results.sort(key=lambda r: r.chunk_id)

        # -------------------------------------------------------------
        # 5. Assemble full per-model probability matrices
        # -------------------------------------------------------------
        logger.info("Aggregating predictions")

        full_model_probabilities: dict[str, np.ndarray] = {}
        for m_id in active_models.keys():
            model_chunk_arrays = [
                res.model_probabilities[m_id] for res in chunk_results
            ]
            full_model_probabilities[m_id] = np.vstack(model_chunk_arrays)

        # -------------------------------------------------------------
        # 6. Aggregate across models
        # -------------------------------------------------------------
        agg_result: AggregatedPredictions = self.aggregator.aggregate(
            model_probabilities=full_model_probabilities,
            threshold=self.prediction_threshold,
            weights=self.weights,
        )

        total_inference_seconds = time.perf_counter() - inference_start
        rows_per_second = (
            num_rows / total_inference_seconds if total_inference_seconds > 0 else 0.0
        )

        # Calculate average model inference time across chunks
        total_chunk_worker_time = sum(res.inference_seconds for res in chunk_results)
        avg_model_inference_seconds = (
            total_chunk_worker_time / (num_chunks * num_models)
            if num_chunks > 0 and num_models > 0
            else 0.0
        )

        logger.info(
            "Final inference completed in %.2f seconds (%.1f rows/s)",
            total_inference_seconds,
            rows_per_second,
        )

        metadata = {
            "inference_mode": mode_name,
            "num_flows": num_rows,
            "num_models": num_models,
            "models_used": list(active_models.keys()),
            "num_workers": worker_count,
            "max_rows_per_worker": self.max_rows_per_worker,
            "num_chunks": num_chunks,
            "rows_per_chunk": [c.row_count for c in chunks],
            "ensemble_method": (
                "Weighted mean"
                if agg_result.ensemble_method == "weighted_mean"
                else "Mean probability"
                if num_models > 1
                else "Single model"
            ),
            "prediction_threshold": self.prediction_threshold,
            "total_inference_seconds": round(total_inference_seconds, 4),
            "avg_model_inference_seconds": round(avg_model_inference_seconds, 4),
            "rows_per_second": round(rows_per_second, 1),
            "weights": agg_result.weights,
        }

        return InferenceResult(
            predictions=agg_result.predictions,
            prediction_labels=agg_result.prediction_labels,
            probabilities=agg_result.probabilities,
            normal_probabilities=agg_result.normal_probabilities,
            attack_probabilities=agg_result.attack_probabilities,
            per_model_probabilities=agg_result.per_model_probabilities,
            metadata=metadata,
        )
