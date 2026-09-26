"""Worker module for chunked TabPFN inference.

Encapsulates chunk data structures and execution logic across threads or processes.
Reuses loaded models to avoid disk reload overhead.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InferenceChunk:
    """Represents a row slice of the feature matrix assigned to an inference worker."""

    chunk_id: int
    start_idx: int
    end_idx: int

    @property
    def row_count(self) -> int:
        return self.end_idx - self.start_idx


@dataclass
class ChunkResult:
    """Prediction results produced by inference workers for a single chunk."""

    chunk_id: int
    start_idx: int
    end_idx: int
    row_count: int
    model_probabilities: dict[str, np.ndarray]  # {model_id: (chunk_rows, 2)}
    inference_seconds: float
    worker_id: str | int = 0


class ModelWorker:
    """Executes model inference on a single chunk of preprocessed data."""

    def __init__(self, worker_id: str | int = 0) -> None:
        self.worker_id = worker_id

    def run_chunk(
        self,
        X_chunk: np.ndarray,
        chunk: InferenceChunk,
        models: dict[str, Any],
    ) -> ChunkResult:
        """Run inference on the chunk for all specified models.

        Args:
            X_chunk: 2D numpy array view representing the chunk's feature rows.
            chunk: Metadata for the chunk (indices, id).
            models: Dictionary mapping model_id -> loaded model instance.

        Returns:
            ChunkResult containing per-model probabilities and timings.
        """
        logger.debug(
            "Worker %s starting chunk %d (rows %d to %d, total %d)",
            self.worker_id,
            chunk.chunk_id,
            chunk.start_idx,
            chunk.end_idx,
            chunk.row_count,
        )

        chunk_start = time.perf_counter()
        probabilities: dict[str, np.ndarray] = {}

        for model_id, model in models.items():
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_chunk)
                else:
                    preds = model.predict(X_chunk)
                    preds = np.asarray(preds)
                    # Convert hard predictions to pseudo-probabilities if predict_proba is absent
                    p_attack = (preds == 1).astype(float)
                    p_normal = 1.0 - p_attack
                    proba = np.column_stack([p_normal, p_attack])

                proba = np.asarray(proba)
                if proba.ndim == 1:
                    p_attack = np.clip(proba, 0.0, 1.0)
                    p_normal = 1.0 - p_attack
                    proba = np.column_stack([p_normal, p_attack])

                probabilities[model_id] = proba

            except Exception as exc:
                logger.error(
                    "Worker %s failed on chunk %d with model '%s': %s",
                    self.worker_id,
                    chunk.chunk_id,
                    model_id,
                    exc,
                )
                raise RuntimeError(
                    f"Worker {self.worker_id} failed on chunk {chunk.chunk_id} "
                    f"(rows {chunk.start_idx}:{chunk.end_idx}) using model '{model_id}': {exc}"
                ) from exc

        chunk_seconds = time.perf_counter() - chunk_start
        logger.info(
            "Chunk %d completed (%d rows) in %.2fs",
            chunk.chunk_id,
            chunk.row_count,
            chunk_seconds,
        )

        return ChunkResult(
            chunk_id=chunk.chunk_id,
            start_idx=chunk.start_idx,
            end_idx=chunk.end_idx,
            row_count=chunk.row_count,
            model_probabilities=probabilities,
            inference_seconds=chunk_seconds,
            worker_id=self.worker_id,
        )
