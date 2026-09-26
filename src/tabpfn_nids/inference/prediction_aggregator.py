"""Prediction Aggregator for TabPFN NIDS models.

Aggregates probability predictions across multiple models via equal-weighted
or custom-weighted ensemble methods and applies classification thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AggregatedPredictions:
    """Encapsulates aggregated ensemble predictions and associated probabilities."""

    predictions: np.ndarray  # Shape (N,) with integer class labels (0 or 1)
    prediction_labels: list[str]  # e.g. ["Normal", "Attack", ...]
    probabilities: np.ndarray  # Shape (N, 2) with [P(Normal), P(Attack)]
    normal_probabilities: np.ndarray  # Shape (N,)
    attack_probabilities: np.ndarray  # Shape (N,)
    per_model_probabilities: dict[str, np.ndarray] = field(default_factory=dict)
    models_used: list[str] = field(default_factory=list)
    ensemble_size: int = 1
    ensemble_method: str = "mean"
    threshold: float = 0.5
    weights: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert summary information to dictionary."""
        return {
            "total_samples": len(self.predictions),
            "normal_count": int((self.predictions == 0).sum()),
            "attack_count": int((self.predictions == 1).sum()),
            "models_used": self.models_used,
            "ensemble_size": self.ensemble_size,
            "ensemble_method": self.ensemble_method,
            "threshold": self.threshold,
            "weights": self.weights,
        }


class PredictionAggregator:
    """Aggregates probability predictions from one or more TabPFN models."""

    def __init__(
        self,
        method: str = "mean",
        threshold: float = 0.5,
        weights: dict[str, float] | None = None,
        include_per_model: bool = False,
    ) -> None:
        """Initialise aggregator.

        Args:
            method: Aggregation method ("mean" or "weighted").
            threshold: Probability threshold for classifying as Attack (default 0.5).
            weights: Optional dictionary of model weights {model_id: weight}.
            include_per_model: Whether to preserve per-model probability outputs.
        """
        self.method = method.lower()
        self.threshold = float(threshold)
        self.weights = weights
        self.include_per_model = include_per_model

        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold must be between 0.0 and 1.0, got {self.threshold}")

        if self.method not in ("mean", "weighted"):
            raise ValueError(f"Unsupported aggregation method: {self.method}. Choose 'mean' or 'weighted'.")

    def validate_weights(
        self,
        model_ids: list[str],
        weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Validate and normalise model weights.

        Args:
            model_ids: List of model IDs participating in inference.
            weights: Optional dictionary of weights.

        Returns:
            Dictionary of normalised weights summing to 1.0.
        """
        weights = weights or self.weights
        if not weights or self.method == "mean":
            # Equal weighting
            w_val = 1.0 / len(model_ids)
            return {m_id: w_val for m_id in model_ids}

        # Validate weights
        missing = [m for m in model_ids if m not in weights]
        if missing:
            raise ValueError(f"Weights missing for model(s): {missing}. Provided: {list(weights.keys())}")

        total_weight = sum(weights[m] for m in model_ids)
        if total_weight <= 0:
            raise ValueError(f"Sum of weights must be positive, got {total_weight}")

        for m in model_ids:
            if weights[m] < 0:
                raise ValueError(f"Weight for model '{m}' cannot be negative, got {weights[m]}")

        # Normalise so sum = 1.0
        return {m: weights[m] / total_weight for m in model_ids}

    def aggregate(
        self,
        model_probabilities: dict[str, np.ndarray],
        threshold: float | None = None,
        weights: dict[str, float] | None = None,
    ) -> AggregatedPredictions:
        """Aggregate prediction probabilities from one or multiple models.

        Args:
            model_probabilities: Dict mapping model_id -> np.ndarray of shape (N, 2)
                                 or (N,) where values represent P(Attack).
            threshold: Optional override for decision threshold.
            weights: Optional override for model weights.

        Returns:
            AggregatedPredictions object.
        """
        if not model_probabilities:
            raise ValueError("No model probabilities provided for aggregation.")

        model_ids = list(model_probabilities.keys())
        thresh = self.threshold if threshold is None else float(threshold)

        # Standardise all arrays to shape (N, 2)
        std_probabilities: dict[str, np.ndarray] = {}
        n_samples: int | None = None

        for m_id, proba in model_probabilities.items():
            arr = np.asarray(proba, dtype=np.float64)
            if arr.ndim == 1:
                # Given only P(Attack), derive P(Normal) = 1 - P(Attack)
                p_attack = np.clip(arr, 0.0, 1.0)
                p_normal = 1.0 - p_attack
                arr = np.column_stack([p_normal, p_attack])
            elif arr.ndim == 2:
                if arr.shape[1] == 1:
                    p_attack = np.clip(arr[:, 0], 0.0, 1.0)
                    p_normal = 1.0 - p_attack
                    arr = np.column_stack([p_normal, p_attack])
                elif arr.shape[1] >= 2:
                    arr = arr[:, :2]
                else:
                    raise ValueError(f"Model '{m_id}' probabilities have invalid shape {arr.shape}")
            else:
                raise ValueError(f"Model '{m_id}' probabilities have invalid dimension {arr.ndim}")

            if n_samples is None:
                n_samples = len(arr)
            elif len(arr) != n_samples:
                raise ValueError(
                    f"Sample count mismatch: model '{m_id}' returned {len(arr)} rows, "
                    f"expected {n_samples}"
                )

            std_probabilities[m_id] = arr

        ensemble_size = len(model_ids)

        if ensemble_size == 1:
            m_id = model_ids[0]
            agg_proba = std_probabilities[m_id]
            norm_weights = {m_id: 1.0}
            ensemble_method = "single_model"
        else:
            norm_weights = self.validate_weights(model_ids, weights)
            ensemble_method = "weighted_mean" if (self.weights or weights) and self.method == "weighted" else "mean"

            agg_proba = np.zeros((n_samples, 2), dtype=np.float64)
            for m_id, proba in std_probabilities.items():
                agg_proba += norm_weights[m_id] * proba

        # Clip probabilities to [0, 1]
        agg_proba = np.clip(agg_proba, 0.0, 1.0)
        normal_prob = agg_proba[:, 0]
        attack_prob = agg_proba[:, 1]

        # Apply threshold to derive predictions
        predictions = (attack_prob >= thresh).astype(int)
        prediction_labels = ["Attack" if p == 1 else "Normal" for p in predictions]

        per_model = std_probabilities if self.include_per_model else {}

        return AggregatedPredictions(
            predictions=predictions,
            prediction_labels=prediction_labels,
            probabilities=agg_proba,
            normal_probabilities=normal_prob,
            attack_probabilities=attack_prob,
            per_model_probabilities=per_model,
            models_used=model_ids,
            ensemble_size=ensemble_size,
            ensemble_method=ensemble_method,
            threshold=thresh,
            weights=norm_weights if ensemble_size > 1 else None,
        )
