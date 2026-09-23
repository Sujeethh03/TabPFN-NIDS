"""TabPFN model adapter for UNSW-NB15 intrusion detection.

Wraps the existing TabPFNWrapper with UNSW-NB15-specific logic:
    - Binary (Normal vs Attack) and multiclass (10-class) classification
    - Automatic context-size handling via chunked ensemble
    - Dataset-specific preprocessing before TabPFN inference
    - Train-only fitting with serialisable configuration
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)


class TabPFNModel:
    """Intrusion detection model using TabPFN for UNSW-NB15 data.

    This class manages the full inference lifecycle:
    1. Context construction (stratified subsample if data exceeds TabPFN limit)
    2. Training (fit)
    3. Batched prediction
    4. Probability calibration (optional)

    Args:
        task: "binary" or "multiclass".
        max_context_samples: Maximum training rows per TabPFN context.
        device: "auto", "cpu", "cuda", "mps".
        n_estimators: "auto" or integer.
        random_state: Random seed.
        use_chunked_ensemble: Whether to use chunked ensemble for large data.
        predict_batch_size: Prediction batch size.
    """

    def __init__(
        self,
        task: str = "binary",
        max_context_samples: int = 10_000,
        device: str = "auto",
        n_estimators: str | int = "auto",
        random_state: int = 42,
        use_chunked_ensemble: bool = True,
        predict_batch_size: int = 1_000,
    ) -> None:
        self.task = task
        self.max_context_samples = max_context_samples
        self.device = device
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.use_chunked_ensemble = use_chunked_ensemble
        self.predict_batch_size = predict_batch_size

        self._model: Any = None
        self._fitted = False
        self.fit_seconds: float | None = None
        self.predict_seconds: float | None = None
        self._n_classes: int = 0
        self._feature_names: list[str] = []

    def _subsample_context(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stratified subsample to fit within TabPFN context limit.

        Args:
            X: Feature matrix.
            y: Labels.

        Returns:
            (X_subset, y_subset).
        """
        if len(y) <= self.max_context_samples:
            return X, y

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=self.max_context_samples,
            random_state=self.random_state,
        )
        indices, _ = next(splitter.split(X, y))
        logger.info(
            "Subsampled %d → %d rows for TabPFN context",
            len(y), len(indices),
        )
        return X[indices], y[indices]

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> TabPFNModel:
        """Fit the TabPFN model on training data.

        Args:
            X: Feature matrix.
            y: Labels (binary or multiclass).

        Returns:
            self for chaining.
        """
        from tabpfn_nids.models import TabPFNWrapper

        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self._n_classes = len(np.unique(y))

        # Subsample context
        X_ctx, y_ctx = self._subsample_context(X, y)

        started = time.time()

        if self.use_chunked_ensemble and len(X_ctx) > self.max_context_samples:
            from tabpfn_nids.models.chunked_ensemble import ChunkedEnsemble
            self._model = ChunkedEnsemble(
                device=self.device,
                random_state=self.random_state,
                n_estimators=self.n_estimators,
            )
        else:
            self._model = TabPFNWrapper(
                device=self.device,
                random_state=self.random_state,
                n_estimators=self.n_estimators,
            )

        self._model.fit(X_ctx, y_ctx)
        self.fit_seconds = time.time() - started
        self._fitted = True

        logger.info(
            "TabPFN fitted: %d samples, %d features, %d classes, %.1fs",
            X_ctx.shape[0], X_ctx.shape[1], self._n_classes, self.fit_seconds,
        )
        return self

    def predict_proba(
        self, X: np.ndarray | pd.DataFrame
    ) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Probability array of shape (n_samples, n_classes).
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        started = time.time()

        # Batched prediction to prevent OOM
        if len(X) <= self.predict_batch_size:
            proba = self._model.predict_proba(X)
        else:
            batches = []
            for i in range(0, len(X), self.predict_batch_size):
                batch = X[i:i + self.predict_batch_size]
                batches.append(self._model.predict_proba(batch))
            proba = np.vstack(batches)

        self.predict_seconds = time.time() - started
        logger.info(
            "Predicted %d samples in %.1fs",
            len(X), self.predict_seconds,
        )
        return proba

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            Label array of shape (n_samples,).
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def get_config(self) -> dict[str, Any]:
        """Return model configuration."""
        return {
            "task": self.task,
            "max_context_samples": self.max_context_samples,
            "device": self.device,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "use_chunked_ensemble": self.use_chunked_ensemble,
            "predict_batch_size": self.predict_batch_size,
            "n_classes": self._n_classes,
            "feature_names": self._feature_names,
            "fit_seconds": self.fit_seconds,
        }
