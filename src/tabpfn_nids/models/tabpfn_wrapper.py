"""A thin wrapper over TabPFNClassifier.

The wrapper exists to enforce three things the raw estimator leaves to the
caller, each of which has bitten this project already:

1. **The context limit is checked up front.** TabPFN v2 was pretrained for at
   most 10,000 in-context training samples. Passing more is permitted by the
   library via ``ignore_pretraining_limits`` but degrades accuracy silently.
   This wrapper raises instead, and names the chunked ensemble as the fix.
2. **The checkpoint is pinned.** ``model_path="auto"`` resolves to a gated
   HuggingFace repository holding a *different, newer* model than the one
   published in Nature 2025. See ``config.TABPFN_CHECKPOINT``.
3. **The backend is chosen explicitly and recorded.** MPS measured roughly
   3.7x faster than CPU at 1,000 context rows and 5.3x at 10,000, and the
   library refuses CPU inference above 1,000 samples by default.

Timing for fit and predict is logged and retained on the instance so the
experiment runners can write it into the results CSV without re-measuring.
"""

from __future__ import annotations

import gc
import logging
import time
from typing import Any

import numpy as np

from tabpfn_nids import config

logger = logging.getLogger(__name__)


class TabPFNWrapper:
    """TabPFNClassifier with an enforced context limit and recorded timings.

    Attributes:
        context_limit: Maximum permitted training rows.
        device: The backend actually in use, "mps" or "cpu".
        fit_seconds: Wall-clock seconds spent in the last ``fit``.
        predict_seconds: Wall-clock seconds spent in the last predict call.
        n_estimators_: The ensemble size TabPFN resolved to, after fitting.

    Example:
        >>> model = TabPFNWrapper(random_state=42)
        >>> model.fit(X_train, y_train)
        >>> proba = model.predict_proba(X_test)
    """

    def __init__(
        self,
        device: str = "auto",
        random_state: int = config.SEED,
        context_limit: int = config.MAX_CONTEXT_SAMPLES,
        n_estimators: int | str = "auto",
        model_path: str | None = None,
        predict_batch_size: int | None = None,
        **tabpfn_kwargs: Any,
    ) -> None:
        """Initialise the wrapper.

        Args:
            device: "auto" to prefer MPS where available, or "mps" / "cpu".
            random_state: Seed passed to TabPFN (RULE 5).
            context_limit: Maximum training rows accepted by ``fit``.
            n_estimators: TabPFN ensemble size. "auto" lets the library scale
                it for feature coverage, which is the accuracy-optimal default
                but grows runtime roughly linearly. Fix it to an integer to
                make runs comparable and to bound cost.
            model_path: Checkpoint filename. Defaults to the pinned TabPFN v2
                classifier; override only with a deliberate reason.
            predict_batch_size: Rows per prediction batch. Required on MPS for
                large test sets, which otherwise raise
                TabPFNMPSOutOfMemoryError. None predicts in one pass.
            **tabpfn_kwargs: Passed through to TabPFNClassifier.
        """
        self.context_limit = context_limit
        self.device = config.resolve_device(device)
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.model_path = model_path or config.TABPFN_CHECKPOINT
        self.predict_batch_size = predict_batch_size
        self._tabpfn_kwargs = tabpfn_kwargs

        self._model: Any = None
        self.fit_seconds: float | None = None
        self.predict_seconds: float | None = None
        self.n_estimators_: int | None = None
        self.n_train_: int | None = None
        self.n_features_: int | None = None

    # -- validation --------------------------------------------------------

    def _check_context_size(self, X: np.ndarray) -> None:
        """Reject a training set larger than the pretraining context limit.

        Args:
            X: The training feature matrix.

        Raises:
            ValueError: If X has more rows than ``context_limit``.
        """
        n_rows = X.shape[0]
        if n_rows > self.context_limit:
            raise ValueError(
                f"TabPFN context limit exceeded: {n_rows:,} training rows were "
                f"given but the limit is {self.context_limit:,}.\n"
                "\n"
                "TabPFN v2 is pretrained for at most 10,000 in-context training "
                "samples. Passing more is possible with "
                "ignore_pretraining_limits=True, but accuracy degrades and the "
                "result is no longer a faithful use of the model.\n"
                "\n"
                "Use the chunked ensemble instead: split the training set into "
                "stratified chunks of at most "
                f"{self.context_limit:,} rows, run TabPFN on each, and combine "
                "the per-chunk predictions by weighted voting. See "
                "tabpfn_nids.models.chunked_ensemble (Enhancement 1).\n"
                "\n"
                "To subsample to a single context instead, use "
                "sklearn.model_selection.StratifiedShuffleSplit with "
                f"train_size={self.context_limit}."
            )

    def _check_feature_count(self, X: np.ndarray) -> None:
        """Warn if the feature count exceeds TabPFN's pretraining range.

        Args:
            X: The feature matrix.
        """
        if X.shape[1] > config.MAX_FEATURES:
            logger.warning(
                "%d features exceeds TabPFN's pretraining limit of %d; "
                "accuracy may degrade.",
                X.shape[1],
                config.MAX_FEATURES,
            )

    def _build(self) -> Any:
        """Construct the underlying TabPFNClassifier.

        Returns:
            A configured, unfitted TabPFNClassifier.
        """
        from tabpfn import TabPFNClassifier

        # The library refuses CPU inference above MAX_CPU_SAMPLES rows unless
        # limits are ignored. Our own context check has already run by this
        # point, so relaxing the library's guard cannot let an oversized
        # context through.
        ignore_limits = (
            self.device == "cpu" and (self.n_train_ or 0) > config.MAX_CPU_SAMPLES
        )
        if ignore_limits:
            logger.warning(
                "Running %d rows on CPU; this is far slower than MPS.",
                self.n_train_,
            )

        return TabPFNClassifier(
            model_path=self.model_path,
            device=self.device,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            ignore_pretraining_limits=ignore_limits,
            **self._tabpfn_kwargs,
        )

    # -- public API --------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> TabPFNWrapper:
        """Load the training set into TabPFN's context.

        TabPFN does not train: ``fit`` caches the in-context examples and is
        near-instant. Essentially all compute happens in ``predict``.

        Args:
            X_train: Training features, at most ``context_limit`` rows.
            y_train: Training labels.

        Returns:
            self, to allow chaining.

        Raises:
            ValueError: If the context limit is exceeded, or if X and y
                disagree on row count.
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"X_train has {X_train.shape[0]} rows but y_train has "
                f"{y_train.shape[0]}."
            )

        self._check_context_size(X_train)
        self._check_feature_count(X_train)
        self.n_train_, self.n_features_ = X_train.shape

        self._model = self._build()

        logger.info(
            "Fitting TabPFN on %d rows x %d features (device=%s, checkpoint=%s)",
            self.n_train_,
            self.n_features_,
            self.device,
            self.model_path,
        )
        started = time.time()
        try:
            self._model.fit(X_train, y_train)
        except Exception as exc:
            raise RuntimeError(
                f"TabPFN fit failed on device '{self.device}' "
                f"({type(exc).__name__}: {exc}). If this is an MPS backend "
                "problem, retry with device='cpu'."
            ) from exc
        self.fit_seconds = time.time() - started

        self.n_estimators_ = getattr(self._model, "n_estimators_", None)
        logger.info(
            "Fit completed in %.2fs (n_estimators=%s)",
            self.fit_seconds,
            self.n_estimators_,
        )
        return self

    def _require_fitted(self) -> Any:
        """Return the underlying model, raising if it is not fitted.

        Returns:
            The fitted TabPFNClassifier.

        Raises:
            RuntimeError: If ``fit`` has not been called.
        """
        if self._model is None:
            raise RuntimeError("TabPFNWrapper.fit() must be called first.")
        return self._model

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X_test: Test features.

        Returns:
            Predicted labels, one per test row.
        """
        model = self._require_fitted()
        logger.info("Predicting on %d rows...", len(X_test))
        started = time.time()
        predictions = model.predict(np.asarray(X_test))
        self.predict_seconds = time.time() - started
        logger.info(
            "Predict completed in %.2fs (%.1f rows/s)",
            self.predict_seconds,
            len(X_test) / max(self.predict_seconds, 1e-9),
        )
        return predictions

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        When ``predict_batch_size`` is set, the test set is processed in
        batches. This is required on Apple Silicon: MPS raises
        TabPFNMPSOutOfMemoryError once the test set is large enough, and the
        failure is cumulative across repeated model instances, so a chunked
        ensemble hits it partway through even when a single run succeeds.

        Args:
            X_test: Test features.

        Returns:
            An array of shape ``(n_rows, n_classes)``.
        """
        model = self._require_fitted()
        X_test = np.asarray(X_test)
        logger.info("Predicting probabilities on %d rows...", len(X_test))
        started = time.time()

        if not self.predict_batch_size or len(X_test) <= self.predict_batch_size:
            proba = model.predict_proba(X_test)
        else:
            batches = [
                model.predict_proba(X_test[start : start + self.predict_batch_size])
                for start in range(0, len(X_test), self.predict_batch_size)
            ]
            proba = np.vstack(batches)

        self.predict_seconds = time.time() - started
        logger.info("predict_proba completed in %.2fs", self.predict_seconds)
        return proba

    def free(self) -> None:
        """Release the model and any cached device memory.

        MPS does not return memory to the system when a Python reference is
        dropped, so a long-running ensemble accumulates allocations until it
        runs out. Callers that create many wrappers in sequence should call
        this between them.
        """
        self._model = None
        gc.collect()
        try:
            import torch

            if self.device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, AttributeError):  # pragma: no cover - env specific
            pass

    def describe(self) -> dict[str, Any]:
        """Return the run configuration for the results CSV.

        Returns:
            A dict of the settings and timings for this fit/predict cycle.
        """
        return {
            "device": self.device,
            "checkpoint": self.model_path,
            "n_estimators": self.n_estimators_,
            "context_rows": self.n_train_,
            "n_features": self.n_features_,
            "fit_seconds": self.fit_seconds,
            "predict_seconds": self.predict_seconds,
        }
