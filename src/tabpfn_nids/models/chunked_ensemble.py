"""Chunked TabPFN ensemble (Enhancement 1).

TabPFN v2 caps its in-context training set at 10,000 samples, which is two
orders of magnitude smaller than a real NIDS capture. This module lifts that
ceiling by partitioning the training data into stratified chunks, running
TabPFN independently on each chunk against the same test set, and combining
the per-chunk probability estimates.

Two aggregation strategies are provided:

``majority``
    A plain unweighted mean of the per-chunk probability matrices. Every
    chunk contributes equally.

``weighted_vote``
    A mean weighted by each chunk's mean prediction confidence, where
    confidence is ``mean(max(P(class), axis=1))``. A chunk whose contexts
    leave it unsure across the test set is given less say than one that is
    consistently decisive.

A note on cost, because it drives every design choice here: TabPFN's ``fit``
merely caches the context and takes well under a second, while ``predict``
runs the transformer. Total runtime is therefore
``n_chunks x predict(chunk_size, n_test)`` and grows linearly in both the
number of chunks and the size of the test set.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from tabpfn_nids import config
from tabpfn_nids.models.chunker import describe_chunks, stratified_chunk
from tabpfn_nids.models.tabpfn_wrapper import TabPFNWrapper

logger = logging.getLogger(__name__)

AGGREGATION_STRATEGIES = ("weighted_vote", "majority")


class ChunkedTabPFNEnsemble:
    """Run TabPFN across stratified chunks and aggregate the probabilities.

    Attributes:
        chunk_size: Maximum training rows per chunk.
        aggregation: "weighted_vote" or "majority".
        chunks_: The fitted chunks, available after ``fit``.
        chunk_weights_: Per-chunk weights from the last predict call.
        fit_seconds: Seconds spent chunking in the last ``fit``.
        predict_seconds: Seconds spent in the last predict call.

    Example:
        >>> ensemble = ChunkedTabPFNEnsemble(chunk_size=10_000)
        >>> ensemble.fit(X_train, y_train)
        >>> proba = ensemble.predict_proba(X_test)
    """

    def __init__(
        self,
        chunk_size: int = config.MAX_CONTEXT_SAMPLES,
        aggregation: str = "weighted_vote",
        random_state: int = config.SEED,
        device: str = "auto",
        n_estimators: int | str = "auto",
        max_chunks: int | None = None,
        predict_batch_size: int | None = 1_000,
        show_progress: bool = True,
    ) -> None:
        """Initialise the ensemble.

        Args:
            chunk_size: Maximum rows per chunk; must not exceed TabPFN's
                context limit.
            aggregation: Aggregation strategy, one of AGGREGATION_STRATEGIES.
            random_state: Seed for chunking and for each TabPFN instance.
            device: Torch backend; "auto" prefers MPS.
            n_estimators: TabPFN ensemble size per chunk. Must be held equal
                to the baseline's value, or a comparison between the two
                measures ensemble size rather than chunking.
            max_chunks: Optional cap on chunk count, to bound runtime.
            predict_batch_size: Test rows per prediction batch. Defaults to
                1,000: on MPS the allocator does not release memory between
                chunks, and predicting 5,000 rows in one pass exhausts a 16 GB
                M1 by the fourth chunk. None disables batching.
            show_progress: Whether to display a tqdm progress bar.

        Raises:
            ValueError: If the aggregation strategy is unknown or chunk_size
                exceeds TabPFN's context limit.
        """
        if aggregation not in AGGREGATION_STRATEGIES:
            raise ValueError(
                f"unknown aggregation {aggregation!r}; expected one of "
                f"{AGGREGATION_STRATEGIES}"
            )
        if chunk_size > config.MAX_CONTEXT_SAMPLES:
            raise ValueError(
                f"chunk_size {chunk_size:,} exceeds TabPFN's context limit of "
                f"{config.MAX_CONTEXT_SAMPLES:,}; each chunk must fit in one "
                "context."
            )

        self.chunk_size = chunk_size
        self.aggregation = aggregation
        self.random_state = random_state
        self.device = device
        self.n_estimators = n_estimators
        self.max_chunks = max_chunks
        self.predict_batch_size = predict_batch_size
        self.show_progress = show_progress

        self.chunks_: list[tuple[np.ndarray, np.ndarray]] | None = None
        self.classes_: np.ndarray | None = None
        self.chunk_weights_: np.ndarray | None = None
        self.chunk_confidences_: list[float] = []
        self.fit_seconds: float | None = None
        self.predict_seconds: float | None = None

    # -- fit ---------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> ChunkedTabPFNEnsemble:
        """Partition the training data into stratified chunks.

        No model is trained here: TabPFN is a prior-fitted network, so the
        chunks are simply stored and become in-context examples at predict
        time.

        Args:
            X_train: Full training features; may exceed the context limit.
            y_train: Full training labels.

        Returns:
            self, to allow chaining.
        """
        started = time.time()
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        self.chunks_ = stratified_chunk(
            X_train,
            y_train,
            chunk_size=self.chunk_size,
            random_state=self.random_state,
            max_chunks=self.max_chunks,
        )
        self.classes_ = np.unique(y_train)
        self.fit_seconds = time.time() - started

        logger.info(
            "Ensemble fitted: %d chunks over %d rows in %.2fs",
            len(self.chunks_),
            len(y_train),
            self.fit_seconds,
        )
        return self

    def _require_fitted(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return the stored chunks, raising if unfitted.

        Returns:
            The chunk list.

        Raises:
            RuntimeError: If ``fit`` has not been called.
        """
        if self.chunks_ is None:
            raise RuntimeError("ChunkedTabPFNEnsemble.fit() must be called first.")
        return self.chunks_

    # -- aggregation -------------------------------------------------------

    @staticmethod
    def _confidence(proba: np.ndarray) -> float:
        """Mean decisiveness of a probability matrix.

        Args:
            proba: Array of shape ``(n_rows, n_classes)``.

        Returns:
            The mean of the per-row maximum probability. For binary problems
            this lies in [0.5, 1.0]: 0.5 is a coin flip, 1.0 is certainty.
        """
        return float(np.mean(np.max(proba, axis=1)))

    def _aggregate(self, per_chunk: list[np.ndarray]) -> np.ndarray:
        """Combine per-chunk probability matrices into one.

        Args:
            per_chunk: One ``(n_rows, n_classes)`` array per chunk.

        Returns:
            The aggregated probability matrix, rows summing to 1.
        """
        stacked = np.stack(per_chunk, axis=0)

        if self.aggregation == "majority":
            weights = np.ones(len(per_chunk), dtype=np.float64)
        else:
            weights = np.array(self.chunk_confidences_, dtype=np.float64)
            # Degenerate confidences would make the weighted mean undefined;
            # fall back to equal weighting rather than producing NaNs.
            if not np.isfinite(weights).all() or weights.sum() <= 0:
                logger.warning(
                    "Chunk confidences are degenerate; falling back to "
                    "equal weights."
                )
                weights = np.ones(len(per_chunk), dtype=np.float64)

        weights = weights / weights.sum()
        self.chunk_weights_ = weights

        aggregated = np.tensordot(weights, stacked, axes=(0, 0))
        # Guard against drift from floating-point accumulation.
        return aggregated / aggregated.sum(axis=1, keepdims=True)

    # -- predict -----------------------------------------------------------

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Run TabPFN on every chunk and aggregate the probabilities.

        Args:
            X_test: Test features.

        Returns:
            Aggregated probabilities of shape ``(n_test, n_classes)``.
        """
        chunks = self._require_fitted()
        X_test = np.asarray(X_test)

        logger.info(
            "Predicting %d test rows across %d chunks (aggregation=%s)",
            len(X_test),
            len(chunks),
            self.aggregation,
        )
        started = time.time()
        per_chunk: list[np.ndarray] = []
        self.chunk_confidences_ = []

        iterator = tqdm(
            enumerate(chunks, start=1),
            total=len(chunks),
            desc="TabPFN chunks",
            unit="chunk",
            disable=not self.show_progress,
        )
        for index, (X_chunk, y_chunk) in iterator:
            chunk_started = time.time()
            model = TabPFNWrapper(
                device=self.device,
                random_state=self.random_state,
                n_estimators=self.n_estimators,
                context_limit=self.chunk_size,
                predict_batch_size=self.predict_batch_size,
            )
            model.fit(X_chunk, y_chunk)
            proba = model.predict_proba(X_test)

            confidence = self._confidence(proba)
            per_chunk.append(proba)
            self.chunk_confidences_.append(confidence)

            # Release the model before building the next one. MPS does not
            # return memory when the Python reference is dropped, so without
            # this the run dies with TabPFNMPSOutOfMemoryError partway through
            # -- observed at chunk 4 of 13 on a 16 GB M1.
            model.free()
            del model

            logger.info(
                "chunk %d/%d: %d context rows, confidence %.4f, %.1fs",
                index,
                len(chunks),
                len(y_chunk),
                confidence,
                time.time() - chunk_started,
            )

        aggregated = self._aggregate(per_chunk)
        self.predict_seconds = time.time() - started
        logger.info(
            "Aggregated %d chunks in %.1fs total (%.1fs per chunk)",
            len(chunks),
            self.predict_seconds,
            self.predict_seconds / len(chunks),
        )
        return aggregated

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class labels from the aggregated probabilities.

        Args:
            X_test: Test features.

        Returns:
            Predicted labels, one per test row.
        """
        proba = self.predict_proba(X_test)
        classes = self.classes_ if self.classes_ is not None else np.arange(
            proba.shape[1]
        )
        return np.asarray(classes)[np.argmax(proba, axis=1)]

    def describe(self) -> dict[str, Any]:
        """Return ensemble configuration and timings for the results CSV.

        Returns:
            A dict describing the chunking, aggregation and timings.
        """
        described: dict[str, Any] = {
            "aggregation": self.aggregation,
            "chunk_size": self.chunk_size,
            "n_estimators": self.n_estimators,
            "device": config.resolve_device(self.device),
            "fit_seconds": self.fit_seconds,
            "predict_seconds": self.predict_seconds,
        }
        if self.chunks_ is not None:
            described.update(describe_chunks(self.chunks_))
        if self.chunk_confidences_:
            described["mean_chunk_confidence"] = float(
                np.mean(self.chunk_confidences_)
            )
        return described
