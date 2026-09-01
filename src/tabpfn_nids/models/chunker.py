"""Stratified chunking for the TabPFN chunked ensemble (Enhancement 1).

TabPFN v2 accepts at most 10,000 in-context training samples. To use a full
NIDS training set, the data is partitioned into disjoint chunks that each fit
inside that limit and each preserve the class balance of the whole.

Stratification matters more here than in an ordinary train/test split. A
random partition of NSL-KDD would give chunks whose attack rate drifts around
the 46.5% population value, and because each chunk becomes an independent
TabPFN context, that drift turns directly into a per-chunk prior shift. Rare
attack classes are the ones that suffer: a chunk that happens to contain no
examples of an attack family cannot recognise it at all.

The partition is disjoint by default. ``StratifiedKFold`` is used rather than
repeated ``StratifiedShuffleSplit`` draws because its folds are guaranteed to
be mutually exclusive and jointly exhaustive, so every training row is used
exactly once and no row is duplicated across contexts.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from sklearn.model_selection import StratifiedKFold

from tabpfn_nids import config

logger = logging.getLogger(__name__)

Chunk = tuple[np.ndarray, np.ndarray]


def n_chunks_for(n_rows: int, chunk_size: int) -> int:
    """Return the number of chunks needed to cover n_rows.

    Args:
        n_rows: Total rows to partition.
        chunk_size: Maximum rows per chunk.

    Returns:
        The smallest chunk count such that every chunk fits in chunk_size.
    """
    if n_rows <= chunk_size:
        return 1
    return math.ceil(n_rows / chunk_size)


def stratified_chunk(
    X: np.ndarray,
    y: np.ndarray,
    chunk_size: int = config.MAX_CONTEXT_SAMPLES,
    random_state: int = config.SEED,
    max_chunks: int | None = None,
) -> list[Chunk]:
    """Partition a dataset into stratified chunks of at most ``chunk_size``.

    Chunks are disjoint: each input row appears in exactly one chunk (unless
    ``max_chunks`` truncates the partition). Every chunk reproduces the class
    proportions of the input to within one sample per class.

    Args:
        X: Feature matrix of shape ``(n_rows, n_features)``.
        y: Labels of shape ``(n_rows,)``.
        chunk_size: Maximum rows per chunk. Defaults to TabPFN's 10,000-sample
            context limit.
        random_state: Seed controlling the shuffle before partitioning.
        max_chunks: If given, return at most this many chunks. Used to bound
            runtime, since inference cost grows linearly with chunk count.

    Returns:
        A list of ``(X_chunk, y_chunk)`` tuples.

    Raises:
        ValueError: If X and y disagree on length, if the input is empty, or
            if chunk_size is not positive.

    Example:
        >>> chunks = stratified_chunk(X, y, chunk_size=10_000)
        >>> all(len(yc) <= 10_000 for _, yc in chunks)
        True
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X has {X.shape[0]} rows but y has {y.shape[0]}."
        )
    if X.shape[0] == 0:
        raise ValueError("cannot chunk an empty dataset")

    n_rows = X.shape[0]
    n_chunks = n_chunks_for(n_rows, chunk_size)

    # A single chunk needs no partitioning, and StratifiedKFold requires at
    # least two splits.
    if n_chunks == 1:
        logger.info(
            "Dataset of %d rows fits in one chunk of %d; no partitioning.",
            n_rows,
            chunk_size,
        )
        return [(X, y)]

    splitter = StratifiedKFold(
        n_splits=n_chunks, shuffle=True, random_state=random_state
    )
    chunks: list[Chunk] = [
        (X[index], y[index]) for _, index in splitter.split(X, y)
    ]

    if max_chunks is not None and len(chunks) > max_chunks:
        logger.info(
            "Truncating %d chunks to max_chunks=%d.", len(chunks), max_chunks
        )
        chunks = chunks[:max_chunks]

    _log_chunk_balance(chunks, y)
    return chunks


def random_chunk(
    X: np.ndarray,
    y: np.ndarray,
    chunk_size: int = config.MAX_CONTEXT_SAMPLES,
    random_state: int = config.SEED,
    max_chunks: int | None = None,
) -> list[Chunk]:
    """Partition a dataset into chunks WITHOUT preserving class balance.

    This exists as the control arm for the stratification ablation. It is not
    a supported production path: each chunk's class balance is left to chance,
    which is exactly the failure mode ``stratified_chunk`` avoids.

    The difference is largest for rare classes. With a 46.5% positive rate and
    9,691-row chunks the drift is small by the law of large numbers, so a
    binary NSL-KDD comparison understates the risk; the danger case is a rare
    attack family that lands entirely outside some chunk, leaving that chunk
    unable to recognise it at all.

    Args:
        X: Feature matrix of shape ``(n_rows, n_features)``.
        y: Labels of shape ``(n_rows,)``.
        chunk_size: Maximum rows per chunk.
        random_state: Seed for the shuffle.
        max_chunks: If given, return at most this many chunks.

    Returns:
        A list of ``(X_chunk, y_chunk)`` tuples.

    Raises:
        ValueError: If X and y disagree on length, the input is empty, or
            chunk_size is not positive.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}.")
    if X.shape[0] == 0:
        raise ValueError("cannot chunk an empty dataset")

    n_rows = X.shape[0]
    n_chunks = n_chunks_for(n_rows, chunk_size)

    if n_chunks == 1:
        return [(X, y)]

    rng = np.random.default_rng(random_state)
    order = rng.permutation(n_rows)
    # array_split handles a non-divisible length by making the first chunks
    # one row larger, so the partition stays exhaustive.
    chunks: list[Chunk] = [
        (X[index], y[index]) for index in np.array_split(order, n_chunks)
    ]

    if max_chunks is not None and len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]

    _log_chunk_balance(chunks, y, label="random")
    return chunks


def make_chunks(
    X: np.ndarray,
    y: np.ndarray,
    chunk_size: int = config.MAX_CONTEXT_SAMPLES,
    random_state: int = config.SEED,
    max_chunks: int | None = None,
    stratified: bool = True,
) -> list[Chunk]:
    """Partition a dataset, stratified or not.

    Args:
        X: Feature matrix.
        y: Labels.
        chunk_size: Maximum rows per chunk.
        random_state: Seed.
        max_chunks: Optional cap on chunk count.
        stratified: True for class-balance-preserving chunks, False for the
            random control used by the stratification ablation.

    Returns:
        A list of ``(X_chunk, y_chunk)`` tuples.
    """
    chunker = stratified_chunk if stratified else random_chunk
    return chunker(
        X, y, chunk_size=chunk_size, random_state=random_state,
        max_chunks=max_chunks,
    )


def _log_chunk_balance(
    chunks: list[Chunk], y_full: np.ndarray, label: str = "stratified"
) -> None:
    """Log chunk count, sizes and per-chunk class balance.

    Args:
        chunks: The produced chunks.
        y_full: Labels of the full dataset, for the reference balance.
    """
    overall = float(np.mean(y_full))
    sizes = [len(chunk_y) for _, chunk_y in chunks]
    rates = [float(np.mean(chunk_y)) for _, chunk_y in chunks]
    drift = max(abs(rate - overall) for rate in rates)

    logger.info(
        "Created %d %s chunks: sizes %d-%d, positive rate %.4f "
        "(population %.4f), max drift %.4f",
        len(chunks),
        label,
        min(sizes),
        max(sizes),
        float(np.mean(rates)),
        overall,
        drift,
    )
    if drift > 0.02:
        logger.warning(
            "Chunk class balance drifts by %.4f from the population rate; "
            "stratification may be degraded by very rare classes.",
            drift,
        )
    return None


def describe_chunks(chunks: list[Chunk]) -> dict[str, object]:
    """Summarise a chunk list for logging and the results CSV.

    Args:
        chunks: The chunks to describe.

    Returns:
        A dict of chunk count, size range, total rows and positive rates.
    """
    sizes = [len(chunk_y) for _, chunk_y in chunks]
    rates = [float(np.mean(chunk_y)) for _, chunk_y in chunks]
    return {
        "n_chunks": len(chunks),
        "min_chunk_size": min(sizes),
        "max_chunk_size": max(sizes),
        "total_rows": sum(sizes),
        "mean_positive_rate": float(np.mean(rates)),
        "max_positive_rate_drift": float(max(rates) - min(rates)),
    }
