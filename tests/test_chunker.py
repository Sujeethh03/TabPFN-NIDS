"""Tests for the stratified chunker (Build Plan step 3.5)."""

from __future__ import annotations

import numpy as np
import pytest

from tabpfn_nids import config
from tabpfn_nids.models.chunker import (
    describe_chunks,
    make_chunks,
    n_chunks_for,
    random_chunk,
    stratified_chunk,
)

BALANCE_TOLERANCE = 0.02


def make_data(n_rows: int, positive_rate: float = 0.465, n_features: int = 8):
    """Build a synthetic dataset with a chosen class balance.

    Args:
        n_rows: Number of rows.
        positive_rate: Fraction of rows labelled 1.
        n_features: Number of feature columns.

    Returns:
        An ``(X, y)`` tuple.
    """
    rng = np.random.default_rng(config.SEED)
    n_positive = int(round(n_rows * positive_rate))
    y = np.array([1] * n_positive + [0] * (n_rows - n_positive))
    rng.shuffle(y)
    return rng.normal(size=(n_rows, n_features)), y


# --------------------------------------------------------------------------
# Chunk counts
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_rows", "chunk_size", "expected"),
    [
        (500, 10_000, 1),  # smaller than a chunk
        (10_000, 10_000, 1),  # exactly one chunk
        (10_001, 10_000, 2),  # one row over
        (20_000, 10_000, 2),
        (125_973, 10_000, 13),  # NSL-KDD training split
        (100, 25, 4),
    ],
)
def test_chunk_count_for_various_sizes(n_rows, chunk_size, expected) -> None:
    """The partition uses the fewest chunks that respect the size cap."""
    assert n_chunks_for(n_rows, chunk_size) == expected
    X, y = make_data(n_rows)
    assert len(stratified_chunk(X, y, chunk_size=chunk_size)) == expected


def test_data_smaller_than_chunk_size_returns_one_chunk() -> None:
    """A dataset below the limit is returned whole, not split."""
    X, y = make_data(500)
    chunks = stratified_chunk(X, y, chunk_size=10_000)
    assert len(chunks) == 1
    assert chunks[0][0].shape == X.shape
    assert np.array_equal(chunks[0][1], y)


# --------------------------------------------------------------------------
# Size and coverage
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_rows", [1_000, 25_000, 125_973])
def test_no_chunk_exceeds_chunk_size(n_rows) -> None:
    """The hard constraint: every chunk must fit in one TabPFN context."""
    X, y = make_data(n_rows)
    chunks = stratified_chunk(X, y, chunk_size=10_000)
    assert all(len(chunk_y) <= 10_000 for _, chunk_y in chunks)
    assert all(chunk_X.shape[0] <= 10_000 for chunk_X, _ in chunks)


def test_chunks_are_disjoint_and_exhaustive() -> None:
    """Every row appears exactly once across the partition."""
    X, y = make_data(5_000)
    chunks = stratified_chunk(X, y, chunk_size=1_000)
    assert sum(len(chunk_y) for _, chunk_y in chunks) == len(y)
    rows = np.vstack([chunk_X for chunk_X, _ in chunks])
    assert rows.shape == X.shape
    # Row sums are a cheap fingerprint for set equality of the rows.
    assert np.allclose(np.sort(rows.sum(axis=1)), np.sort(X.sum(axis=1)))


def test_features_and_labels_stay_aligned() -> None:
    """Chunking must not shuffle X against y."""
    rng = np.random.default_rng(config.SEED)
    y = rng.integers(0, 2, 400)
    # Encode the label in a feature column so misalignment is detectable.
    X = np.column_stack([y.astype(float), rng.normal(size=(400, 3))])
    for chunk_X, chunk_y in stratified_chunk(X, y, chunk_size=100):
        assert np.array_equal(chunk_X[:, 0].astype(int), chunk_y)


# --------------------------------------------------------------------------
# Stratification
# --------------------------------------------------------------------------


@pytest.mark.parametrize("positive_rate", [0.5, 0.465, 0.2, 0.05])
def test_chunks_preserve_class_balance_within_two_percent(positive_rate) -> None:
    """Each chunk's positive rate must track the population within +/-2%."""
    X, y = make_data(50_000, positive_rate=positive_rate)
    chunks = stratified_chunk(X, y, chunk_size=10_000)
    overall = y.mean()
    for _, chunk_y in chunks:
        assert abs(chunk_y.mean() - overall) <= BALANCE_TOLERANCE


def test_real_world_balance_drift_is_negligible() -> None:
    """At NSL-KDD's size and balance, drift should be far below tolerance."""
    X, y = make_data(125_973, positive_rate=0.4654)
    chunks = stratified_chunk(X, y, chunk_size=10_000)
    drift = max(abs(chunk_y.mean() - y.mean()) for _, chunk_y in chunks)
    assert drift < 0.001


def test_every_chunk_contains_both_classes() -> None:
    """A chunk missing a class could not predict it at all."""
    X, y = make_data(30_000, positive_rate=0.05)
    for _, chunk_y in stratified_chunk(X, y, chunk_size=10_000):
        assert set(np.unique(chunk_y)) == {0, 1}


def test_chunking_is_reproducible() -> None:
    """The same seed yields the same partition (RULE 5)."""
    X, y = make_data(20_000)
    first = stratified_chunk(X, y, chunk_size=5_000, random_state=42)
    second = stratified_chunk(X, y, chunk_size=5_000, random_state=42)
    for (Xa, ya), (Xb, yb) in zip(first, second):
        assert np.array_equal(Xa, Xb)
        assert np.array_equal(ya, yb)


def test_different_seeds_give_different_partitions() -> None:
    """The shuffle actually depends on the seed."""
    X, y = make_data(20_000)
    a = stratified_chunk(X, y, chunk_size=5_000, random_state=42)
    b = stratified_chunk(X, y, chunk_size=5_000, random_state=7)
    assert not np.array_equal(a[0][0], b[0][0])


# --------------------------------------------------------------------------
# Options and validation
# --------------------------------------------------------------------------


def test_max_chunks_caps_the_partition() -> None:
    """max_chunks bounds runtime by truncating the chunk list."""
    X, y = make_data(100_000)
    chunks = stratified_chunk(X, y, chunk_size=10_000, max_chunks=3)
    assert len(chunks) == 3
    assert all(len(chunk_y) <= 10_000 for _, chunk_y in chunks)


def test_mismatched_lengths_raise() -> None:
    """X and y must agree on row count."""
    with pytest.raises(ValueError, match="rows but y has"):
        stratified_chunk(np.zeros((10, 3)), np.zeros(9))


def test_empty_input_raises() -> None:
    """Chunking nothing is a caller error, not an empty list."""
    with pytest.raises(ValueError, match="empty"):
        stratified_chunk(np.zeros((0, 3)), np.zeros(0))


def test_non_positive_chunk_size_raises() -> None:
    """A zero or negative chunk size is rejected."""
    X, y = make_data(100)
    with pytest.raises(ValueError, match="must be positive"):
        stratified_chunk(X, y, chunk_size=0)


def test_describe_chunks_summary() -> None:
    """describe_chunks reports the fields used by the results CSV."""
    X, y = make_data(50_000)
    summary = describe_chunks(stratified_chunk(X, y, chunk_size=10_000))
    assert summary["n_chunks"] == 5
    assert summary["total_rows"] == 50_000
    assert summary["max_chunk_size"] <= 10_000
    assert summary["max_positive_rate_drift"] < BALANCE_TOLERANCE


# --------------------------------------------------------------------------
# Random chunking — the control arm for the stratification ablation
# --------------------------------------------------------------------------


def test_random_chunk_partitions_exhaustively() -> None:
    """The random control still uses every row exactly once."""
    X, y = make_data(50_000)
    chunks = random_chunk(X, y, chunk_size=10_000)
    assert len(chunks) == 5
    assert sum(len(chunk_y) for _, chunk_y in chunks) == len(y)
    assert all(len(chunk_y) <= 10_000 for _, chunk_y in chunks)


def test_random_chunk_drifts_more_than_stratified() -> None:
    """The mechanism the ablation measures: stratified drift is zero."""
    X, y = make_data(30_000, positive_rate=0.05)
    strat_drift = max(
        abs(cy.mean() - y.mean())
        for _, cy in stratified_chunk(X, y, chunk_size=10_000)
    )
    rand_drift = max(
        abs(cy.mean() - y.mean())
        for _, cy in random_chunk(X, y, chunk_size=10_000)
    )
    assert strat_drift < 1e-3
    assert rand_drift > strat_drift


def test_random_chunk_keeps_features_and_labels_aligned() -> None:
    """Shuffling must permute X and y together."""
    rng = np.random.default_rng(config.SEED)
    y = rng.integers(0, 2, 400)
    X = np.column_stack([y.astype(float), rng.normal(size=(400, 3))])
    for chunk_X, chunk_y in random_chunk(X, y, chunk_size=100):
        assert np.array_equal(chunk_X[:, 0].astype(int), chunk_y)


def test_random_chunk_is_reproducible() -> None:
    """Same seed, same partition (RULE 5)."""
    X, y = make_data(20_000)
    a = random_chunk(X, y, chunk_size=5_000, random_state=42)
    b = random_chunk(X, y, chunk_size=5_000, random_state=42)
    for (Xa, ya), (Xb, yb) in zip(a, b):
        assert np.array_equal(Xa, Xb) and np.array_equal(ya, yb)


def test_random_chunk_validates_input() -> None:
    """Same validation contract as the stratified path."""
    with pytest.raises(ValueError, match="rows but y has"):
        random_chunk(np.zeros((10, 3)), np.zeros(9))
    with pytest.raises(ValueError, match="empty"):
        random_chunk(np.zeros((0, 3)), np.zeros(0))


def test_make_chunks_dispatches_on_the_flag() -> None:
    """make_chunks selects the stratified or random implementation."""
    X, y = make_data(20_000, positive_rate=0.1)
    strat = make_chunks(X, y, chunk_size=5_000, stratified=True)
    rand = make_chunks(X, y, chunk_size=5_000, stratified=False)
    strat_drift = max(abs(cy.mean() - y.mean()) for _, cy in strat)
    rand_drift = max(abs(cy.mean() - y.mean()) for _, cy in rand)
    assert strat_drift <= rand_drift
    assert len(strat) == len(rand) == 4
