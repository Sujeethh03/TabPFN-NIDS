"""Tests for the chunked TabPFN ensemble (Build Plan step 3.5).

Aggregation is tested directly against hand-built probability matrices, so
the maths is verified without paying for TabPFN inference. The end-to-end
tests run the real model on a tiny synthetic problem and are marked ``slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tabpfn_nids import config
from tabpfn_nids.models.chunked_ensemble import (
    AGGREGATION_STRATEGIES,
    ChunkedTabPFNEnsemble,
)


@pytest.fixture
def separable():
    """A small, linearly separable binary problem with a held-out test set."""
    rng = np.random.default_rng(config.SEED)
    X = np.vstack([rng.normal(0, 1, (100, 5)), rng.normal(4, 1, (100, 5))])
    y = np.array([0] * 100 + [1] * 100)
    X_test = np.vstack([rng.normal(0, 1, (20, 5)), rng.normal(4, 1, (20, 5))])
    y_test = np.array([0] * 20 + [1] * 20)
    return X, y, X_test, y_test


# --------------------------------------------------------------------------
# Configuration and validation
# --------------------------------------------------------------------------


def test_rejects_unknown_aggregation() -> None:
    """Only the documented strategies are accepted."""
    with pytest.raises(ValueError, match="unknown aggregation"):
        ChunkedTabPFNEnsemble(aggregation="stacking")


def test_rejects_chunk_size_above_the_context_limit() -> None:
    """A chunk that cannot fit one context defeats the whole design."""
    with pytest.raises(ValueError, match="exceeds TabPFN's context limit"):
        ChunkedTabPFNEnsemble(chunk_size=config.MAX_CONTEXT_SAMPLES + 1)


def test_predict_before_fit_raises(separable) -> None:
    """An unfitted ensemble has no chunks to predict from."""
    _, _, X_test, _ = separable
    with pytest.raises(RuntimeError, match="fit"):
        ChunkedTabPFNEnsemble().predict_proba(X_test)


def test_fit_creates_the_expected_chunks(separable) -> None:
    """fit partitions but trains nothing; TabPFN runs at predict time."""
    X, y, _, _ = separable
    ensemble = ChunkedTabPFNEnsemble(chunk_size=50, show_progress=False).fit(X, y)
    assert len(ensemble.chunks_) == 4
    assert all(len(chunk_y) <= 50 for _, chunk_y in ensemble.chunks_)
    assert ensemble.fit_seconds is not None
    assert np.array_equal(ensemble.classes_, np.array([0, 1]))


# --------------------------------------------------------------------------
# Aggregation maths, without running TabPFN
# --------------------------------------------------------------------------


def _prepared(aggregation: str, confidences: list[float]) -> ChunkedTabPFNEnsemble:
    """Build an ensemble with confidences pre-set for aggregation testing."""
    ensemble = ChunkedTabPFNEnsemble(aggregation=aggregation, show_progress=False)
    ensemble.chunk_confidences_ = confidences
    return ensemble


@pytest.mark.parametrize("aggregation", AGGREGATION_STRATEGIES)
def test_aggregated_probabilities_are_valid(aggregation) -> None:
    """Output rows sum to 1 and every entry lies in [0, 1]."""
    per_chunk = [
        np.array([[0.9, 0.1], [0.2, 0.8]]),
        np.array([[0.6, 0.4], [0.3, 0.7]]),
        np.array([[0.5, 0.5], [0.1, 0.9]]),
    ]
    ensemble = _prepared(aggregation, [0.9, 0.7, 0.5])
    aggregated = ensemble._aggregate(per_chunk)

    assert aggregated.shape == (2, 2)
    assert np.allclose(aggregated.sum(axis=1), 1.0)
    assert (aggregated >= 0).all() and (aggregated <= 1).all()


def test_majority_is_an_unweighted_mean() -> None:
    """'majority' averages the chunk probabilities equally."""
    per_chunk = [
        np.array([[1.0, 0.0]]),
        np.array([[0.0, 1.0]]),
        np.array([[0.5, 0.5]]),
    ]
    ensemble = _prepared("majority", [0.99, 0.51, 0.60])
    aggregated = ensemble._aggregate(per_chunk)
    assert np.allclose(aggregated, [[0.5, 0.5]])
    assert np.allclose(ensemble.chunk_weights_, 1 / 3)


def test_weighted_vote_favours_confident_chunks() -> None:
    """A confident chunk pulls the result further than an unsure one."""
    per_chunk = [np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])]
    ensemble = _prepared("weighted_vote", [0.9, 0.5])
    aggregated = ensemble._aggregate(per_chunk)

    expected_first = 0.9 / 1.4
    assert aggregated[0, 0] == pytest.approx(expected_first)
    assert aggregated[0, 0] > aggregated[0, 1]
    assert ensemble.chunk_weights_[0] > ensemble.chunk_weights_[1]


def test_weights_are_normalised() -> None:
    """Chunk weights always sum to 1 regardless of raw confidences."""
    per_chunk = [np.array([[0.7, 0.3]])] * 4
    ensemble = _prepared("weighted_vote", [0.55, 0.99, 0.72, 0.61])
    ensemble._aggregate(per_chunk)
    assert ensemble.chunk_weights_.sum() == pytest.approx(1.0)


def test_degenerate_confidences_fall_back_to_equal_weights() -> None:
    """Zero total confidence must not produce NaNs."""
    per_chunk = [np.array([[0.8, 0.2]]), np.array([[0.4, 0.6]])]
    ensemble = _prepared("weighted_vote", [0.0, 0.0])
    aggregated = ensemble._aggregate(per_chunk)
    assert np.isfinite(aggregated).all()
    assert np.allclose(ensemble.chunk_weights_, 0.5)


def test_single_chunk_aggregation_is_the_identity() -> None:
    """With one chunk the ensemble reduces to plain TabPFN."""
    per_chunk = [np.array([[0.73, 0.27], [0.11, 0.89]])]
    ensemble = _prepared("weighted_vote", [0.8])
    assert np.allclose(ensemble._aggregate(per_chunk), per_chunk[0])


def test_confidence_measures_decisiveness() -> None:
    """Confidence is the mean per-row maximum probability."""
    confident = np.array([[0.99, 0.01], [0.02, 0.98]])
    unsure = np.array([[0.51, 0.49], [0.48, 0.52]])
    assert ChunkedTabPFNEnsemble._confidence(confident) == pytest.approx(0.985)
    assert ChunkedTabPFNEnsemble._confidence(unsure) == pytest.approx(0.515)


# --------------------------------------------------------------------------
# End to end, with real TabPFN inference
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("aggregation", AGGREGATION_STRATEGIES)
def test_end_to_end_on_synthetic_data(separable, aggregation) -> None:
    """The ensemble runs, and its output has the right shape and support."""
    X, y, X_test, y_test = separable
    ensemble = ChunkedTabPFNEnsemble(
        chunk_size=50,
        aggregation=aggregation,
        n_estimators=1,
        show_progress=False,
    ).fit(X, y)

    proba = ensemble.predict_proba(X_test)
    predictions = ensemble.predict(X_test)

    assert proba.shape == (len(y_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all() and (proba <= 1).all()
    assert predictions.shape == (len(y_test),)
    assert set(np.unique(predictions)) <= {0, 1}
    assert (predictions == y_test).mean() > 0.9
    assert ensemble.predict_seconds is not None
    assert len(ensemble.chunk_confidences_) == len(ensemble.chunks_)


@pytest.mark.slow
def test_describe_reports_chunking_and_timings(separable) -> None:
    """describe() supplies the provenance columns for the results CSV."""
    X, y, X_test, _ = separable
    ensemble = ChunkedTabPFNEnsemble(
        chunk_size=50, n_estimators=1, show_progress=False
    ).fit(X, y)
    ensemble.predict_proba(X_test)

    described = ensemble.describe()
    assert described["n_chunks"] == 4
    assert described["aggregation"] == "weighted_vote"
    assert described["total_rows"] == len(y)
    assert 0.5 <= described["mean_chunk_confidence"] <= 1.0
