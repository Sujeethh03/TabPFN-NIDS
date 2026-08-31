"""Tests for tabpfn_nids.models.tabpfn_wrapper.

These cover validation, configuration and error handling without running
TabPFN inference, which is far too slow for a unit test suite. The one test
that does run the model is marked ``slow`` and deselected by default.
"""

from __future__ import annotations

import numpy as np
import pytest

from tabpfn_nids import config
from tabpfn_nids.models import TabPFNWrapper


@pytest.fixture
def tiny() -> tuple[np.ndarray, np.ndarray]:
    """A small, well-separated binary problem."""
    rng = np.random.default_rng(config.SEED)
    X = np.vstack([rng.normal(0, 1, (40, 6)), rng.normal(4, 1, (40, 6))])
    y = np.array([0] * 40 + [1] * 40)
    return X, y


def test_context_limit_raises_with_actionable_message() -> None:
    """Over the limit must raise, and the message must name the fix."""
    X = np.zeros((config.MAX_CONTEXT_SAMPLES + 1, 4))
    y = np.zeros(config.MAX_CONTEXT_SAMPLES + 1)
    with pytest.raises(ValueError) as excinfo:
        TabPFNWrapper().fit(X, y)
    message = str(excinfo.value)
    assert "context limit exceeded" in message
    assert "chunked ensemble" in message
    assert "10,000" in message


def test_exactly_at_the_limit_is_accepted() -> None:
    """The check is > limit, not >= limit."""
    wrapper = TabPFNWrapper()
    wrapper._check_context_size(np.zeros((config.MAX_CONTEXT_SAMPLES, 4)))


def test_custom_context_limit_is_honoured() -> None:
    """A smaller limit can be imposed, e.g. for per-chunk sizing."""
    with pytest.raises(ValueError, match="limit is 500"):
        TabPFNWrapper(context_limit=500).fit(np.zeros((501, 3)), np.zeros(501))


def test_mismatched_x_and_y_raise() -> None:
    """A row-count mismatch is caught before reaching TabPFN."""
    with pytest.raises(ValueError, match="rows but y_train"):
        TabPFNWrapper().fit(np.zeros((10, 3)), np.zeros(9))


def test_predict_before_fit_raises(tiny) -> None:
    """Predicting from an unfitted wrapper is an error."""
    X, _ = tiny
    with pytest.raises(RuntimeError, match="fit"):
        TabPFNWrapper().predict(X)
    with pytest.raises(RuntimeError, match="fit"):
        TabPFNWrapper().predict_proba(X)


def test_device_resolution() -> None:
    """'auto' resolves to a real backend; explicit choices pass through."""
    assert TabPFNWrapper(device="auto").device in {"mps", "cpu"}
    assert TabPFNWrapper(device="cpu").device == "cpu"


def test_checkpoint_defaults_to_the_pinned_v2_model() -> None:
    """Never the gated 'auto' default (design-setup.md 2.3)."""
    assert TabPFNWrapper().model_path == config.TABPFN_CHECKPOINT
    assert TabPFNWrapper().model_path != "auto"


def test_describe_reports_configuration() -> None:
    """describe() supplies the provenance columns for the results CSV."""
    described = TabPFNWrapper(device="cpu").describe()
    assert described["device"] == "cpu"
    assert described["checkpoint"] == config.TABPFN_CHECKPOINT
    assert described["fit_seconds"] is None


def test_excess_features_warn_but_do_not_raise(caplog) -> None:
    """Feature count is a soft limit, unlike the context size."""
    wrapper = TabPFNWrapper()
    with caplog.at_level("WARNING"):
        wrapper._check_feature_count(np.zeros((5, config.MAX_FEATURES + 1)))
    assert "exceeds TabPFN's pretraining limit" in caplog.text


@pytest.mark.slow
def test_end_to_end_inference_on_a_tiny_problem(tiny) -> None:
    """A real fit/predict cycle; slow, so deselected by default."""
    X, y = tiny
    model = TabPFNWrapper(random_state=config.SEED, n_estimators=1).fit(X, y)
    predictions = model.predict(X)
    proba = model.predict_proba(X)

    assert predictions.shape == (len(y),)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (predictions == y).mean() > 0.9
    assert model.fit_seconds is not None and model.predict_seconds is not None
