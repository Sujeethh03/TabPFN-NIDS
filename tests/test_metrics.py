"""Tests for tabpfn_nids.evaluation.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from tabpfn_nids.evaluation.metrics import compute_metrics, format_metrics


def test_perfect_prediction_scores_one() -> None:
    """A perfect classifier scores 1.0 on every metric."""
    y = np.array([0, 1, 0, 1, 1])
    metrics = compute_metrics(y, y, y.astype(float))
    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        assert metrics[key] == pytest.approx(1.0)


def test_confusion_matrix_orientation() -> None:
    """The matrix is [[TN, FP], [FN, TP]] with attack as the positive class."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0, 0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["confusion_matrix"] == [[2, 1], [2, 2]]
    assert metrics["true_negatives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 2
    assert metrics["true_positives"] == 2


def test_precision_recall_use_attack_as_positive() -> None:
    """Positive class is 1 (attack), not 0 (normal)."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["precision"] == pytest.approx(0.5)  # 1 TP / (1 TP + 1 FP)
    assert metrics["recall"] == pytest.approx(0.5)  # 1 TP / (1 TP + 1 FN)


def test_accepts_2d_predict_proba_output() -> None:
    """A (n, 2) array from predict_proba uses column 1 for ROC-AUC."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    proba_2d = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.1, 0.9]])
    flat = compute_metrics(y_true, y_pred, proba_2d[:, 1])
    wide = compute_metrics(y_true, y_pred, proba_2d)
    assert flat["roc_auc"] == wide["roc_auc"] == pytest.approx(1.0)


def test_roc_auc_is_none_without_probabilities() -> None:
    """ROC-AUC is reported as None rather than silently substituted."""
    metrics = compute_metrics([0, 1], [0, 1])
    assert metrics["roc_auc"] is None


def test_single_class_ground_truth_yields_none_auc() -> None:
    """ROC-AUC is undefined with one class present; matrix stays 2x2."""
    metrics = compute_metrics([1, 1, 1], [1, 1, 0], [0.9, 0.8, 0.2])
    assert metrics["roc_auc"] is None
    assert np.array(metrics["confusion_matrix"]).shape == (2, 2)


def test_all_normal_predictions_do_not_divide_by_zero() -> None:
    """A degenerate model scores 0, not NaN."""
    metrics = compute_metrics([0, 1, 1], [0, 0, 0])
    assert metrics["precision"] == 0.0
    assert metrics["f1_score"] == 0.0


def test_mismatched_lengths_raise() -> None:
    """Silently truncating would corrupt every downstream metric."""
    with pytest.raises(ValueError, match="rows"):
        compute_metrics([0, 1, 1], [0, 1])


def test_bad_proba_shape_raises() -> None:
    """A multi-class probability array is rejected, not misread."""
    with pytest.raises(ValueError, match="expected"):
        compute_metrics([0, 1], [0, 1], np.zeros((2, 3)))


def test_support_counts_match_input() -> None:
    """Class supports are reported for the report tables."""
    metrics = compute_metrics([0, 0, 0, 1, 1], [0, 0, 0, 1, 1])
    assert metrics["support_normal"] == 3
    assert metrics["support_attack"] == 2


def test_format_metrics_renders_all_fields() -> None:
    """The printable block includes every headline metric."""
    text = format_metrics(compute_metrics([0, 1], [0, 1], [0.1, 0.9]), title="T")
    for token in ("accuracy", "precision", "recall", "f1_score", "roc_auc", "T"):
        assert token in text


def test_format_metrics_handles_missing_auc() -> None:
    """A None ROC-AUC renders as 'n/a' rather than crashing."""
    assert "n/a" in format_metrics(compute_metrics([0, 1], [0, 1]))
