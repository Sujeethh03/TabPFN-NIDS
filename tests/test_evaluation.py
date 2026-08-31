"""Tests for significance, reporting and plotting (Build Plan phase 5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tabpfn_nids.evaluation.metrics import compute_metrics
from tabpfn_nids.evaluation.plots import (
    plot_all,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from tabpfn_nids.evaluation.reporter import (
    build_result_row,
    load_results,
    write_results,
)
from tabpfn_nids.evaluation.significance import (
    minimum_reachable_p,
    summarize_paired,
    wilcoxon_test,
)


@pytest.fixture
def predictions():
    """A modest binary problem with scores, for plotting and metrics."""
    rng = np.random.default_rng(42)
    y_true = np.array([0] * 60 + [1] * 40)
    scores = np.clip(
        np.concatenate([rng.normal(0.3, 0.15, 60), rng.normal(0.7, 0.15, 40)]),
        0,
        1,
    )
    return y_true, scores


# --------------------------------------------------------------------------
# Significance
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "floor"),
    [(1, 1.0), (2, 0.5), (3, 0.25), (4, 0.125), (5, 0.0625), (6, 0.03125)],
)
def test_minimum_reachable_p(n, floor) -> None:
    """The published floors on Wilcoxon's two-sided p-value."""
    assert minimum_reachable_p(n) == pytest.approx(floor)


def test_three_seeds_are_flagged_underpowered() -> None:
    """The Build Plan's 3-seed design cannot reach p < 0.05."""
    result = wilcoxon_test([0.73, 0.75, 0.77], [0.74, 0.76, 0.79])
    assert result["underpowered"] is True
    assert result["p_value"] is None
    assert result["significant"] is None
    assert result["min_reachable_p"] == pytest.approx(0.25)


def test_six_seeds_are_powered_and_return_a_p_value() -> None:
    """At n=6 the floor drops below alpha, so a test is meaningful."""
    baseline = [0.70, 0.71, 0.72, 0.73, 0.74, 0.75]
    enhanced = [0.74, 0.75, 0.76, 0.77, 0.78, 0.79]
    result = wilcoxon_test(baseline, enhanced)
    assert result["underpowered"] is False
    assert result["p_value"] is not None
    assert result["significant"] is True
    assert result["mean_difference"] == pytest.approx(0.04)


def test_mean_and_median_differences_are_reported() -> None:
    """Effect size is reported even when the test is underpowered."""
    result = wilcoxon_test([0.70, 0.80], [0.75, 0.90])
    assert result["mean_difference"] == pytest.approx(0.075)
    assert result["median_difference"] == pytest.approx(0.075)


def test_mismatched_lengths_raise() -> None:
    """Scores must be paired by seed."""
    with pytest.raises(ValueError, match="equal lengths"):
        wilcoxon_test([0.1, 0.2], [0.1])


def test_empty_scores_raise() -> None:
    """An empty comparison is a caller error."""
    with pytest.raises(ValueError, match="empty"):
        wilcoxon_test([], [])


def test_summarize_paired_reports_means_and_spread() -> None:
    """The honest small-n summary: means, stds and the difference."""
    summary = summarize_paired([0.73, 0.75, 0.78], [0.75, 0.77, 0.80])
    assert summary["n"] == 3
    assert summary["baseline_mean"] == pytest.approx(0.7533, abs=1e-4)
    assert summary["mean_difference"] == pytest.approx(0.02, abs=1e-9)
    assert summary["baseline_std"] > 0


def test_difference_is_compared_against_baseline_spread() -> None:
    """A delta inside the seed-to-seed spread is not an effect."""
    noisy = summarize_paired([0.70, 0.75, 0.80], [0.71, 0.76, 0.81])
    assert noisy["difference_exceeds_baseline_spread"] is False

    clear = summarize_paired([0.700, 0.701, 0.702], [0.80, 0.81, 0.82])
    assert clear["difference_exceeds_baseline_spread"] is True


# --------------------------------------------------------------------------
# Reporter
# --------------------------------------------------------------------------


def test_build_result_row_includes_provenance(predictions) -> None:
    """RULE 5: every run records how it was produced."""
    y_true, scores = predictions
    metrics = compute_metrics(y_true, (scores > 0.5).astype(int), scores)
    row = build_result_row("baseline", "nsl-kdd", 42, metrics, 12.5)

    for field in (
        "timestamp", "experiment", "dataset", "seed", "hardware",
        "tabpfn_version", "checkpoint", "torch_version", "python_version",
        "git_commit", "runtime_seconds",
    ):
        assert field in row, f"{field} missing from result row"
    assert row["seed"] == 42
    assert row["f1_score"] == metrics["f1_score"]
    assert row["checkpoint"] != "auto"


def test_write_and_load_round_trip(tmp_path: Path, predictions) -> None:
    """Rows written to CSV read back with the same values."""
    y_true, scores = predictions
    metrics = compute_metrics(y_true, (scores > 0.5).astype(int), scores)
    row = build_result_row("unittest", "nsl-kdd", 7, metrics, 1.0)

    path = write_results(row, "unittest", seed=7, directory=tmp_path)
    assert path.exists() and path.suffix == ".csv"

    loaded = load_results("unittest", directory=tmp_path)
    assert len(loaded) == 1
    assert int(loaded[0]["seed"]) == 7
    assert float(loaded[0]["f1_score"]) == pytest.approx(metrics["f1_score"])


def test_write_results_handles_rows_with_differing_keys(tmp_path: Path) -> None:
    """Ablation arms carry different extras; the union must serialise."""
    rows = [
        {"experiment": "a", "seed": 1, "f1_score": 0.5},
        {"experiment": "a", "seed": 2, "f1_score": 0.6, "n_chunks": 13},
    ]
    write_results(rows, "mixed", directory=tmp_path)
    loaded = load_results("mixed", directory=tmp_path)
    assert len(loaded) == 2
    assert "n_chunks" in loaded[0]


def test_write_results_rejects_empty(tmp_path: Path) -> None:
    """Writing nothing is a caller error, not an empty file."""
    with pytest.raises(ValueError, match="no result rows"):
        write_results([], "empty", directory=tmp_path)


def test_load_results_on_empty_directory(tmp_path: Path) -> None:
    """A missing experiment returns an empty list, not an error."""
    assert load_results("nothing", directory=tmp_path) == []


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def test_confusion_matrix_png_is_written(tmp_path: Path) -> None:
    """The heatmap renders and is a non-trivial PNG."""
    path = plot_confusion_matrix(
        [[1995, 159], [1115, 1731]], path=tmp_path / "cm.png"
    )
    assert path.exists()
    assert path.stat().st_size > 1000


def test_roc_curve_png_is_written(tmp_path: Path, predictions) -> None:
    """The ROC curve renders from 1-D scores."""
    y_true, scores = predictions
    path = plot_roc_curve(y_true, scores, path=tmp_path / "roc.png")
    assert path.exists() and path.stat().st_size > 1000


def test_pr_curve_png_is_written(tmp_path: Path, predictions) -> None:
    """The PR curve renders and marks the 0.5 operating point."""
    y_true, scores = predictions
    path = plot_precision_recall_curve(y_true, scores, path=tmp_path / "pr.png")
    assert path.exists() and path.stat().st_size > 1000


def test_plots_accept_2d_predict_proba(tmp_path: Path, predictions) -> None:
    """A (n, 2) probability matrix works as well as 1-D scores."""
    y_true, scores = predictions
    proba = np.column_stack([1 - scores, scores])
    assert plot_roc_curve(y_true, proba, path=tmp_path / "roc2.png").exists()
    assert plot_precision_recall_curve(
        y_true, proba, path=tmp_path / "pr2.png"
    ).exists()


def test_plot_all_writes_three_figures(tmp_path: Path, predictions) -> None:
    """plot_all produces the full figure set for one run."""
    y_true, scores = predictions
    metrics = compute_metrics(y_true, (scores > 0.5).astype(int), scores)
    paths = plot_all(
        y_true,
        scores,
        metrics["confusion_matrix"],
        prefix="test_run",
        directory=tmp_path,
    )
    assert set(paths) == {"confusion_matrix", "roc_curve", "pr_curve"}
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 1000


def test_degenerate_confusion_matrix_does_not_divide_by_zero(
    tmp_path: Path,
) -> None:
    """An all-zero row must not produce NaN percentages."""
    path = plot_confusion_matrix([[0, 0], [5, 5]], path=tmp_path / "deg.png")
    assert path.exists()
