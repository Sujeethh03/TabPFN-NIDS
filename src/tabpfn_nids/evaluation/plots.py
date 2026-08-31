"""Figures for the report (Build Plan step 5.4).

Three figures, each saved as a PNG into ``reports/figures/``: a confusion
matrix heatmap, an ROC curve and a precision-recall curve.

The PR curve is included deliberately alongside ROC. NSL-KDD's test split is
57% attack, and the baseline runs show ROC-AUC around 0.955 while F1 sits
near 0.754 -- the model ranks attacks well but the default 0.5 threshold
places the operating point poorly. A PR curve makes that gap visible in a way
ROC alone does not.

Matplotlib's Agg backend is selected on import so figures render in headless
runs without a display.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics import confusion_matrix as sk_confusion_matrix  # noqa: E402

from tabpfn_nids import config  # noqa: E402

logger = logging.getLogger(__name__)

CLASS_LABELS: tuple[str, str] = ("normal", "attack")
FIGURE_DPI: int = 200

# Metrics shown, in this order, by plot_metrics_comparison.
COMPARISON_METRICS: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
)


def _resolve(path: str | Path | None, default_name: str) -> Path:
    """Resolve an output path, defaulting into reports/figures/.

    Args:
        path: Explicit destination, or None.
        default_name: Filename used when path is None.

    Returns:
        The resolved path, with parent directories created.
    """
    target = Path(path) if path is not None else config.FIGURES_DIR / default_name
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _positive_scores(y_proba: np.ndarray) -> np.ndarray:
    """Return the positive-class score column.

    Args:
        y_proba: 1-D positive-class scores, or a 2-D predict_proba output.

    Returns:
        A 1-D array of positive-class probabilities.
    """
    y_proba = np.asarray(y_proba)
    return y_proba if y_proba.ndim == 1 else y_proba[:, 1]


def plot_confusion_matrix(
    y_true: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    output_path: str | Path | None = None,
    title: str = "Confusion matrix",
    confusion: np.ndarray | list[list[int]] | None = None,
) -> Path:
    """Save a Seaborn confusion-matrix heatmap with counts and row shares.

    Each cell shows the raw count and its share of the true class, so recall
    per class is readable straight off the diagonal -- which is the number
    that matters on NSL-KDD, where the model's weakness is recall rather than
    precision.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        output_path: Destination PNG; defaults to
            reports/figures/confusion_matrix.png.
        title: Figure title.
        confusion: A precomputed 2x2 matrix ``[[TN, FP], [FN, TP]]``, used in
            place of ``y_true``/``y_pred`` when the matrix is already known.

    Returns:
        The path written.

    Raises:
        ValueError: If neither the label arrays nor a matrix are supplied.
    """
    if confusion is not None:
        matrix = np.asarray(confusion, dtype=float)
    elif y_true is not None and y_pred is not None:
        # labels=[0, 1] keeps the matrix 2x2 even if a split holds one class.
        matrix = sk_confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
    else:
        raise ValueError(
            "plot_confusion_matrix needs either (y_true, y_pred) or confusion="
        )

    target = _resolve(output_path, "confusion_matrix.png")

    row_totals = matrix.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1  # avoid 0/0 on a degenerate matrix
    shares = 100 * matrix / row_totals
    annotations = np.array(
        [
            [f"{int(matrix[i, j]):,}\n({shares[i, j]:.1f}%)" for j in range(2)]
            for i in range(2)
        ]
    )

    figure, axes = plt.subplots(figsize=(5.4, 4.5))
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="white",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        annot_kws={"fontsize": 11},
        ax=axes,
    )
    axes.set_xlabel("Predicted")
    axes.set_ylabel("Actual")
    axes.set_title(title)
    figure.tight_layout()
    figure.savefig(target, dpi=FIGURE_DPI)
    plt.close(figure)

    logger.info("Confusion matrix saved to %s", target)
    return target


def plot_metrics_comparison(
    results: dict[str, dict[str, float]],
    output_path: str | Path | None = None,
    title: str = "Metric comparison across experiments",
    errors: dict[str, dict[str, float]] | None = None,
) -> Path:
    """Save a grouped bar chart comparing several runs across metrics.

    Args:
        results: Mapping of run label to a metric dict, e.g.
            ``{"baseline": {"f1_score": 0.75, ...}, "enhanced": {...}}``.
        output_path: Destination PNG; defaults to
            reports/figures/metrics_comparison.png.
        title: Figure title.
        errors: Optional matching mapping of standard deviations, drawn as
            error bars. Without them a reader cannot tell a real difference
            from seed-to-seed noise.

    Returns:
        The path written.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("plot_metrics_comparison needs at least one run")

    target = _resolve(output_path, "metrics_comparison.png")
    metrics = [m for m in COMPARISON_METRICS
               if any(m in run for run in results.values())]
    labels = list(results)

    positions = np.arange(len(metrics))
    bar_width = min(0.8 / len(labels), 0.35)
    palette = sns.color_palette("colorblind", n_colors=len(labels))

    figure, axes = plt.subplots(figsize=(1.9 * len(metrics) + 2, 4.8))
    for index, label in enumerate(labels):
        values = [results[label].get(metric, np.nan) for metric in metrics]
        deviations = (
            [errors.get(label, {}).get(metric, 0.0) for metric in metrics]
            if errors
            else None
        )
        offset = (index - (len(labels) - 1) / 2) * bar_width
        bars = axes.bar(
            positions + offset,
            values,
            bar_width,
            label=label,
            color=palette[index],
            yerr=deviations,
            capsize=3,
        )
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                axes.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.015,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    axes.set_xticks(positions, [m.replace("_", " ") for m in metrics])
    axes.set_ylim(0, 1.12)
    axes.set_ylabel("Score")
    axes.set_title(title)
    axes.legend(loc="lower right", framealpha=0.9)
    axes.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(target, dpi=FIGURE_DPI)
    plt.close(figure)

    logger.info("Metric comparison saved to %s", target)
    return target


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "ROC curve",
    path: str | Path | None = None,
) -> Path:
    """Save an ROC curve with its AUC.

    Args:
        y_true: Ground-truth binary labels.
        y_proba: Positive-class scores, 1-D or 2-D.
        title: Figure title.
        path: Destination PNG; defaults to reports/figures/roc_curve.png.

    Returns:
        The path written.
    """
    scores = _positive_scores(y_proba)
    target = _resolve(path, "roc_curve.png")

    fpr, tpr, _ = roc_curve(y_true, scores)
    area = roc_auc_score(y_true, scores)

    figure, axes = plt.subplots(figsize=(5.2, 4.6))
    axes.plot(fpr, tpr, lw=2, label=f"TabPFN (AUC = {area:.4f})")
    axes.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1.02)
    axes.set_xlabel("False positive rate")
    axes.set_ylabel("True positive rate")
    axes.set_title(title)
    axes.legend(loc="lower right")
    axes.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(target, dpi=FIGURE_DPI)
    plt.close(figure)

    logger.info("ROC curve saved to %s (AUC %.4f)", target, area)
    return target


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "Precision-recall curve",
    path: str | Path | None = None,
) -> Path:
    """Save a precision-recall curve with average precision.

    The operating point at threshold 0.5 is marked, because on NSL-KDD the
    gap between that point and the rest of the curve is the finding.

    Args:
        y_true: Ground-truth binary labels.
        y_proba: Positive-class scores, 1-D or 2-D.
        title: Figure title.
        path: Destination PNG; defaults to reports/figures/pr_curve.png.

    Returns:
        The path written.
    """
    scores = _positive_scores(y_proba)
    y_true = np.asarray(y_true)
    target = _resolve(path, "pr_curve.png")

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    average_precision = average_precision_score(y_true, scores)
    pr_auc = auc(recall, precision)
    baseline = float(np.mean(y_true))

    figure, axes = plt.subplots(figsize=(5.2, 4.6))
    axes.plot(recall, precision, lw=2, label=f"AP = {average_precision:.4f}")
    axes.axhline(
        baseline,
        color="grey",
        ls="--",
        lw=1,
        label=f"Always-attack ({baseline:.3f})",
    )

    # Mark the default decision threshold. thresholds is one shorter than
    # precision/recall, so index into it before offsetting.
    if len(thresholds):
        index = int(np.argmin(np.abs(thresholds - 0.5)))
        axes.plot(
            recall[index],
            precision[index],
            "ro",
            ms=7,
            label="threshold = 0.5",
        )

    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1.02)
    axes.set_xlabel("Recall")
    axes.set_ylabel("Precision")
    axes.set_title(f"{title}  (PR-AUC {pr_auc:.4f})")
    axes.legend(loc="lower left")
    axes.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(target, dpi=FIGURE_DPI)
    plt.close(figure)

    logger.info("PR curve saved to %s (AP %.4f)", target, average_precision)
    return target


def plot_all(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    confusion: np.ndarray | list[list[int]] | None = None,
    prefix: str = "run",
    directory: Path | None = None,
    y_pred: np.ndarray | None = None,
) -> dict[str, Path]:
    """Generate all three figures for one experiment run.

    Args:
        y_true: Ground-truth binary labels.
        y_proba: Positive-class scores, 1-D or 2-D.
        confusion: A precomputed 2x2 confusion matrix, if available.
        prefix: Filename prefix, e.g. "baseline_seed42".
        directory: Destination directory; defaults to reports/figures/.
        y_pred: Predicted labels, used when no matrix is supplied.

    Returns:
        A mapping of figure name to the path written.
    """
    target = directory or config.FIGURES_DIR
    return {
        "confusion_matrix": plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            confusion=confusion,
            title=f"Confusion matrix — {prefix}",
            output_path=target / f"{prefix}_confusion_matrix.png",
        ),
        "roc_curve": plot_roc_curve(
            y_true,
            y_proba,
            title=f"ROC curve — {prefix}",
            path=target / f"{prefix}_roc_curve.png",
        ),
        "pr_curve": plot_precision_recall_curve(
            y_true,
            y_proba,
            title=f"Precision-recall — {prefix}",
            path=target / f"{prefix}_pr_curve.png",
        ),
    }
