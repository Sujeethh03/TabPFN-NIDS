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
from sklearn.metrics import (  # noqa: E402
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from tabpfn_nids import config  # noqa: E402

logger = logging.getLogger(__name__)

CLASS_LABELS: tuple[str, str] = ("normal", "attack")
FIGURE_DPI: int = 150


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
    confusion: np.ndarray | list[list[int]],
    title: str = "Confusion matrix",
    path: str | Path | None = None,
) -> Path:
    """Save a confusion-matrix heatmap annotated with counts and percentages.

    Args:
        confusion: A 2x2 matrix ordered ``[[TN, FP], [FN, TP]]``.
        title: Figure title.
        path: Destination PNG; defaults to reports/figures/confusion_matrix.png.

    Returns:
        The path written.
    """
    matrix = np.asarray(confusion, dtype=float)
    target = _resolve(path, "confusion_matrix.png")

    figure, axes = plt.subplots(figsize=(5.2, 4.4))
    image = axes.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)

    row_totals = matrix.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1  # avoid 0/0 on a degenerate matrix

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            share = 100 * matrix[i, j] / row_totals[i, 0]
            # Pick a readable text colour against the cell's shade.
            colour = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            axes.text(
                j,
                i,
                f"{int(matrix[i, j]):,}\n({share:.1f}%)",
                ha="center",
                va="center",
                color=colour,
                fontsize=11,
            )

    axes.set_xticks([0, 1], CLASS_LABELS)
    axes.set_yticks([0, 1], CLASS_LABELS)
    axes.set_xlabel("Predicted")
    axes.set_ylabel("Actual")
    axes.set_title(title)
    figure.tight_layout()
    figure.savefig(target, dpi=FIGURE_DPI)
    plt.close(figure)

    logger.info("Confusion matrix saved to %s", target)
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
    confusion: np.ndarray | list[list[int]],
    prefix: str = "run",
    directory: Path | None = None,
) -> dict[str, Path]:
    """Generate all three figures for one experiment run.

    Args:
        y_true: Ground-truth binary labels.
        y_proba: Positive-class scores, 1-D or 2-D.
        confusion: The 2x2 confusion matrix.
        prefix: Filename prefix, e.g. "baseline_seed42".
        directory: Destination directory; defaults to reports/figures/.

    Returns:
        A mapping of figure name to the path written.
    """
    target = directory or config.FIGURES_DIR
    return {
        "confusion_matrix": plot_confusion_matrix(
            confusion,
            title=f"Confusion matrix — {prefix}",
            path=target / f"{prefix}_confusion_matrix.png",
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
