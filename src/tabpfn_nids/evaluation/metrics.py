"""Classification metrics for the NIDS experiments.

All metrics are computed with the attack class (label 1) as the positive
class, which is the convention throughout this project: precision and recall
answer "of the flows we flagged as attacks, how many were", and "of the real
attacks, how many did we catch".
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

POSITIVE_LABEL: int = 1


def _positive_class_scores(y_proba: np.ndarray) -> np.ndarray:
    """Extract the positive-class score column from a probability array.

    Args:
        y_proba: Either a 1-D array already holding P(class=1), or a 2-D
            ``(n_rows, n_classes)`` array as returned by ``predict_proba``.

    Returns:
        A 1-D array of positive-class probabilities.

    Raises:
        ValueError: If the array is neither 1-D nor 2-D with 2 columns.
    """
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 1:
        return y_proba
    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        return y_proba[:, POSITIVE_LABEL]
    raise ValueError(
        f"y_proba has shape {y_proba.shape}; expected a 1-D array of "
        "positive-class probabilities or a 2-D (n_rows, 2) array."
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute the standard binary classification metrics.

    Args:
        y_true: Ground-truth binary labels, 0 for normal and 1 for attack.
        y_pred: Predicted binary labels.
        y_proba: Positive-class probabilities, either 1-D or a 2-D
            ``predict_proba`` output. Required for ROC-AUC; when omitted,
            ``roc_auc`` is None rather than silently substituted.

    Returns:
        A flat dict suitable for direct CSV serialisation:

        - ``accuracy``, ``precision``, ``recall``, ``f1_score``: floats
        - ``roc_auc``: float, or None if ``y_proba`` was not supplied
        - ``confusion_matrix``: 2x2 nested list, ``[[TN, FP], [FN, TP]]``
        - ``true_negatives`` / ``false_positives`` / ``false_negatives`` /
          ``true_positives``: ints, the same values as flat columns
        - ``support_normal`` / ``support_attack``: class counts

    Raises:
        ValueError: If y_true and y_pred have different lengths.

    Example:
        >>> compute_metrics([0, 1, 1], [0, 1, 0])["recall"]
        0.5
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true has {y_true.shape[0]} rows but y_pred has {y_pred.shape[0]}."
        )

    # labels=[0, 1] keeps the matrix 2x2 even when a split or a degenerate
    # model contains only one class, which would otherwise return a 1x1.
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "roc_auc": None,
        "confusion_matrix": matrix.tolist(),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "support_normal": int((y_true == 0).sum()),
        "support_attack": int((y_true == POSITIVE_LABEL).sum()),
    }

    if y_proba is not None:
        if len(np.unique(y_true)) < 2:
            logger.warning(
                "ROC-AUC is undefined when y_true contains a single class; "
                "reporting None."
            )
        else:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, _positive_class_scores(y_proba))
            )

    return metrics


def format_metrics(metrics: dict[str, Any], title: str = "Results") -> str:
    """Render a metrics dict as an aligned text block.

    Args:
        metrics: A dict as returned by ``compute_metrics``.
        title: Heading for the block.

    Returns:
        A printable multi-line string.
    """
    width = 58
    lines = ["=" * width, title, "=" * width]

    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        value = metrics.get(key)
        label = key
        rendered = "n/a" if value is None else f"{value:.4f}"
        lines.append(f"  {label:<22}{rendered:>12}")

    tn, fp = metrics["confusion_matrix"][0]
    fn, tp = metrics["confusion_matrix"][1]
    lines += [
        "-" * width,
        "  Confusion matrix",
        "                        predicted",
        "                     normal    attack",
        f"    actual normal  {tn:>8,}  {fp:>8,}",
        f"    actual attack  {fn:>8,}  {tp:>8,}",
        "=" * width,
    ]
    return "\n".join(lines)
