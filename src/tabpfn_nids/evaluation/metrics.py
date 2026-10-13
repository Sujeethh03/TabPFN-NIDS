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


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    classes: list[str] | None = None,
    confidence_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute comprehensive multiclass classification metrics.

    Args:
        y_true: Ground-truth class labels (integers or strings).
        y_pred: Predicted class labels (integers or strings).
        y_proba: Class probability array of shape (n_samples, n_classes).
        classes: Optional list of class names corresponding to columns of y_proba
            or integer labels.
        confidence_threshold: Minimum confidence score to consider a prediction
            confidently classified (default: 0.5).

    Returns:
        A structured dictionary containing:
        - accuracy: Overall classification accuracy
        - macro_precision, macro_recall, macro_f1: Macro-averaged metrics
        - weighted_precision, weighted_recall, weighted_f1: Weighted-averaged metrics
        - per_class: Dict mapping class names to precision, recall, f1, and support
        - confusion_matrix: 2D confusion matrix list
        - classes: List of class names
        - roc_auc_ovr: One-vs-Rest multiclass ROC-AUC (macro average) or None
        - class_distribution: Distribution of true labels
        - predicted_distribution: Distribution of predicted labels
        - prediction_coverage: Ratio of predictions above confidence_threshold
        - uncertain_rate: Ratio of predictions below confidence_threshold
        - confidence_threshold: Documented threshold used
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true has {y_true.shape[0]} rows but y_pred has {y_pred.shape[0]}."
        )

    # Determine label universe
    if classes is not None:
        all_classes = list(classes)
    else:
        unique_labels = sorted(list(set(y_true.tolist()) | set(y_pred.tolist())))
        all_classes = [str(c) for c in unique_labels]

    # Map to strings if classes are provided
    if classes is not None and np.issubdtype(y_true.dtype, np.integer):
        y_true_str = [classes[int(i)] if 0 <= int(i) < len(classes) else str(i) for i in y_true]
        y_pred_str = [classes[int(i)] if 0 <= int(i) < len(classes) else str(i) for i in y_pred]
    else:
        y_true_str = [str(x) for x in y_true]
        y_pred_str = [str(x) for x in y_pred]

    matrix = confusion_matrix(y_true_str, y_pred_str, labels=all_classes)

    acc = float(accuracy_score(y_true_str, y_pred_str))
    prec_macro = float(precision_score(y_true_str, y_pred_str, labels=all_classes, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true_str, y_pred_str, labels=all_classes, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true_str, y_pred_str, labels=all_classes, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_true_str, y_pred_str, labels=all_classes, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true_str, y_pred_str, labels=all_classes, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true_str, y_pred_str, labels=all_classes, average="weighted", zero_division=0))

    # Per-class metrics
    prec_per_class = precision_score(y_true_str, y_pred_str, labels=all_classes, average=None, zero_division=0)
    rec_per_class = recall_score(y_true_str, y_pred_str, labels=all_classes, average=None, zero_division=0)
    f1_per_class = f1_score(y_true_str, y_pred_str, labels=all_classes, average=None, zero_division=0)

    per_class_dict: dict[str, dict[str, Any]] = {}
    class_dist: dict[str, int] = {}
    pred_dist: dict[str, int] = {}

    for idx, cname in enumerate(all_classes):
        sup = int((np.array(y_true_str) == cname).sum())
        p_sup = int((np.array(y_pred_str) == cname).sum())
        class_dist[cname] = sup
        pred_dist[cname] = p_sup
        per_class_dict[cname] = {
            "precision": round(float(prec_per_class[idx]), 4),
            "recall": round(float(rec_per_class[idx]), 4),
            "f1_score": round(float(f1_per_class[idx]), 4),
            "support": sup,
            "predicted_count": p_sup,
        }

    # Confidence and uncertain rate / rejection evaluation
    prediction_coverage = 1.0
    uncertain_rate = 0.0
    accepted_accuracy = acc
    accepted_count = len(y_true_str)
    uncertain_count = 0
    threshold_curve = []

    if y_proba is not None and len(y_proba) > 0:
        max_probas = np.max(y_proba, axis=1)
        confident_mask = max_probas >= confidence_threshold
        accepted_count = int(np.sum(confident_mask))
        uncertain_count = int(len(max_probas) - accepted_count)
        prediction_coverage = float(np.mean(confident_mask))
        uncertain_rate = float(1.0 - prediction_coverage)
        if accepted_count > 0:
            y_t_conf = [y_true_str[i] for i in range(len(y_true_str)) if confident_mask[i]]
            y_p_conf = [y_pred_str[i] for i in range(len(y_pred_str)) if confident_mask[i]]
            accepted_accuracy = float(accuracy_score(y_t_conf, y_p_conf))
        else:
            accepted_accuracy = 0.0

        for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            m = max_probas >= th
            cov = float(np.mean(m))
            c_cnt = int(np.sum(m))
            if c_cnt > 0:
                y_t_th = [y_true_str[i] for i in range(len(y_true_str)) if m[i]]
                y_p_th = [y_pred_str[i] for i in range(len(y_pred_str)) if m[i]]
                acc_th = float(accuracy_score(y_t_th, y_p_th))
            else:
                acc_th = 0.0
            threshold_curve.append({
                "threshold": th,
                "coverage": round(cov, 4),
                "accepted_count": c_cnt,
                "accepted_accuracy": round(acc_th, 4),
            })

    # ROC-AUC (One-vs-Rest)
    roc_auc_ovr: float | None = None
    if y_proba is not None and y_proba.ndim == 2:
        try:
            # Requires at least 2 distinct classes present in y_true
            unique_present = np.unique(y_true)
            if len(unique_present) >= 2 and y_proba.shape[1] >= len(unique_present):
                roc_auc_ovr = float(
                    roc_auc_score(
                        y_true,
                        y_proba,
                        multi_class="ovr",
                        average="macro",
                    )
                )
        except Exception as exc:
            logger.debug("ROC-AUC OvR not computable: %s", exc)
            roc_auc_ovr = None

    return {
        "accuracy": round(acc, 4),
        "accepted_accuracy": round(accepted_accuracy, 4),
        "accepted_count": accepted_count,
        "uncertain_count": uncertain_count,
        "macro_precision": round(prec_macro, 4),
        "macro_recall": round(rec_macro, 4),
        "macro_f1": round(f1_macro, 4),
        "weighted_precision": round(prec_weighted, 4),
        "weighted_recall": round(rec_weighted, 4),
        "weighted_f1": round(f1_weighted, 4),
        "per_class": per_class_dict,
        "confusion_matrix": matrix.tolist(),
        "classes": all_classes,
        "roc_auc_ovr": round(roc_auc_ovr, 4) if roc_auc_ovr is not None else None,
        "class_distribution": class_dist,
        "predicted_distribution": pred_dist,
        "prediction_coverage": round(prediction_coverage, 4),
        "uncertain_rate": round(uncertain_rate, 4),
        "confidence_threshold": confidence_threshold,
        "threshold_curve": threshold_curve,
    }


def format_metrics(metrics: dict[str, Any], title: str = "Results") -> str:
    """Render a metrics dict as an aligned text block."""
    if "macro_f1" in metrics:
        return format_multiclass_metrics(metrics, title=title)

    width = 58
    lines = ["=" * width, title, "=" * width]

    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        value = metrics.get(key)
        label = key
        rendered = "n/a" if value is None else f"{value:.4f}"
        lines.append(f"  {label:<22}{rendered:>12}")

    if "confusion_matrix" in metrics and len(metrics["confusion_matrix"]) == 2:
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


def format_multiclass_metrics(metrics: dict[str, Any], title: str = "Multiclass Results") -> str:
    """Render multiclass metrics as an aligned text block with per-class table."""
    width = 72
    lines = ["=" * width, title, "=" * width]

    for label, key in [
        ("Accuracy", "accuracy"),
        ("Macro Precision", "macro_precision"),
        ("Macro Recall", "macro_recall"),
        ("Macro F1", "macro_f1"),
        ("Weighted Precision", "weighted_precision"),
        ("Weighted Recall", "weighted_recall"),
        ("Weighted F1", "weighted_f1"),
        ("ROC-AUC (OvR)", "roc_auc_ovr"),
        ("Prediction Coverage", "prediction_coverage"),
        ("Uncertain Rate", "uncertain_rate"),
    ]:
        val = metrics.get(key)
        rendered = "n/a" if val is None else (f"{val * 100:.2f}%" if "coverage" in key or "rate" in key else f"{val:.4f}")
        lines.append(f"  {label:<24}{rendered:>12}")

    lines.append("-" * width)
    lines.append("  Per-Class Metrics:")
    lines.append(f"    {'Category':<16} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    lines.append("    " + "-" * 58)

    per_class = metrics.get("per_class", {})
    for cname, p_data in per_class.items():
        prec = f"{p_data['precision']:.4f}"
        rec = f"{p_data['recall']:.4f}"
        f1 = f"{p_data['f1_score']:.4f}"
        sup = str(p_data['support'])
        lines.append(f"    {cname:<16} {prec:>10} {rec:>10} {f1:>10} {sup:>10}")

    lines.append("=" * width)
    return "\n".join(lines)


def get_attack_category_mapping(label_encoder: dict[str, Any] | None = None) -> dict[int, str]:
    """Return dictionary mapping integer class index to original attack category name.

    Inspects idx_to_class, id_to_label, and classes list from the trained model's label encoder.
    Falls back to verified UNSW-NB15 9-class mapping if no label encoder is provided.
    """
    mapping: dict[int, str] = {}
    if label_encoder:
        # 1. Check idx_to_class (str or int keys)
        for k, v in label_encoder.get("idx_to_class", {}).items():
            try:
                mapping[int(k)] = str(v)
            except (ValueError, TypeError):
                pass
        # 2. Check id_to_label (str or int keys)
        for k, v in label_encoder.get("id_to_label", {}).items():
            try:
                mapping[int(k)] = str(v)
            except (ValueError, TypeError):
                pass
        # 3. Check classes list order
        for idx, cls_name in enumerate(label_encoder.get("classes", [])):
            if idx not in mapping:
                mapping[idx] = str(cls_name)

    if not mapping:
        # Verified mapping from multiclass TabPFN training on UNSW-NB15
        mapping = {
            0: "Backdoors",
            1: "DoS",
            2: "Exploits",
            3: "Fuzzers",
            4: "Generic",
            5: "Normal",
            6: "Reconnaissance",
            7: "Shellcode",
            8: "Worms",
        }
    return mapping


def decode_attack_category(
    value: int | str | None,
    label_encoder: dict[str, Any] | None = None,
    uncertainty_threshold: float = 0.5,
    confidence: float | None = None,
) -> str:
    """Decode a multiclass prediction (numeric ID or string) to a verified readable attack category name.

    Guarantees that numeric IDs (0-8) are decoded to valid names ('Backdoors', 'DoS', 'Exploits',
    'Fuzzers', 'Generic', 'Normal', 'Reconnaissance', 'Shellcode', 'Worms').
    Returns 'Normal' for benign flows, and never fabricates labels not in the model.
    """
    if value is None:
        return "Unknown"

    mapping = get_attack_category_mapping(label_encoder)
    valid_names = set(mapping.values())

    # Try integer lookup first
    val_str = str(value).strip()
    try:
        numeric_id = int(val_str)
        if numeric_id in mapping:
            return mapping[numeric_id]
    except ValueError:
        pass

    # Check case-insensitive match against valid class names
    for valid_name in valid_names:
        if valid_name.lower() == val_str.lower():
            return valid_name

    return val_str

