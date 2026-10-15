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


def plot_metrics_radar(
    results: dict[str, dict[str, float]] | dict[str, float],
    output_path: str | Path | None = None,
    title: str = "Performance Metrics Radar Analysis",
    baseline_benchmark: dict[str, float] | None = None,
) -> Path:
    """Save an interactive radar / spider chart analyzing model evaluation metrics.

    Args:
        results: A dictionary of metric names to scores (e.g.
            ``{"accuracy": 0.788, "precision": 0.984, "recall": 0.639, ...}``)
            or a mapping of run labels to metric dicts.
        output_path: Destination PNG/SVG path; defaults to
            reports/figures/metrics_radar.png.
        title: Figure title.
        baseline_benchmark: Optional baseline dict to compare against, e.g.
            vanilla TabPFN baseline.

    Returns:
        The path written.
    """
    target = _resolve(output_path, "metrics_radar.png")

    # Normalize inputs to a mapping of label -> metrics
    series: dict[str, dict[str, float]] = {}
    if results and any(isinstance(v, dict) for v in results.values()):
        series = {str(k): dict(v) for k, v in results.items() if isinstance(v, dict)}
    elif results:
        series = {"TabPFN Model": dict(results)}  # type: ignore

    if baseline_benchmark:
        series["Baseline (Vanilla 10K)"] = baseline_benchmark

    metrics = [
        m for m in COMPARISON_METRICS
        if any(m in run or m.replace("_", "") in run for run in series.values())
    ]
    if not metrics:
        metrics = list(COMPARISON_METRICS)

    labels = [m.replace("_", " ").title() for m in metrics]
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle loop

    figure, axes = plt.subplots(figsize=(6.5, 6.0), subplot_kw={"polar": True})

    palette = sns.color_palette("muted", n_colors=max(len(series), 1))
    colors = ["#16A34A", "#2563EB", "#D97706", "#7C3AED", "#DC2626"]

    # Draw polygon for each series
    for idx, (name, metric_dict) in enumerate(series.items()):
        vals = [
            float(metric_dict.get(m, metric_dict.get(m.replace("_", ""), 0.0)))
            for m in metrics
        ]
        vals += vals[:1]

        c = colors[idx % len(colors)]
        is_primary = idx == 0
        axes.plot(
            angles,
            vals,
            color=c,
            linewidth=2.5 if is_primary else 1.8,
            linestyle="-" if is_primary else "--",
            label=f"{name}",
            marker="o" if is_primary else "s",
            markersize=5,
        )
        axes.fill(angles, vals, color=c, alpha=0.18 if is_primary else 0.08)

    # Add 95% production target ring
    target_line = [0.95] * (num_vars + 1)
    axes.plot(
        angles,
        target_line,
        color="#9CA3AF",
        linewidth=1.0,
        linestyle=":",
        label="Production Target (95%)",
    )

    axes.set_theta_offset(np.pi / 2)
    axes.set_theta_direction(-1)
    axes.set_xticks(angles[:-1])
    axes.set_xticklabels(labels, fontsize=10, fontweight="bold")
    axes.set_ylim(0, 1.05)
    axes.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    axes.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8, color="#6B7280")
    axes.grid(color="#E5E7EB", linestyle="--", linewidth=0.8)

    axes.set_title(title, fontsize=13, fontweight="bold", pad=20)
    axes.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9, framealpha=0.9)

    figure.tight_layout()
    figure.savefig(target, dpi=FIGURE_DPI)
    plt.close(figure)

    logger.info("Metrics radar analysis saved to %s", target)
    return target


def plot_comprehensive_metrics_dashboard(
    metrics: dict[str, float],
    output_path: str | Path | None = None,
    title: str = "TabPFN-NIDS Comprehensive Metrics Analysis",
    confusion: np.ndarray | list[list[int]] | None = None,
    comparison_baseline: dict[str, float] | None = None,
) -> Path:
    """Save a multi-panel visual dashboard figure for metrics analysis.

    Panel 1: Horizontal performance bar chart with benchmarks and deltas.
    Panel 2: Polar radar analysis chart showing multi-metric balance.
    Panel 3 (if confusion provided): Confusion matrix heatmap with rates.

    Args:
        metrics: Dictionary of core metrics.
        output_path: Destination image path.
        title: Figure title.
        confusion: Optional 2x2 confusion matrix [[TN, FP], [FN, TP]].
        comparison_baseline: Optional baseline metrics to compare against.

    Returns:
        The path written.
    """
    target = _resolve(output_path, "metrics_analysis_dashboard.png")
    has_cm = confusion is not None

    cols = 3 if has_cm else 2
    figure = plt.figure(figsize=(5.5 * cols, 5.2))

    # Panel 1: Horizontal Bar Chart
    ax1 = figure.add_subplot(1, cols, 1)
    metric_keys = [m for m in COMPARISON_METRICS if m in metrics or m.replace("_", "") in metrics]
    if not metric_keys:
        metric_keys = list(COMPARISON_METRICS)

    labels = [m.replace("_", " ").title() for m in metric_keys]
    values = [float(metrics.get(m, metrics.get(m.replace("_", ""), 0.0))) for m in metric_keys]
    y_pos = np.arange(len(labels))

    colors = ["#16A34A" if v >= 0.75 else "#D97706" for v in values]
    bars = ax1.barh(y_pos, values, height=0.55, color=colors, alpha=0.85, edgecolor="#15803D")

    # Annotate bar values
    for bar, val in zip(bars, values):
        ax1.text(
            val + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val * 100:.1f}%",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#1F2937",
        )

    # If baseline is provided, draw comparison markers
    if comparison_baseline:
        base_vals = [
            float(comparison_baseline.get(m, comparison_baseline.get(m.replace("_", ""), 0.0)))
            for m in metric_keys
        ]
        ax1.scatter(base_vals, y_pos, color="#2563EB", s=60, marker="|", linewidth=2.5, label="Vanilla Baseline", zorder=3)
        ax1.legend(loc="lower right", fontsize=8)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=10, fontweight="bold")
    ax1.set_xlim(0, 1.15)
    ax1.set_xlabel("Performance Score", fontsize=10)
    ax1.set_title("Metric Evaluation Scores", fontsize=11, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    ax1.invert_yaxis()

    # Panel 2: Polar Radar Chart
    ax2 = figure.add_subplot(1, cols, 2, polar=True)
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    plot_vals = values + values[:1]

    ax2.plot(angles, plot_vals, color="#16A34A", linewidth=2.2, marker="o", label="Enhanced Model")
    ax2.fill(angles, plot_vals, color="#16A34A", alpha=0.22)

    if comparison_baseline:
        base_loop = base_vals + base_vals[:1]
        ax2.plot(angles, base_loop, color="#2563EB", linewidth=1.8, linestyle="--", label="Baseline")
        ax2.fill(angles, base_loop, color="#2563EB", alpha=0.08)

    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels([m.replace("_", " ").title() for m in metric_keys], fontsize=8, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Balance & Trade-Off Radar", fontsize=11, fontweight="bold", pad=14)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.15, 1.08), fontsize=8)

    # Panel 3: Confusion Matrix Heatmap (if provided)
    if has_cm and confusion is not None:
        ax3 = figure.add_subplot(1, cols, 3)
        matrix = np.asarray(confusion, dtype=float)
        row_totals = matrix.sum(axis=1, keepdims=True)
        row_totals[row_totals == 0] = 1
        shares = 100 * matrix / row_totals
        annot = np.array([
            [f"{int(matrix[i, j]):,}\n({shares[i, j]:.1f}%)" for j in range(2)]
            for i in range(2)
        ])
        sns.heatmap(
            matrix,
            annot=annot,
            fmt="",
            cmap="Greens",
            cbar=False,
            square=True,
            linewidths=0.5,
            linecolor="white",
            xticklabels=CLASS_LABELS,
            yticklabels=CLASS_LABELS,
            annot_kws={"fontsize": 9, "fontweight": "bold"},
            ax=ax3,
        )
        ax3.set_xlabel("Predicted Label", fontsize=9)
        ax3.set_ylabel("True Ground Truth", fontsize=9)
        ax3.set_title("Confusion Heatmap", fontsize=11, fontweight="bold")

    figure.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    figure.tight_layout()
    figure.savefig(target, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)

    logger.info("Comprehensive metrics dashboard saved to %s", target)
    return target


def plot_prediction_distribution_donut(
    counts: dict[str, int] | tuple[int, int] | list[int] | np.ndarray,
    output_path: str | Path | None = None,
    title: str = "Prediction Distribution (Normal vs Attack)",
) -> Path:
    """Save a donut chart showing the number and percentage of normal and attack predictions.

    Args:
        counts: A dict e.g. ``{"normal": 9694, "attack": 12850}``,
            a tuple/list ``(normal_count, attack_count)``,
            or a 1D array of binary predictions (0=normal, 1=attack).
        output_path: Destination PNG/SVG path; defaults to
            reports/figures/prediction_distribution_donut.png.
        title: Figure title.

    Returns:
        The path written.
    """
    target = _resolve(output_path, "prediction_distribution_donut.png")

    normal_count = 0
    attack_count = 0

    if isinstance(counts, dict):
        normal_count = int(counts.get("normal", counts.get("normal_count", 0)))
        attack_count = int(counts.get("attack", counts.get("attack_count", 0)))
    elif isinstance(counts, (list, tuple)) and len(counts) == 2:
        normal_count = int(counts[0])
        attack_count = int(counts[1])
    else:
        arr = np.asarray(counts)
        normal_count = int(np.sum(arr == 0))
        attack_count = int(np.sum(arr == 1))

    total = normal_count + attack_count
    if total <= 0:
        total = 1

    norm_pct = 100 * normal_count / total
    atk_pct = 100 * attack_count / total

    figure, axes = plt.subplots(figsize=(5.2, 4.8))
    sizes = [normal_count, attack_count]
    colors = ["#16A34A", "#DC2626"]
    labels = [
        f"Normal: {normal_count:,}\n({norm_pct:.1f}%)",
        f"Attack: {attack_count:,}\n({atk_pct:.1f}%)",
    ]

    wedges, texts, autotexts = axes.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        pctdistance=0.75,
        startangle=90,
        textprops={"fontsize": 10, "fontweight": "bold"},
        wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 2},
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")

    axes.text(
        0,
        0,
        f"{total:,}\nTOTAL FLOWS",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#374151",
    )

    axes.set_title(title, fontsize=12, fontweight="bold", pad=15)
    figure.tight_layout()
    figure.savefig(target, dpi=FIGURE_DPI)
    plt.close(figure)

    logger.info("Prediction distribution donut saved to %s", target)
    return target

