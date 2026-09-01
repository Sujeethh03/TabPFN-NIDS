"""Metrics, statistical comparison, result reporting and figures."""

from __future__ import annotations

from tabpfn_nids.evaluation.metrics import compute_metrics, format_metrics
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

__all__ = [
    "build_result_row",
    "compute_metrics",
    "format_metrics",
    "load_results",
    "minimum_reachable_p",
    "plot_all",
    "plot_confusion_matrix",
    "plot_precision_recall_curve",
    "plot_roc_curve",
    "summarize_paired",
    "wilcoxon_test",
    "write_results",
]
