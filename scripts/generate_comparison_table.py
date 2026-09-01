#!/usr/bin/env python3
"""Build the headline three-row comparison table.

Reads every results CSV in ``reports/`` and produces one table with the three
arms the project is built around:

    Vanilla TabPFN (10K subsample)       from scripts/run_baseline.py
    Enhanced (Chunked Ensemble)          from scripts/run_enhanced.py
    Enhanced + Feature Engineering       from scripts/run_feature_ablation.py

Where several seeds exist for an arm, metrics are reported as mean +/- std.
Where an arm has not been run, its row is retained and marked "not run" rather
than silently dropped -- a comparison table that hides its own gaps is worse
than one that shows them.

Also writes a bar chart of the metrics to reports/figures/.

Usage:
    python scripts/generate_comparison_table.py
    python scripts/generate_comparison_table.py --no-plot
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Any

from tabpfn_nids import config
from tabpfn_nids.evaluation.plots import plot_metrics_comparison
from tabpfn_nids.evaluation.reporter import load_results

METRICS = ("accuracy", "precision", "recall", "f1_score", "roc_auc")

# Display label -> (results tag, optional filter on the `arm` column).
# The ablation writes two rows to one file, distinguished by `arm`.
ARMS: tuple[tuple[str, str, str | None], ...] = (
    ("Vanilla TabPFN (10K subsample)", "baseline", None),
    ("Enhanced (Chunked Ensemble)", "enhanced", None),
    ("Enhanced + Feature Engineering", "feature_ablation", "engineered"),
)

SETTINGS = (
    ("n_runs", "seeds"),
    ("context_rows", "context rows"),
    ("test_rows", "test rows"),
    ("n_features", "features"),
    ("n_chunks", "chunks"),
    ("n_estimators", "n_estimators"),
    ("runtime_seconds_mean", "runtime (s)"),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None,
                        help="Directory for the tables; defaults to reports/.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip the metric comparison bar chart.")
    return parser.parse_args()


def collect(tag: str, arm: str | None) -> list[dict[str, str]]:
    """Load result rows for one arm.

    Args:
        tag: Filename prefix of the results CSVs.
        arm: Optional value of the `arm` column to filter on, used to pick one
            side of an ablation file.

    Returns:
        Matching result rows.
    """
    rows = load_results(tag)
    if arm is not None:
        rows = [r for r in rows if r.get("arm") == arm]
    return rows


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Reduce an arm's rows to means, standard deviations and settings.

    Args:
        rows: Result rows for one arm.

    Returns:
        A summary dict; ``available`` is False when the arm has no runs.
    """
    if not rows:
        return {"available": False, "n_runs": 0}

    summary: dict[str, Any] = {"available": True, "n_runs": len(rows)}
    summary["seeds"] = ",".join(
        sorted({r.get("seed", "?") for r in rows}, key=str)
    )

    for field, _ in SETTINGS:
        if field in ("n_runs", "runtime_seconds_mean"):
            continue
        values = {r[field] for r in rows if r.get(field)}
        summary[field] = (
            values.pop() if len(values) == 1 else ("mixed" if values else "-")
        )

    for metric in METRICS:
        values = [
            float(r[metric])
            for r in rows
            if r.get(metric) not in (None, "", "None")
        ]
        summary[f"{metric}_mean"] = statistics.fmean(values) if values else None
        summary[f"{metric}_std"] = (
            statistics.stdev(values) if len(values) > 1 else 0.0
        )

    runtimes = [
        float(r["runtime_seconds"]) for r in rows if r.get("runtime_seconds")
    ]
    summary["runtime_seconds_mean"] = (
        statistics.fmean(runtimes) if runtimes else None
    )
    return summary


def cell(summary: dict[str, Any], metric: str) -> str:
    """Format one metric cell as ``mean ± std``.

    Args:
        summary: An arm summary.
        metric: Metric name.

    Returns:
        The formatted cell, or "not run" / "n/a".
    """
    if not summary.get("available"):
        return "not run"
    mean = summary.get(f"{metric}_mean")
    if mean is None:
        return "n/a"
    return f"{mean:.4f} ± {summary.get(f'{metric}_std', 0.0):.4f}"


def render_markdown(summaries: dict[str, dict[str, Any]]) -> str:
    """Render the comparison as Markdown.

    Args:
        summaries: Mapping of display label to arm summary.

    Returns:
        Markdown source.
    """
    labels = list(summaries)
    lines = [
        "# Comparison table",
        "",
        "TabPFN v2 on NSL-KDD, binary classification (0 = normal, 1 = attack).",
        "Values are mean ± standard deviation across seeds; std is 0.0000 where",
        "only one seed was run.",
        "",
        "| Metric | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    for metric in METRICS:
        cells = [cell(summaries[label], metric) for label in labels]
        lines.append(f"| **{metric}** | " + " | ".join(cells) + " |")

    lines += ["", "## Run settings", "",
              "| Setting | " + " | ".join(labels) + " |",
              "|---|" + "---|" * len(labels)]
    for field, display in SETTINGS:
        cells = []
        for label in labels:
            summary = summaries[label]
            if not summary.get("available"):
                cells.append("—")
                continue
            value = summary.get(field)
            if value is None:
                cells.append("—")
            elif isinstance(value, float):
                cells.append(f"{value:.1f}")
            else:
                cells.append(str(value))
        lines.append(f"| {display} | " + " | ".join(cells) + " |")

    missing = [label for label, s in summaries.items() if not s.get("available")]
    if missing:
        lines += [
            "",
            "## Not yet run",
            "",
            "These arms have no results in `reports/` and are shown as "
            "*not run* rather than omitted:",
            "",
        ]
        lines += [f"- {label}" for label in missing]

    baseline = summaries.get("Vanilla TabPFN (10K subsample)", {})
    std = baseline.get("f1_score_std")
    if std:
        lines += [
            "",
            "## Reading the deltas",
            "",
            f"The baseline's F1 standard deviation across seeds is **{std:.4f}** "
            f"({100 * std:.2f} pp). A difference between arms smaller than that "
            "is within seed-to-seed noise and should not be reported as an "
            "improvement.",
        ]
    return "\n".join(lines) + "\n"


def render_text(summaries: dict[str, dict[str, Any]]) -> str:
    """Render the comparison as an aligned console table.

    Args:
        summaries: Mapping of display label to arm summary.

    Returns:
        A printable table.
    """
    labels = list(summaries)
    col = 24
    width = 20 + col * len(labels)
    lines = ["=" * width, "COMPARISON TABLE", "=" * width]
    lines.append(f"  {'metric':<18}" + "".join(f"{l[:22]:>{col}}" for l in labels))
    lines.append("-" * width)
    for metric in METRICS:
        cells = "".join(f"{cell(summaries[l], metric):>{col}}" for l in labels)
        lines.append(f"  {metric:<18}{cells}")
    lines.append("-" * width)
    for field, display in SETTINGS:
        cells = ""
        for label in labels:
            summary = summaries[label]
            value = summary.get(field) if summary.get("available") else None
            if value is None:
                cells += f"{'-':>{col}}"
            elif isinstance(value, float):
                cells += f"{value:>{col}.1f}"
            else:
                cells += f"{str(value):>{col}}"
        lines.append(f"  {display:<18}{cells}")
    lines.append("=" * width)
    return "\n".join(lines)


def main() -> int:
    """Build the comparison table, write both formats and the bar chart."""
    args = parse_args()
    config.ensure_dirs()

    summaries: dict[str, dict[str, Any]] = {
        label: summarize(collect(tag, arm)) for label, tag, arm in ARMS
    }

    if not any(s.get("available") for s in summaries.values()):
        print(f"No results found in {config.REPORTS_DIR}", file=sys.stderr)
        return 1

    print(render_text(summaries))

    out_dir = Path(args.out_dir) if args.out_dir else config.REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / "comparison_table.md"
    md_path.write_text(render_markdown(summaries), encoding="utf-8")

    csv_path = out_dir / "comparison_table.csv"
    fields = ["experiment", "available", "n_runs", "seeds",
              *[f for f, _ in SETTINGS if f != "n_runs"],
              *[f"{m}_{s}" for m in METRICS for s in ("mean", "std")]]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for label, summary in summaries.items():
            writer.writerow({"experiment": label, **summary})

    written = [csv_path, md_path]

    if not args.no_plot:
        available = {
            label: {
                m: s[f"{m}_mean"]
                for m in METRICS
                if s.get(f"{m}_mean") is not None
            }
            for label, s in summaries.items()
            if s.get("available")
        }
        errors = {
            label: {
                m: summaries[label].get(f"{m}_std", 0.0)
                for m in METRICS
                if summaries[label].get(f"{m}_mean") is not None
            }
            for label in available
        }
        written.append(
            plot_metrics_comparison(
                available,
                output_path=config.FIGURES_DIR / "metrics_comparison.png",
                title="TabPFN on NSL-KDD — metric comparison",
                errors=errors,
            )
        )

    print("\nWritten:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
