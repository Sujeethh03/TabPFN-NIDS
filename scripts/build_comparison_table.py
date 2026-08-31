#!/usr/bin/env python3
"""Combine every result CSV into one comparison table (Build Plan step 8.2).

Reads all result CSVs from reports/, groups them by experiment, and emits a
single table with mean +/- std across seeds for each arm. Writes both
``reports/tables/comparison_table.csv`` and a Markdown version for pasting
into the report.

Where an arm has three or fewer seeds the table reports mean +/- std rather
than a p-value, because the Wilcoxon test cannot reach significance at that
sample size (see tabpfn_nids.evaluation.significance).

Usage:
    python scripts/build_comparison_table.py
    python scripts/build_comparison_table.py --experiments baseline enhanced
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Any

from tabpfn_nids import config
from tabpfn_nids.evaluation.reporter import load_results

METRICS = ("accuracy", "precision", "recall", "f1_score", "roc_auc")

# Experiment tags that are exploratory rather than reportable results.
EXCLUDED_TAGS = ("probe", "sweep-n1", "sweep-n2", "sweep-n4")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Experiment tags to include; default is all found.")
    parser.add_argument("--out-dir", default=None,
                        help="Directory for the output tables.")
    return parser.parse_args()


def discover_experiments(exclude: tuple[str, ...] = EXCLUDED_TAGS) -> list[str]:
    """Find reportable experiment tags present in reports/.

    Args:
        exclude: Tags to skip.

    Returns:
        Sorted experiment tags.
    """
    tags: set[str] = set()
    for path in config.REPORTS_DIR.glob("*.csv"):
        tag = path.stem.split("_")[0]
        if tag not in exclude:
            tags.add(tag)
    return sorted(tags)


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Summarise one experiment's rows.

    Args:
        rows: Result rows for a single experiment arm.

    Returns:
        A dict of metric means and standard deviations plus run metadata.
    """
    summary: dict[str, Any] = {
        "n_runs": len(rows),
        "seeds": ",".join(sorted({r.get("seed", "?") for r in rows}, key=str)),
    }
    for field in ("context_rows", "test_rows", "n_features", "n_chunks",
                  "n_estimators"):
        values = {r[field] for r in rows if r.get(field)}
        if not values:
            summary[field] = "-"          # column absent for this experiment
        elif len(values) == 1:
            summary[field] = values.pop()
        else:
            summary[field] = "mixed"

    for metric in METRICS:
        values = [
            float(r[metric]) for r in rows if r.get(metric) not in (None, "", "None")
        ]
        if not values:
            summary[f"{metric}_mean"] = None
            summary[f"{metric}_std"] = None
            continue
        summary[f"{metric}_mean"] = statistics.fmean(values)
        summary[f"{metric}_std"] = (
            statistics.stdev(values) if len(values) > 1 else 0.0
        )

    runtimes = [
        float(r["runtime_seconds"]) for r in rows if r.get("runtime_seconds")
    ]
    summary["runtime_seconds_mean"] = statistics.fmean(runtimes) if runtimes else None
    return summary


def render_text(summaries: dict[str, dict[str, Any]]) -> str:
    """Render the comparison as an aligned text table.

    Args:
        summaries: Mapping of experiment tag to its summary.

    Returns:
        A printable table.
    """
    width = 22 + 16 * len(summaries)
    lines = ["=" * width, "COMPARISON TABLE", "=" * width]
    header = f"  {'metric':<20}" + "".join(f"{tag:>16}" for tag in summaries)
    lines += [header, "-" * width]

    for metric in METRICS:
        cells = ""
        for summary in summaries.values():
            mean = summary.get(f"{metric}_mean")
            std = summary.get(f"{metric}_std")
            cells += "             n/a" if mean is None else f"{mean:>10.4f}+-{std:<4.3f}"
        lines.append(f"  {metric:<20}{cells}")

    lines.append("-" * width)
    for field, label in (
        ("n_runs", "seeds run"),
        ("context_rows", "context rows"),
        ("test_rows", "test rows"),
        ("n_features", "features"),
        ("n_chunks", "chunks"),
        ("runtime_seconds_mean", "runtime (s)"),
    ):
        cells = ""
        for summary in summaries.values():
            value = summary.get(field)
            if value is None:
                cells += f"{'-':>16}"
            elif isinstance(value, float):
                cells += f"{value:>16.1f}"
            else:
                cells += f"{str(value):>16}"
        lines.append(f"  {label:<20}{cells}")
    lines.append("=" * width)
    return "\n".join(lines)


def render_markdown(summaries: dict[str, dict[str, Any]]) -> str:
    """Render the comparison as a Markdown table for the report.

    Args:
        summaries: Mapping of experiment tag to its summary.

    Returns:
        Markdown source.
    """
    tags = list(summaries)
    lines = [
        "# Results comparison",
        "",
        "Mean +/- standard deviation across seeds. Standard deviation is 0.000 "
        "where only one seed was run.",
        "",
        "| Metric | " + " | ".join(tags) + " |",
        "|---|" + "---|" * len(tags),
    ]
    for metric in METRICS:
        cells = []
        for summary in summaries.values():
            mean = summary.get(f"{metric}_mean")
            std = summary.get(f"{metric}_std")
            cells.append("n/a" if mean is None else f"{mean:.4f} ± {std:.4f}")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("| Setting | " + " | ".join(tags) + " |")
    lines.append("|---|" + "---|" * len(tags))
    for field, label in (
        ("n_runs", "seeds"),
        ("context_rows", "context rows"),
        ("test_rows", "test rows"),
        ("n_features", "features"),
        ("n_chunks", "chunks"),
    ):
        cells = [str(summaries[tag].get(field, "-")) for tag in tags]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Build and write the comparison tables."""
    args = parse_args()
    config.ensure_dirs()

    experiments = args.experiments or discover_experiments()
    summaries: dict[str, dict[str, Any]] = {}
    for tag in experiments:
        rows = load_results(tag)
        if rows:
            summaries[tag] = summarize(rows)

    if not summaries:
        print(f"No result CSVs found in {config.REPORTS_DIR}", file=sys.stderr)
        return 1

    print(render_text(summaries))

    out_dir = Path(args.out_dir) if args.out_dir else config.TABLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "comparison_table.csv"
    fields = ["experiment", *sorted(next(iter(summaries.values())))]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for tag, summary in summaries.items():
            writer.writerow({"experiment": tag, **summary})

    md_path = out_dir / "comparison_table.md"
    md_path.write_text(render_markdown(summaries), encoding="utf-8")

    print(f"\nWritten:\n  {csv_path}\n  {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
