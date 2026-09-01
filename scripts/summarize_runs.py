#!/usr/bin/env python3
"""Aggregate experiment result CSVs into a mean +/- std table.

Reads the per-run CSVs written by the experiment runners into reports/ and
summarises each experiment tag across seeds. Reporting mean and standard
deviation across seeds is the honest alternative to a significance test at
this sample size: a Wilcoxon signed-rank test needs at least six paired
observations before a two-sided p-value below 0.05 is even reachable, so
three seeds cannot produce a significant result no matter what the data says.

Usage:
    python scripts/summarize_runs.py
    python scripts/summarize_runs.py --tag baseline
    python scripts/summarize_runs.py --save reports/tables/baseline_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import statistics
import sys
from pathlib import Path

from tabpfn_nids import config

METRICS = ("accuracy", "precision", "recall", "f1_score", "roc_auc")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag", default="baseline", help="Experiment tag prefix to aggregate."
    )
    parser.add_argument(
        "--save", default=None, help="Optional path to write the summary CSV."
    )
    return parser.parse_args()


def load_rows(tag: str) -> list[dict[str, str]]:
    """Read every results CSV matching an experiment tag.

    Args:
        tag: Filename prefix, e.g. "baseline".

    Returns:
        One dict per run, sorted by seed.
    """
    pattern = str(config.REPORTS_DIR / f"{tag}_*.csv")
    rows: list[dict[str, str]] = []
    for path in sorted(glob.glob(pattern)):
        with open(path, newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return sorted(rows, key=lambda r: int(r["seed"]))


def summarize(rows: list[dict[str, str]]) -> dict[str, tuple[float, float]]:
    """Compute mean and sample standard deviation for each metric.

    Args:
        rows: Result rows from load_rows.

    Returns:
        Mapping of metric name to a ``(mean, std)`` pair. Std is 0.0 when
        fewer than two runs are present.
    """
    summary: dict[str, tuple[float, float]] = {}
    for metric in METRICS:
        values = [float(r[metric]) for r in rows if r.get(metric) not in (None, "")]
        if not values:
            continue
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        summary[metric] = (statistics.fmean(values), std)
    return summary


def main() -> int:
    """Print the summary table and optionally save it."""
    args = parse_args()
    rows = load_rows(args.tag)
    if not rows:
        print(f"No results found for tag '{args.tag}' in {config.REPORTS_DIR}",
              file=sys.stderr)
        return 1

    first = rows[0]
    print("=" * 74)
    print(f"{args.tag.upper()} — {len(rows)} run(s) on {first['dataset']}")
    print("=" * 74)
    print(f"  context rows    {int(first['context_rows']):,}")
    print(f"  test rows       {int(first['test_rows']):,}")
    print(f"  features        {first['n_features']}")
    print(f"  n_estimators    {first['n_estimators']}")
    print(f"  device          {first['device']}   tabpfn {first['tabpfn_version']}")
    print(f"  checkpoint      {first['checkpoint']}")
    print(f"  seeds           {', '.join(r['seed'] for r in rows)}")

    print("\n" + "-" * 74)
    header = f"  {'metric':<14}" + "".join(f"{'seed ' + r['seed']:>13}" for r in rows)
    print(header + f"{'mean':>13}{'std':>10}")
    print("-" * 74)

    summary = summarize(rows)
    for metric in METRICS:
        if metric not in summary:
            continue
        mean, std = summary[metric]
        cells = "".join(f"{float(r[metric]):>13.4f}" for r in rows)
        print(f"  {metric:<14}{cells}{mean:>13.4f}{std:>10.4f}")

    runtimes = [float(r["runtime_seconds"]) for r in rows]
    print("-" * 74)
    cells = "".join(f"{t:>12.1f}s" for t in runtimes)
    print(f"  {'runtime':<14}{cells}{statistics.fmean(runtimes):>12.1f}s")
    print("=" * 74)

    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", *[f"seed_{r['seed']}" for r in rows],
                             "mean", "std"])
            for metric in METRICS:
                if metric not in summary:
                    continue
                mean, std = summary[metric]
                writer.writerow(
                    [metric, *[r[metric] for r in rows], f"{mean:.6f}", f"{std:.6f}"]
                )
        print(f"Summary written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
