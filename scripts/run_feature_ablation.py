#!/usr/bin/env python3
"""Enhancement 2 ablation: engineered features on vs off (Build Plan 4.3).

Runs the chunked TabPFN ensemble twice on NSL-KDD -- once with
``use_engineered_features=False`` and once with it True -- holding everything
else fixed: the same seed, the same chunk size, the same aggregation, the same
n_estimators, the same test rows. The only difference between the two arms is
the feature set, so the delta is attributable to Enhancement 2 alone.

Cost warning: engineering raises the feature count from 122 to 463, because
``common_service_flag`` has 336 distinct values in the training split.
Prediction time grows with feature count, so the "on" arm is substantially
slower than the "off" arm. Use --max-chunks and --test-size to keep the
comparison tractable; both arms always receive identical values.

Usage:
    python scripts/run_feature_ablation.py --seed 42
    python scripts/run_feature_ablation.py --test-size 2000 --max-chunks 4
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from tabpfn_nids import config
from tabpfn_nids.data_pipeline import load_and_preprocess_nsl_kdd
from tabpfn_nids.evaluation import compute_metrics
from tabpfn_nids.models import AGGREGATION_STRATEGIES, ChunkedTabPFNEnsemble

logger = logging.getLogger("run_feature_ablation")

COMPARED_METRICS = ("accuracy", "precision", "recall", "f1_score", "roc_auc")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Random seed (default: 42).")
    parser.add_argument("--chunk-size", type=int, default=config.MAX_CONTEXT_SAMPLES,
                        help="Rows per chunk (default: 10000).")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Cap chunk count in BOTH arms, to bound runtime.")
    parser.add_argument("--test-size", type=int, default=None,
                        help="Stratified test subsample used by BOTH arms.")
    parser.add_argument("--aggregation", default="weighted_vote",
                        choices=AGGREGATION_STRATEGIES)
    parser.add_argument("--n-estimators", default="auto",
                        help="TabPFN ensemble size per chunk; identical in both arms.")
    parser.add_argument("--max-service-flag-categories", type=int, default=None,
                        help="Cap common_service_flag cardinality (336 uncapped).")
    parser.add_argument("--predict-batch-size", type=int, default=500,
                        help="Test rows per prediction batch; lower it if MPS "
                             "runs out of memory at higher feature counts.")
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    parser.add_argument("--tag", default="feature_ablation",
                        help="Prefix for the results filenames.")
    return parser.parse_args()


def stratified_subsample(
    X: np.ndarray, y: np.ndarray, n_samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Take a class-balance-preserving subsample.

    Args:
        X: Feature matrix.
        y: Labels.
        n_samples: Rows to draw; input returned unchanged if larger.
        seed: Random seed.

    Returns:
        An ``(X_subset, y_subset)`` tuple.
    """
    if n_samples >= len(y):
        return X, y
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=n_samples, random_state=seed
    )
    indices, _ = next(splitter.split(X, y))
    return X[indices], y[indices]


def run_arm(
    use_engineered: bool, args: argparse.Namespace
) -> dict[str, Any]:
    """Run one arm of the ablation.

    Args:
        use_engineered: Whether to enable the engineered features.
        args: Parsed command-line arguments, shared by both arms.

    Returns:
        A flat result dict of metrics plus run metadata.
    """
    label = "ENGINEERED" if use_engineered else "BASELINE"
    n_estimators = (
        int(args.n_estimators)
        if str(args.n_estimators).isdigit()
        else args.n_estimators
    )

    print("\n" + "=" * 66)
    print(f"ARM: {label}  (use_engineered_features={use_engineered})")
    print("=" * 66)

    started = time.time()
    X_train, y_train, X_test, y_test = load_and_preprocess_nsl_kdd(
        use_engineered_features=use_engineered,
        max_service_flag_categories=args.max_service_flag_categories,
    )
    if args.test_size:
        X_test, y_test = stratified_subsample(X_test, y_test, args.test_size, args.seed)

    print(f"  features        {X_train.shape[1]}")
    print(f"  train rows      {X_train.shape[0]:,}")
    print(f"  test rows       {X_test.shape[0]:,}")

    ensemble = ChunkedTabPFNEnsemble(
        chunk_size=args.chunk_size,
        aggregation=args.aggregation,
        random_state=args.seed,
        device=args.device,
        n_estimators=n_estimators,
        max_chunks=args.max_chunks,
        predict_batch_size=args.predict_batch_size,
    )
    ensemble.fit(X_train, y_train)
    print(f"  chunks          {len(ensemble.chunks_)}\n")

    y_proba = ensemble.predict_proba(X_test)
    y_pred = np.argmax(y_proba, axis=1)

    runtime = time.time() - started
    metrics = compute_metrics(y_test, y_pred, y_proba)
    described = ensemble.describe()

    result: dict[str, Any] = {
        "arm": label.lower(),
        "use_engineered_features": use_engineered,
        "n_features": int(X_train.shape[1]),
        "n_chunks": described["n_chunks"],
        "context_rows": described["total_rows"],
        "test_rows": int(X_test.shape[0]),
        "runtime_seconds": round(runtime, 2),
        "predict_seconds": round(ensemble.predict_seconds or 0.0, 2),
        "mean_chunk_confidence": described.get("mean_chunk_confidence"),
        **{key: metrics[key] for key in COMPARED_METRICS},
        "true_negatives": metrics["true_negatives"],
        "false_positives": metrics["false_positives"],
        "false_negatives": metrics["false_negatives"],
        "true_positives": metrics["true_positives"],
        "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
    }

    print(f"  f1_score {metrics['f1_score']:.4f} | roc_auc {metrics['roc_auc']:.4f} "
          f"| {runtime:.1f}s")
    return result


def print_comparison(off: dict[str, Any], on: dict[str, Any]) -> bool:
    """Print the ablation comparison table.

    Args:
        off: Result dict for the baseline-features arm.
        on: Result dict for the engineered-features arm.

    Returns:
        True if F1 improved with engineered features.
    """
    print("\n" + "=" * 74)
    print("ENHANCEMENT 2 ABLATION — engineered features off vs on")
    print("=" * 74)
    print(f"  {'':<16}{'off':>12}{'on':>12}{'delta':>12}{'delta pp':>12}")
    print("-" * 74)
    for metric in COMPARED_METRICS:
        a, b = off[metric], on[metric]
        if a is None or b is None:
            continue
        delta = b - a
        print(f"  {metric:<16}{a:>12.4f}{b:>12.4f}{delta:>+12.4f}{100 * delta:>+11.2f}pp")
    print("-" * 74)
    print(f"  {'features':<16}{off['n_features']:>12,}{on['n_features']:>12,}"
          f"{on['n_features'] - off['n_features']:>+12,}")
    print(f"  {'runtime (s)':<16}{off['runtime_seconds']:>12.1f}"
          f"{on['runtime_seconds']:>12.1f}"
          f"{on['runtime_seconds'] - off['runtime_seconds']:>+12.1f}")
    print("=" * 74)

    delta_f1 = on["f1_score"] - off["f1_score"]
    improved = delta_f1 > 0
    if improved:
        verdict = (
            f"Feature engineering IMPROVED F1 by {100 * delta_f1:+.2f} "
            "percentage points."
        )
    elif delta_f1 == 0:
        verdict = "Feature engineering left F1 unchanged."
    else:
        verdict = (
            f"Feature engineering HURT F1 by {100 * delta_f1:.2f} "
            "percentage points."
        )
    print(f"\n  VERDICT: {verdict}")
    print("  Note: this is a single seed. Treat a delta smaller than the "
          "baseline's\n  seed-to-seed spread (about 2.3pp on F1) as noise, "
          "not as an effect.")
    return improved


def write_results_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write both arms to one CSV.

    Args:
        rows: One dict per arm.
        path: Destination path; parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Results written to %s", path)


def main() -> int:
    """Run both arms, print the comparison and save results."""
    args = parse_args()
    config.setup_logging()
    config.ensure_dirs()
    config.set_seed(args.seed)
    env = config.capture_environment(seed=args.seed)

    print("=" * 66)
    print("Enhancement 2 ablation: domain-aware feature engineering")
    print("=" * 66)
    print(f"  seed {args.seed} | chunk size {args.chunk_size:,} | "
          f"aggregation {args.aggregation}")
    print(f"  n_estimators {args.n_estimators} | max_chunks {args.max_chunks} | "
          f"test size {args.test_size or 'full'}")

    try:
        off = run_arm(False, args)
        on = run_arm(True, args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1

    improved = print_comparison(off, on)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shared = {
        "timestamp": timestamp,
        "experiment": args.tag,
        "dataset": "nsl-kdd",
        "seed": args.seed,
        "hardware": f"{env.platform} ({env.machine}, {env.cpu_count} cores)",
        "tabpfn_version": env.tabpfn_version,
        "checkpoint": config.TABPFN_CHECKPOINT,
        "git_commit": env.git_commit,
        "aggregation": args.aggregation,
        "chunk_size": args.chunk_size,
        "n_estimators": args.n_estimators,
    }
    write_results_csv(
        [{**shared, **off}, {**shared, **on}],
        config.REPORTS_DIR / f"{args.tag}_{timestamp}_seed{args.seed}.csv",
    )
    return 0 if improved else 0


if __name__ == "__main__":
    sys.exit(main())
