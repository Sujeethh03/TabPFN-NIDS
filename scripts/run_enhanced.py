#!/usr/bin/env python3
"""Enhancement 1: chunked TabPFN ensemble on NSL-KDD (Build Plan step 3.4).

Uses the FULL 125,973-row NSL-KDD training split rather than the 10,000-row
subsample the baseline is limited to. The data is partitioned into stratified
chunks that each fit one TabPFN context, TabPFN is run on every chunk against
the same test set, and the per-chunk probabilities are aggregated.

To be a valid comparison against scripts/run_baseline.py, three settings must
match the baseline run: --n-estimators, --test-size and the seed. Otherwise
the measured difference reflects those knobs rather than the chunking.

Cost note: runtime is n_chunks x predict(chunk_size, n_test). With 13 chunks
that is roughly 13x the baseline's prediction time, so --test-size is the
main lever for keeping a run tractable.

Usage:
    python scripts/run_enhanced.py --seed 42
    python scripts/run_enhanced.py --seed 42 --test-size 5000 --n-estimators 2
    python scripts/run_enhanced.py --aggregation majority
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

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from tabpfn_nids import config
from tabpfn_nids.data_pipeline import load_and_preprocess_nsl_kdd
from tabpfn_nids.evaluation import compute_metrics, format_metrics, plot_all
from tabpfn_nids.models import AGGREGATION_STRATEGIES, ChunkedTabPFNEnsemble

logger = logging.getLogger("run_enhanced")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Random seed (default: 42).")
    parser.add_argument("--chunk-size", type=int, default=config.MAX_CONTEXT_SAMPLES,
                        help="Rows per chunk (default: 10000).")
    parser.add_argument("--aggregation", default="weighted_vote",
                        choices=AGGREGATION_STRATEGIES,
                        help="How to combine per-chunk probabilities.")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Cap the chunk count to bound runtime.")
    parser.add_argument("--test-size", type=int, default=None,
                        help="Stratified test subsample; default is all 22,544 rows.")
    parser.add_argument("--n-estimators", default="auto",
                        help="TabPFN ensemble size per chunk; match the baseline.")
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"),
                        help="Torch backend (default: auto, prefers MPS).")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip figure generation.")
    parser.add_argument("--tag", default="enhanced",
                        help="Prefix for the results filename.")
    return parser.parse_args()


def stratified_subsample(
    X: np.ndarray, y: np.ndarray, n_samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Take a class-balance-preserving subsample.

    Args:
        X: Feature matrix.
        y: Labels.
        n_samples: Rows to draw; the input is returned unchanged if larger.
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


def write_results_csv(row: dict[str, object], path: Path) -> None:
    """Write one result row to a CSV file.

    Args:
        row: Flat mapping of column name to value.
        path: Destination path; parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    logger.info("Results written to %s", path)


def main() -> int:
    """Run the chunked-ensemble experiment and return an exit code."""
    args = parse_args()
    config.setup_logging()
    config.ensure_dirs()
    config.set_seed(args.seed)

    env = config.capture_environment(seed=args.seed)
    n_estimators = (
        int(args.n_estimators)
        if str(args.n_estimators).isdigit()
        else args.n_estimators
    )

    print("=" * 62)
    print("Enhancement 1: chunked TabPFN ensemble on NSL-KDD")
    print("=" * 62)
    print(f"  seed            {args.seed}")
    print(f"  chunk size      {args.chunk_size:,}")
    print(f"  aggregation     {args.aggregation}")
    print(f"  n_estimators    {n_estimators}")
    print(f"  device          {config.resolve_device(args.device)}")
    print(f"  tabpfn          {env.tabpfn_version}")
    print("=" * 62)

    started = time.time()

    try:
        X_train, y_train, X_test, y_test = load_and_preprocess_nsl_kdd()
    except FileNotFoundError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1

    if args.test_size:
        X_test, y_test = stratified_subsample(X_test, y_test, args.test_size, args.seed)

    print(f"\n  FULL training set  {X_train.shape[0]:,} x {X_train.shape[1]}"
          f"   (attack rate {100 * y_train.mean():.2f}%)")
    print(f"  test set           {X_test.shape[0]:,} x {X_test.shape[1]}"
          f"   (attack rate {100 * y_test.mean():.2f}%)")

    ensemble = ChunkedTabPFNEnsemble(
        chunk_size=args.chunk_size,
        aggregation=args.aggregation,
        random_state=args.seed,
        device=args.device,
        n_estimators=n_estimators,
        max_chunks=args.max_chunks,
    )

    try:
        ensemble.fit(X_train, y_train)
        print(f"  chunks created     {len(ensemble.chunks_)}"
              f"   ({ensemble.chunks_[0][0].shape[0]:,} rows each)\n")
        y_proba = ensemble.predict_proba(X_test)
        y_pred = np.argmax(y_proba, axis=1)
    except (ValueError, RuntimeError) as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1

    runtime = time.time() - started
    metrics = compute_metrics(y_test, y_pred, y_proba)
    described = ensemble.describe()

    print()
    print(format_metrics(
        metrics, title=f"NSL-KDD chunked ensemble (seed {args.seed})"
    ))
    print(f"  {described['n_chunks']} chunks | "
          f"mean chunk confidence {described.get('mean_chunk_confidence', float('nan')):.4f}")
    print(f"  fit {ensemble.fit_seconds:.1f}s | predict "
          f"{ensemble.predict_seconds:.1f}s | total {runtime:.1f}s")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not args.no_plot:
        figures = plot_all(
            y_test,
            y_proba,
            confusion=metrics["confusion_matrix"],
            y_pred=y_pred,
            prefix=f"{args.tag}_seed{args.seed}",
        )
        print("\n  figures:")
        for name, path in figures.items():
            print(f"    {name:<18} {path.relative_to(config.PROJECT_ROOT)}")

    row: dict[str, object] = {
        "timestamp": timestamp,
        "experiment": args.tag,
        "dataset": "nsl-kdd",
        "seed": args.seed,
        "hardware": f"{env.platform} ({env.machine}, {env.cpu_count} cores)",
        "device": described["device"],
        "tabpfn_version": env.tabpfn_version,
        "checkpoint": config.TABPFN_CHECKPOINT,
        "torch_version": env.torch_version,
        "sklearn_version": env.sklearn_version,
        "python_version": env.python_version,
        "git_commit": env.git_commit,
        "aggregation": args.aggregation,
        "chunk_size": args.chunk_size,
        "n_chunks": described["n_chunks"],
        "context_rows": described["total_rows"],
        "test_rows": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "n_estimators": n_estimators,
        "mean_chunk_confidence": described.get("mean_chunk_confidence"),
        "runtime_seconds": round(runtime, 2),
        "fit_seconds": round(ensemble.fit_seconds or 0.0, 2),
        "predict_seconds": round(ensemble.predict_seconds or 0.0, 2),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
        "true_negatives": metrics["true_negatives"],
        "false_positives": metrics["false_positives"],
        "false_negatives": metrics["false_negatives"],
        "true_positives": metrics["true_positives"],
        "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
    }
    write_results_csv(
        row, config.REPORTS_DIR / f"{args.tag}_{timestamp}_seed{args.seed}.csv"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
