#!/usr/bin/env python3
"""Vanilla TabPFN baseline on NSL-KDD (Build Plan step 2.2).

This is the reproduction baseline: a single TabPFN context of at most 10,000
stratified training rows, evaluated on the NSL-KDD test split. It is the
number the chunked ensemble (Enhancement 1) has to beat.

The training subsample is stratified so that the 53/47 normal/attack balance
of the full training set is preserved in the context (RULE 3). Results,
together with the seed, hardware and library provenance needed to reproduce
them, are appended to a timestamped CSV in reports/ (RULE 5).

Usage:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --seed 123
    python scripts/run_baseline.py --seed 2024 --test-size 5000
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
from tabpfn_nids.evaluation import compute_metrics, format_metrics
from tabpfn_nids.models import TabPFNWrapper

logger = logging.getLogger("run_baseline")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=config.SEED, help="Random seed (default: 42)."
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=config.MAX_CONTEXT_SAMPLES,
        help="Training rows to subsample (default: 10000, TabPFN's limit).",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help=(
            "Stratified subsample of the test set. Default: the full 22,544 "
            "rows. Prediction cost is linear in this, so it is the main "
            "runtime lever."
        ),
    )
    parser.add_argument(
        "--n-estimators",
        default="auto",
        help=(
            "TabPFN ensemble size. 'auto' scales with feature count; an "
            "integer bounds runtime and makes runs directly comparable."
        ),
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "mps", "cpu"),
        help="Torch backend (default: auto, prefers MPS).",
    )
    parser.add_argument(
        "--tag", default="baseline", help="Prefix for the results filename."
    )
    return parser.parse_args()


def stratified_subsample(
    X: np.ndarray, y: np.ndarray, n_samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Take a class-balance-preserving subsample.

    Uses StratifiedShuffleSplit rather than a plain random draw so the class
    proportions of the source set are reproduced in the subsample (RULE 3).

    Args:
        X: Feature matrix.
        y: Labels.
        n_samples: Rows to draw. If >= len(y), the input is returned unchanged.
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
    """Run the baseline experiment and return a process exit code."""
    args = parse_args()
    config.setup_logging()
    config.ensure_dirs()
    config.set_seed(args.seed)

    env = config.capture_environment(seed=args.seed)
    n_estimators = (
        int(args.n_estimators) if str(args.n_estimators).isdigit() else args.n_estimators
    )

    print("=" * 58)
    print("TabPFN baseline on NSL-KDD")
    print("=" * 58)
    print(f"  seed            {args.seed}")
    print(f"  context size    {args.context_size:,}")
    print(f"  device          {config.resolve_device(args.device)}")
    print(f"  tabpfn          {env.tabpfn_version}")
    print(f"  checkpoint      {config.TABPFN_CHECKPOINT}")
    print("=" * 58)

    started = time.time()

    # --- data ---------------------------------------------------------
    try:
        X_train_full, y_train_full, X_test, y_test = load_and_preprocess_nsl_kdd()
    except FileNotFoundError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1

    X_train, y_train = stratified_subsample(
        X_train_full, y_train_full, args.context_size, args.seed
    )
    if args.test_size:
        X_test, y_test = stratified_subsample(
            X_test, y_test, args.test_size, args.seed
        )

    print(f"\n  train context   {X_train.shape[0]:,} x {X_train.shape[1]}"
          f"   (attack rate {100 * y_train.mean():.2f}%)")
    print(f"  test set        {X_test.shape[0]:,} x {X_test.shape[1]}"
          f"   (attack rate {100 * y_test.mean():.2f}%)")

    # --- model --------------------------------------------------------
    model = TabPFNWrapper(
        device=args.device, random_state=args.seed, n_estimators=n_estimators
    )
    try:
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)
        y_pred = np.argmax(y_proba, axis=1)
    except (ValueError, RuntimeError) as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1

    runtime = time.time() - started
    metrics = compute_metrics(y_test, y_pred, y_proba)

    # --- report -------------------------------------------------------
    print()
    print(format_metrics(metrics, title=f"NSL-KDD baseline (seed {args.seed})"))
    print(f"  fit {model.fit_seconds:.1f}s | predict "
          f"{model.predict_seconds:.1f}s | total {runtime:.1f}s")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    row: dict[str, object] = {
        "timestamp": timestamp,
        "experiment": args.tag,
        "dataset": "nsl-kdd",
        "seed": args.seed,
        "hardware": f"{env.platform} ({env.machine}, {env.cpu_count} cores)",
        "device": model.device,
        "tabpfn_version": env.tabpfn_version,
        "checkpoint": config.TABPFN_CHECKPOINT,
        "torch_version": env.torch_version,
        "sklearn_version": env.sklearn_version,
        "python_version": env.python_version,
        "git_commit": env.git_commit,
        "context_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "n_estimators": model.n_estimators_,
        "runtime_seconds": round(runtime, 2),
        "fit_seconds": round(model.fit_seconds or 0.0, 2),
        "predict_seconds": round(model.predict_seconds or 0.0, 2),
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
