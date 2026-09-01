#!/usr/bin/env python3
"""Design-choice ablations for the chunked ensemble (Week 7 deliverable).

Two studies, both on NSL-KDD with engineered features and a fixed seed:

**A. Context size.** The ensemble is run at chunk sizes 1000, 2500, 5000 and
10000 with the chunk *count* held fixed. Holding count fixed is what isolates
context size: if the whole partition were used instead, a smaller chunk size
would also mean more chunks, and the two effects could not be separated.

The cost of that choice, stated plainly: with a fixed chunk count, a larger
chunk size also means more total training data reaches the model. The
comparison answers "does a bigger context help?", not "is a bigger context
better than more chunks of the same total data?" -- that second question needs
its own study.

**B. Stratification.** Stratified chunking against a random control at the
same chunk size, everything else identical.

An honest prediction for study B before the numbers arrive: on *binary*
NSL-KDD this ablation is expected to show little or no difference. The
positive rate is 46.5% and chunks hold ~9,700 rows, so by the law of large
numbers a random partition lands very close to the population balance anyway.
Stratification earns its place on *rare* classes, where a chunk can miss a
class entirely. The script therefore reports per-chunk class-balance drift
alongside the metrics, so the mechanism is measured directly rather than
inferred from an F1 difference that may not appear.

Usage:
    python scripts/run_ablation.py
    python scripts/run_ablation.py --max-chunks 3 --test-size 1000
    python scripts/run_ablation.py --skip-stratification
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
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
from tabpfn_nids.evaluation.reporter import write_results
from tabpfn_nids.models import ChunkedTabPFNEnsemble

logger = logging.getLogger("run_ablation")

CHUNK_SIZES = (1000, 2500, 5000, 10000)
METRICS = ("accuracy", "precision", "recall", "f1_score", "roc_auc")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--chunk-sizes", type=int, nargs="*", default=list(CHUNK_SIZES))
    parser.add_argument("--max-chunks", type=int, default=3,
                        help="Chunks per run, held fixed across chunk sizes.")
    parser.add_argument("--test-size", type=int, default=1000,
                        help="Stratified test subsample shared by every run.")
    parser.add_argument("--n-estimators", type=int, default=2)
    parser.add_argument("--max-service-flag-categories", type=int, default=40,
                        help="Cap on common_service_flag; 336 uncapped OOMs on MPS.")
    parser.add_argument("--predict-batch-size", type=int, default=500)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    parser.add_argument("--no-features", action="store_true",
                        help="Disable engineered features (they are on by default).")
    parser.add_argument("--skip-stratification", action="store_true",
                        help="Run only the context-size study.")
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


def run_one(
    label: str,
    chunk_size: int,
    stratified: bool,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run the ensemble once and return metrics plus chunk diagnostics.

    Args:
        label: Human-readable run label.
        chunk_size: Rows per chunk.
        stratified: Whether chunking preserves class balance.
        data: ``(X_train, y_train, X_test, y_test)``.
        args: Parsed arguments shared by every run.

    Returns:
        A flat result dict.
    """
    X_train, y_train, X_test, y_test = data
    print(f"\n{'=' * 68}\n{label}\n{'=' * 68}")

    started = time.time()
    ensemble = ChunkedTabPFNEnsemble(
        chunk_size=chunk_size,
        random_state=args.seed,
        device=args.device,
        n_estimators=args.n_estimators,
        max_chunks=args.max_chunks,
        predict_batch_size=args.predict_batch_size,
        stratified=stratified,
    )
    ensemble.fit(X_train, y_train)

    # Measure the mechanism, not just the outcome: how far each chunk's class
    # balance drifts from the population rate.
    rates = [float(np.mean(chunk_y)) for _, chunk_y in ensemble.chunks_]
    population = float(np.mean(y_train))
    drift = max(abs(rate - population) for rate in rates)
    context_rows = sum(len(chunk_y) for _, chunk_y in ensemble.chunks_)

    print(f"  chunks {len(ensemble.chunks_)} x {chunk_size:,} = "
          f"{context_rows:,} context rows")
    print(f"  class balance: population {population:.4f}, "
          f"chunk rates {[f'{r:.4f}' for r in rates]}, max drift {drift:.5f}")

    y_proba = ensemble.predict_proba(X_test)
    y_pred = np.argmax(y_proba, axis=1)
    runtime = time.time() - started
    metrics = compute_metrics(y_test, y_pred, y_proba)

    print(f"  f1 {metrics['f1_score']:.4f} | roc_auc {metrics['roc_auc']:.4f} "
          f"| {runtime:.1f}s")

    return {
        "label": label,
        "chunk_size": chunk_size,
        "stratified": stratified,
        "n_chunks": len(ensemble.chunks_),
        "context_rows": context_rows,
        "test_rows": int(len(y_test)),
        "n_features": int(X_train.shape[1]),
        "n_estimators": args.n_estimators,
        "seed": args.seed,
        "max_chunk_balance_drift": drift,
        "chunk_positive_rates": json.dumps([round(r, 5) for r in rates]),
        "mean_chunk_confidence": float(np.mean(ensemble.chunk_confidences_)),
        "runtime_seconds": round(runtime, 2),
        **{m: metrics[m] for m in METRICS},
        "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
    }


def table(rows: list[dict[str, Any]], key: str, key_label: str) -> str:
    """Render results as an aligned text table.

    Args:
        rows: Result dicts.
        key: Field used as the row identifier.
        key_label: Column heading for that field.

    Returns:
        A printable table.
    """
    width = 16 + 11 * len(METRICS) + 22
    lines = ["-" * width,
             f"  {key_label:<14}" + "".join(f"{m:>11}" for m in METRICS)
             + f"{'chunks':>9}{'drift':>10}{'runtime':>10}"]
    lines.append("-" * width)
    for row in rows:
        cells = "".join(f"{row[m]:>11.4f}" for m in METRICS)
        lines.append(
            f"  {str(row[key]):<14}{cells}{row['n_chunks']:>9}"
            f"{row['max_chunk_balance_drift']:>10.5f}"
            f"{row['runtime_seconds']:>9.0f}s"
        )
    lines.append("-" * width)
    return "\n".join(lines)


def interpret_context(rows: list[dict[str, Any]]) -> list[str]:
    """Interpret the context-size trend.

    Args:
        rows: Context-size results, ordered by chunk size.

    Returns:
        Lines of written interpretation.
    """
    f1s = [r["f1_score"] for r in rows]
    sizes = [r["chunk_size"] for r in rows]
    best = int(np.argmax(f1s))
    spread = max(f1s) - min(f1s)
    times = [r["runtime_seconds"] for r in rows]

    lines = [
        f"Best F1 was {f1s[best]:.4f} at chunk_size={sizes[best]:,}.",
        f"Across a {sizes[-1] // sizes[0]}x range of context size, F1 varied by "
        f"{spread:.4f} ({100 * spread:.2f} pp).",
        f"Runtime rose from {times[0]:.0f}s to {times[-1]:.0f}s, "
        f"a factor of {times[-1] / max(times[0], 1e-9):.1f}x.",
        "",
    ]

    if f1s[-1] > f1s[0] and f1s == sorted(f1s):
        lines.append(
            "F1 increases monotonically with context size, so a bigger context "
            "does help on this dataset."
        )
    elif f1s[best] > f1s[-1]:
        lines.append(
            f"F1 peaks at chunk_size={sizes[best]:,} and is *lower* at the "
            f"10,000 limit. Defaulting to the maximum context would cost "
            f"{100 * (f1s[best] - f1s[-1]):.2f} pp of F1 and "
            f"{times[-1] / max(times[best], 1e-9):.1f}x the runtime -- "
            "the ceiling is not the optimum."
        )
    else:
        lines.append(
            "F1 does not vary systematically with context size here; the "
            "differences are within run-to-run noise."
        )

    lines += [
        "",
        "Caveat on the design: chunk count is held fixed, so a larger chunk "
        "size also means more total training data. This measures the joint "
        "effect of context size and data volume, which is the practical "
        "question when choosing chunk_size, but it is not a pure isolation of "
        "context length.",
    ]
    return lines


def interpret_stratification(
    stratified: dict[str, Any], random_run: dict[str, Any], noise_floor: float
) -> list[str]:
    """Interpret the stratification comparison.

    Args:
        stratified: The stratified-chunking result.
        random_run: The random-chunking result.
        noise_floor: Baseline seed-to-seed F1 standard deviation.

    Returns:
        Lines of written interpretation.
    """
    delta_f1 = stratified["f1_score"] - random_run["f1_score"]
    drift_ratio = random_run["max_chunk_balance_drift"] / max(
        stratified["max_chunk_balance_drift"], 1e-9
    )

    lines = [
        f"Stratified F1 {stratified['f1_score']:.4f} vs random "
        f"{random_run['f1_score']:.4f}, a difference of "
        f"{delta_f1:+.4f} ({100 * delta_f1:+.2f} pp).",
        "",
        f"Class-balance drift is the mechanism, and it differs clearly: "
        f"{stratified['max_chunk_balance_drift']:.5f} stratified against "
        f"{random_run['max_chunk_balance_drift']:.5f} random"
        + (f", a factor of {drift_ratio:.0f}x." if drift_ratio > 1.5 else "."),
        "",
    ]

    if abs(delta_f1) < noise_floor:
        lines += [
            f"The F1 difference ({100 * abs(delta_f1):.2f} pp) is SMALLER than "
            f"the baseline's seed-to-seed spread ({100 * noise_floor:.2f} pp), "
            "so it is not evidence that stratification changes accuracy on "
            "this benchmark.",
            "",
            "That is the expected result and it does not mean stratification "
            "is pointless. NSL-KDD's binary positive rate is 46.5% and chunks "
            "hold thousands of rows, so a random partition lands near the "
            "population balance by the law of large numbers -- there is very "
            "little for stratification to fix. Its value is a guarantee rather "
            "than an average-case gain: it bounds worst-case drift to zero, "
            "which matters when a class is rare enough that a random chunk "
            "could miss it entirely. NSL-KDD's U2R family is about 0.04% of "
            "training rows, so under multi-class labels this ablation would "
            "look very different.",
        ]
    elif delta_f1 > 0:
        lines.append(
            "Stratified chunking outperforms random chunking by more than the "
            "seed-to-seed noise floor, so the design choice is supported "
            "directly by the measurement."
        )
    else:
        lines.append(
            "Random chunking scored higher than stratified by more than the "
            "noise floor. That is unexpected and should be investigated before "
            "the result is reported; a single seed is thin evidence either way."
        )
    return lines


def write_report(
    context_rows: list[dict[str, Any]],
    stratification_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    path: Path,
) -> None:
    """Write the ablation study to Markdown.

    Args:
        context_rows: Context-size results.
        stratification_rows: Stratification results, or an empty list.
        args: Parsed arguments, recorded as the shared configuration.
        path: Destination Markdown file.
    """
    env = config.capture_environment(seed=args.seed)
    noise_floor = 0.0229  # baseline seed-to-seed F1 std, reports/baseline_*.csv

    lines = [
        "# Ablation study",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}. "
        f"Seed {args.seed}, NSL-KDD, binary labels.",
        "",
        "Answers the question a reviewer will ask: **how do you know these "
        "design choices matter?**",
        "",
        "## Shared configuration",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| seed | {args.seed} |",
        f"| chunks per run | {args.max_chunks} (held fixed) |",
        f"| test rows | {context_rows[0]['test_rows']:,} (stratified sample) |",
        f"| features | {context_rows[0]['n_features']} |",
        f"| engineered features | {'off' if args.no_features else 'on'} |",
        f"| n_estimators | {args.n_estimators} |",
        f"| device | {config.resolve_device(args.device)} |",
        f"| tabpfn | {env.tabpfn_version}, checkpoint "
        f"{config.TABPFN_CHECKPOINT} |",
        "",
        "## A. Context size",
        "",
        "Chunk count is held fixed so that context size is the variable under "
        "test. Note that this also varies total training data seen; see the "
        "caveat below.",
        "",
        "| chunk_size | " + " | ".join(METRICS) + " | context rows | runtime (s) |",
        "|---|" + "---|" * (len(METRICS) + 2),
    ]
    for row in context_rows:
        cells = " | ".join(f"{row[m]:.4f}" for m in METRICS)
        lines.append(
            f"| {row['chunk_size']:,} | {cells} | {row['context_rows']:,} | "
            f"{row['runtime_seconds']:.0f} |"
        )

    lines += ["", "### Interpretation", ""]
    lines += interpret_context(context_rows)

    if stratification_rows:
        strat, rand = stratification_rows
        lines += [
            "",
            "## B. Stratified vs random chunking",
            "",
            f"Both at chunk_size={strat['chunk_size']:,}, "
            f"{strat['n_chunks']} chunks, identical in every other respect.",
            "",
            "| chunking | " + " | ".join(METRICS) + " | max balance drift |",
            "|---|" + "---|" * (len(METRICS) + 1),
        ]
        for row in (strat, rand):
            cells = " | ".join(f"{row[m]:.4f}" for m in METRICS)
            name = "stratified" if row["stratified"] else "random"
            lines.append(
                f"| {name} | {cells} | {row['max_chunk_balance_drift']:.5f} |"
            )
        lines += ["", "### Per-chunk class balance", "",
                  "| chunking | population rate | per-chunk rates |",
                  "|---|---|---|"]
        for row in (strat, rand):
            name = "stratified" if row["stratified"] else "random"
            lines.append(
                f"| {name} | 0.4654 | `{row['chunk_positive_rates']}` |"
            )
        lines += ["", "### Interpretation", ""]
        lines += interpret_stratification(strat, rand, noise_floor)

    lines += [
        "",
        "## Limitations",
        "",
        f"- Single seed ({args.seed}). The baseline's seed-to-seed F1 spread is "
        f"{noise_floor:.4f} ({100 * noise_floor:.2f} pp); differences smaller "
        "than that are noise.",
        f"- {args.max_chunks} chunks per run, not the full 13-chunk partition, "
        "and a "
        f"{context_rows[0]['test_rows']:,}-row test sample rather than all "
        "22,544. Both are runtime caps.",
        "- Binary labels only. The stratification ablation is far more "
        "informative under multi-class labels, where rare families can be "
        "missed entirely by a random chunk.",
        "- `common_service_flag` is capped at "
        f"{args.max_service_flag_categories} categories; uncapped it produces "
        "463 features and exhausts MPS memory.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Ablation report written to %s", path)


def main() -> int:
    """Run both ablations, print the tables and write the report."""
    args = parse_args()
    config.setup_logging(level=logging.WARNING)
    config.ensure_dirs()
    config.set_seed(args.seed)

    print("=" * 68)
    print("ABLATION STUDY — chunked ensemble design choices")
    print("=" * 68)
    print(f"  seed {args.seed} | chunks/run {args.max_chunks} | "
          f"test rows {args.test_size} | n_estimators {args.n_estimators}")
    print(f"  engineered features: {'OFF' if args.no_features else 'ON'}")

    try:
        X_train, y_train, X_test, y_test = load_and_preprocess_nsl_kdd(
            use_engineered_features=not args.no_features,
            max_service_flag_categories=args.max_service_flag_categories,
        )
    except FileNotFoundError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1

    X_test, y_test = stratified_subsample(X_test, y_test, args.test_size, args.seed)
    data = (X_train, y_train, X_test, y_test)
    print(f"  features {X_train.shape[1]} | train {X_train.shape[0]:,} | "
          f"test {len(y_test):,}")

    # --- A. context size ------------------------------------------------
    context_rows: list[dict[str, Any]] = []
    for chunk_size in args.chunk_sizes:
        context_rows.append(
            run_one(
                f"A. context size — chunk_size={chunk_size:,}",
                chunk_size, True, data, args,
            )
        )

    print("\n" + "=" * 68)
    print("A. CONTEXT SIZE vs F1")
    print("=" * 68)
    print(table(context_rows, "chunk_size", "chunk_size"))
    for line in interpret_context(context_rows):
        print(f"  {line}")

    # --- B. stratification ----------------------------------------------
    stratification_rows: list[dict[str, Any]] = []
    if not args.skip_stratification:
        largest = max(args.chunk_sizes)
        # Reuse the stratified run at this size rather than repeating it.
        strat = next(
            (r for r in context_rows if r["chunk_size"] == largest), None
        ) or run_one(
            f"B. stratified — chunk_size={largest:,}", largest, True, data, args
        )
        rand = run_one(
            f"B. RANDOM chunking — chunk_size={largest:,}",
            largest, False, data, args,
        )
        stratification_rows = [strat, rand]

        print("\n" + "=" * 68)
        print("B. STRATIFIED vs RANDOM CHUNKING")
        print("=" * 68)
        print(table(stratification_rows, "stratified", "stratified"))
        for line in interpret_stratification(strat, rand, 0.0229):
            print(f"  {line}")

    all_rows = context_rows + [r for r in stratification_rows if not r["stratified"]]
    write_results(all_rows, "ablation", seed=args.seed)
    write_report(
        context_rows, stratification_rows, args,
        config.REPORTS_DIR / "ablation_study.md",
    )
    print(f"\nReport written to {config.REPORTS_DIR / 'ablation_study.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
