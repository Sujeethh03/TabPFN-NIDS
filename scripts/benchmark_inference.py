"""Benchmark dynamic parallel inference across dataset sizes.

Records:
- Rows
- Workers
- Models
- Total inference time (s)
- Rows / second
- Memory usage (MB)
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
import psutil

from tabpfn_nids.config import PROJECT_ROOT
from tabpfn_nids.inference import (
    DynamicInferenceManager,
    ModelInfo,
    ModelRegistry,
    calculate_worker_count,
)


class FastBenchmarkModel:
    """Calibrated model for benchmarking dynamic parallel inference manager scaling."""

    def __init__(self, feature_count: int = 67, latency_per_row_ms: float = 0.05) -> None:
        self.feature_count = feature_count
        self.latency_per_row_ms = latency_per_row_ms

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        if self.latency_per_row_ms > 0:
            time.sleep((n * self.latency_per_row_ms) / 1000.0)
        p_attack = np.full(n, 0.25)
        return np.column_stack([1.0 - p_attack, p_attack])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


def get_process_memory_mb() -> float:
    """Return current process resident memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_benchmark(
    sizes: list[int],
    use_real_model: bool = False,
    max_rows_per_worker: int = 10_000,
    max_workers: int | None = None,
) -> list[dict]:
    """Execute inference benchmark across specified row sizes."""
    print("=" * 80)
    print(f"BENCHMARKING DYNAMIC PARALLEL INFERENCE (Real Model={use_real_model})")
    print(f"Max rows per worker: {max_rows_per_worker}")
    print("=" * 80)

    model = None
    if use_real_model:
        model_path = PROJECT_ROOT / "data" / "artifacts" / "models" / "tabpfn_binary_model.pkl"
        print(f"Loading real TabPFN model from {model_path}...")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        model = FastBenchmarkModel()

    models = {"tabpfn_model": model}

    results = []

    print(
        f"{'Rows':>8} | {'Workers':>7} | {'Chunks':>6} | {'Models':>6} | "
        f"{'Time (s)':>10} | {'Throughput (rows/s)':>20} | {'RAM (MB)':>10}"
    )
    print("-" * 80)

    for n_rows in sizes:
        # Generate synthetic feature matrix with 67 columns
        X = np.random.randn(n_rows, 67).astype(np.float32)

        mem_before = get_process_memory_mb()

        manager = DynamicInferenceManager(
            max_rows_per_worker=max_rows_per_worker,
            max_workers=max_workers,
            executor_type="thread",
        )

        expected_workers = calculate_worker_count(n_rows, max_rows_per_worker, max_workers)

        t_start = time.perf_counter()
        res = manager.predict(X, models=models)
        t_elapsed = time.perf_counter() - t_start

        mem_after = get_process_memory_mb()
        mem_used = max(mem_after, mem_before)

        throughput = n_rows / t_elapsed if t_elapsed > 0 else 0.0
        num_chunks = res.metadata.get("num_chunks", 1)
        actual_workers = res.metadata.get("num_workers", expected_workers)

        row_metric = {
            "rows": n_rows,
            "workers": actual_workers,
            "chunks": num_chunks,
            "models": len(models),
            "total_inference_time": round(t_elapsed, 4),
            "rows_per_second": round(throughput, 1),
            "memory_mb": round(mem_used, 1),
        }
        results.append(row_metric)

        print(
            f"{n_rows:>8} | {actual_workers:>7} | {num_chunks:>6} | {len(models):>6} | "
            f"{t_elapsed:>10.4f} | {throughput:>20.1f} | {mem_used:>10.1f}"
        )

    print("=" * 80)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark dynamic parallel inference.")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1000, 10000, 10001, 20000, 30000],
        help="List of row counts to benchmark.",
    )
    parser.add_argument(
        "--real-model",
        action="store_true",
        default=False,
        help="Use real TabPFN model instead of fast benchmark model.",
    )
    parser.add_argument(
        "--max-rows-per-worker",
        type=int,
        default=10000,
        help="Max rows per worker chunk (default: 10000).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum parallel workers (default: auto).",
    )
    args = parser.parse_args()

    run_benchmark(
        sizes=args.sizes,
        use_real_model=args.real_model,
        max_rows_per_worker=args.max_rows_per_worker,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
