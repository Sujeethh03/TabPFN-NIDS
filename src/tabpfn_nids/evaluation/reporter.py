"""Experiment result recording (Build Plan step 5.3).

Every experiment writes one timestamped CSV into ``reports/`` carrying both
its metrics and the provenance needed to reproduce them: seed, hardware,
library versions, the pinned TabPFN checkpoint and the git commit (RULE 5).

``load_results`` reads those CSVs back, which is what the comparison table
and the notebooks are built on.
"""

from __future__ import annotations

import csv
import glob
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from tabpfn_nids import config

logger = logging.getLogger(__name__)


def build_result_row(
    experiment: str,
    dataset: str,
    seed: int,
    metrics: dict[str, Any],
    runtime_seconds: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one result row with metrics and full provenance.

    Args:
        experiment: Experiment tag, e.g. "baseline" or "enhanced".
        dataset: Dataset name, e.g. "nsl-kdd".
        seed: The seed used.
        metrics: A dict from ``compute_metrics``.
        runtime_seconds: Wall-clock seconds for the run.
        extra: Additional experiment-specific columns, e.g. chunk count.

    Returns:
        A flat dict ready for CSV serialisation.
    """
    env = config.capture_environment(seed=seed)
    row: dict[str, Any] = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "experiment": experiment,
        "dataset": dataset,
        "seed": seed,
        "hardware": f"{env.platform} ({env.machine}, {env.cpu_count} cores)",
        "tabpfn_version": env.tabpfn_version,
        "checkpoint": config.TABPFN_CHECKPOINT,
        "torch_version": env.torch_version,
        "torch_backend": env.torch_backend,
        "sklearn_version": env.sklearn_version,
        "pandas_version": env.pandas_version,
        "python_version": env.python_version,
        "git_commit": env.git_commit,
        "runtime_seconds": round(runtime_seconds, 2),
    }
    row.update(extra or {})

    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        row[key] = metrics.get(key)
    for key in (
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    ):
        row[key] = metrics.get(key)
    if "confusion_matrix" in metrics:
        row["confusion_matrix"] = json.dumps(metrics["confusion_matrix"])
    return row


def write_results(
    rows: list[dict[str, Any]] | dict[str, Any],
    experiment: str,
    seed: int | None = None,
    directory: Path | None = None,
) -> Path:
    """Write result rows to a timestamped CSV in reports/.

    Args:
        rows: One row dict, or a list of them.
        experiment: Tag used as the filename prefix.
        seed: Optional seed, appended to the filename.
        directory: Destination directory; defaults to ``config.REPORTS_DIR``.

    Returns:
        The path written.

    Raises:
        ValueError: If no rows are supplied.
    """
    if isinstance(rows, dict):
        rows = [rows]
    if not rows:
        raise ValueError("no result rows to write")

    target = directory or config.REPORTS_DIR
    target.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_seed{seed}" if seed is not None else ""
    path = target / f"{experiment}_{timestamp}{suffix}.csv"

    # Union the keys so rows with differing extras still serialise cleanly.
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Results written to %s", path)
    return path


def load_results(
    experiment: str = "*", directory: Path | None = None
) -> list[dict[str, str]]:
    """Read result CSVs back from reports/.

    Args:
        experiment: Filename prefix to match, or "*" for all.
        directory: Directory to search; defaults to ``config.REPORTS_DIR``.

    Returns:
        One dict per result row, sorted by timestamp then seed.
    """
    target = directory or config.REPORTS_DIR
    rows: list[dict[str, str]] = []
    for path in sorted(glob.glob(str(target / f"{experiment}_*.csv"))):
        with open(path, newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))

    def sort_key(row: dict[str, str]) -> tuple[str, int]:
        try:
            return (row.get("timestamp", ""), int(row.get("seed", 0)))
        except (TypeError, ValueError):
            return (row.get("timestamp", ""), 0)

    return sorted(rows, key=sort_key)
