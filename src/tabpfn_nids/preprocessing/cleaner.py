"""Data cleaning for PCAP-extracted flow features.

Handles NaN, Inf, negative values, impossible data, and duplicates.
Produces before/after quality reports so every transformation is auditable.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def assess_quality(df: pd.DataFrame, label: str = "data") -> dict[str, Any]:
    """Generate a data quality assessment.

    Args:
        df: DataFrame to assess.
        label: Descriptive label for the report.

    Returns:
        Quality report dict.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    report: dict[str, Any] = {
        "label": label,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "nan_counts": {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()},
        "total_nans": int(df.isna().sum().sum()),
    }

    # Inf counts
    inf_counts = {}
    for col in numeric_cols:
        inf_count = int(np.isinf(df[col].astype(float)).sum()) if df[col].dtype != object else 0
        if inf_count > 0:
            inf_counts[col] = inf_count
    report["inf_counts"] = inf_counts
    report["total_infs"] = sum(inf_counts.values())

    # Negative value counts (for columns that shouldn't be negative)
    non_negative_cols = [c for c in numeric_cols if any(
        kw in c for kw in ["packets", "bytes", "count", "duration", "len"]
    )]
    neg_counts = {}
    for col in non_negative_cols:
        neg = int((df[col] < 0).sum())
        if neg > 0:
            neg_counts[col] = neg
    report["negative_counts"] = neg_counts

    # Duplicate rows
    report["duplicate_rows"] = int(df.duplicated().sum())

    return report


def clean_data(
    df: pd.DataFrame,
    missing_strategy: str = "median",
    remove_impossible: bool = True,
    remove_duplicates: bool = True,
    max_valid_port: int = 65535,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the feature DataFrame.

    Operations (in order):
    1. Replace Inf/-Inf with NaN
    2. Handle missing values (median imputation / drop / zero)
    3. Fix impossible values (negative packets, bytes, duration)
    4. Validate port ranges
    5. Remove exact duplicate flows (optional)

    Args:
        df: Input DataFrame.
        missing_strategy: "median", "drop", or "zero".
        remove_impossible: Whether to clip negative counts to 0.
        remove_duplicates: Whether to remove exact duplicate rows.
        max_valid_port: Maximum valid port number.

    Returns:
        (cleaned_df, cleaning_report) tuple.
    """
    started = time.time()
    out = df.copy()
    initial_rows = len(out)
    operations: list[str] = []

    # 1. Replace Inf with NaN
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    inf_replaced = 0
    for col in numeric_cols:
        mask = np.isinf(out[col].astype(float))
        count = int(mask.sum())
        if count > 0:
            out.loc[mask, col] = np.nan
            inf_replaced += count
    if inf_replaced:
        operations.append(f"Replaced {inf_replaced} Inf values with NaN")

    # 2. Handle missing values
    nan_before = int(out.isna().sum().sum())
    if nan_before > 0:
        if missing_strategy == "median":
            for col in numeric_cols:
                if out[col].isna().any():
                    median_val = out[col].median()
                    out[col] = out[col].fillna(median_val)
            operations.append(f"Imputed {nan_before} NaN values with median")
        elif missing_strategy == "zero":
            out[numeric_cols] = out[numeric_cols].fillna(0)
            operations.append(f"Replaced {nan_before} NaN values with 0")
        elif missing_strategy == "drop":
            out = out.dropna()
            dropped = initial_rows - len(out)
            operations.append(f"Dropped {dropped} rows with NaN")

    # Handle remaining NaN in non-numeric columns
    for col in out.select_dtypes(include=["object", "category"]).columns:
        if out[col].isna().any():
            out[col] = out[col].fillna("unknown")

    # 3. Fix impossible values
    if remove_impossible:
        count_cols = [c for c in numeric_cols if any(
            kw in c for kw in ["packets", "bytes", "count"]
        )]
        clipped = 0
        for col in count_cols:
            neg_mask = out[col] < 0
            count = int(neg_mask.sum())
            if count > 0:
                out.loc[neg_mask, col] = 0
                clipped += count
        if clipped:
            operations.append(f"Clipped {clipped} negative count/byte values to 0")

        # Duration
        if "duration" in out.columns:
            neg_dur = int((out["duration"] < 0).sum())
            if neg_dur > 0:
                out.loc[out["duration"] < 0, "duration"] = 0
                operations.append(f"Clipped {neg_dur} negative durations to 0")

    # 4. Validate ports
    for port_col in ["src_port", "dst_port"]:
        if port_col in out.columns:
            invalid = (out[port_col] < 0) | (out[port_col] > max_valid_port)
            count = int(invalid.sum())
            if count > 0:
                out.loc[invalid, port_col] = 0
                operations.append(f"Fixed {count} invalid {port_col} values")

    # 5. Remove duplicates
    if remove_duplicates:
        dup_count = int(out.duplicated().sum())
        if dup_count > 0:
            out = out.drop_duplicates()
            operations.append(f"Removed {dup_count} duplicate rows")

    elapsed = time.time() - started
    report = {
        "initial_rows": initial_rows,
        "final_rows": len(out),
        "removed_rows": initial_rows - len(out),
        "operations": operations,
        "elapsed_seconds": round(elapsed, 2),
    }

    logger.info(
        "Cleaning: %d → %d rows (%d removed) in %.1fs. Operations: %s",
        initial_rows, len(out), initial_rows - len(out),
        elapsed, "; ".join(operations) if operations else "none needed",
    )

    return out, report


def save_quality_report(
    report: dict[str, Any],
    output_path: Path | str,
) -> None:
    """Save a quality report to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Quality report → %s", output_path)
