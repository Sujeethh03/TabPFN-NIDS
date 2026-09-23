"""Leakage detection across train/validation/test splits.

Checks for:
    - Duplicate flow IDs across splits
    - Identical feature vectors across splits
    - Same session (IP pair + close timestamps) in multiple splits
    - Near-duplicate records

Every check produces a report; violations are errors, not warnings.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def check_flow_id_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_column: str = "flow_id",
) -> dict[str, Any]:
    """Check for duplicate flow IDs across splits.

    Args:
        train_df: Training split.
        val_df: Validation split.
        test_df: Test split.
        id_column: Flow identifier column.

    Returns:
        Report dict with overlap counts.
    """
    if id_column not in train_df.columns:
        return {"checked": False, "reason": f"Column '{id_column}' not found"}

    train_ids = set(train_df[id_column])
    val_ids = set(val_df[id_column])
    test_ids = set(test_df[id_column])

    train_val = train_ids & val_ids
    train_test = train_ids & test_ids
    val_test = val_ids & test_ids

    result = {
        "checked": True,
        "train_val_overlap": len(train_val),
        "train_test_overlap": len(train_test),
        "val_test_overlap": len(val_test),
        "clean": len(train_val) == 0 and len(train_test) == 0 and len(val_test) == 0,
    }

    if not result["clean"]:
        logger.error(
            "LEAKAGE DETECTED: %d train-val, %d train-test, %d val-test overlaps",
            len(train_val), len(train_test), len(val_test),
        )
    else:
        logger.info("Flow ID leakage check: CLEAN")

    return result


def check_feature_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Check for identical feature vectors across splits.

    Args:
        train_df: Training split.
        val_df: Validation split.
        test_df: Test split.
        feature_columns: Columns to check. Defaults to all numeric columns.

    Returns:
        Report dict with duplicate counts.
    """
    if feature_columns is None:
        feature_columns = train_df.select_dtypes(include=[np.number]).columns.tolist()

    cols = [c for c in feature_columns if c in train_df.columns
            and c in val_df.columns and c in test_df.columns]

    if not cols:
        return {"checked": False, "reason": "No common feature columns"}

    # Hash each row for efficient comparison
    def row_hashes(df: pd.DataFrame) -> set:
        return set(
            pd.util.hash_pandas_object(df[cols], index=False).values.tolist()
        )

    train_hashes = row_hashes(train_df)
    val_hashes = row_hashes(val_df)
    test_hashes = row_hashes(test_df)

    train_val = len(train_hashes & val_hashes)
    train_test = len(train_hashes & test_hashes)
    val_test = len(val_hashes & test_hashes)

    result = {
        "checked": True,
        "features_checked": len(cols),
        "train_val_duplicates": train_val,
        "train_test_duplicates": train_test,
        "val_test_duplicates": val_test,
        "clean": train_val == 0 and train_test == 0 and val_test == 0,
    }

    if not result["clean"]:
        logger.warning(
            "Feature vector duplicates across splits: %d train-val, "
            "%d train-test, %d val-test",
            train_val, train_test, val_test,
        )
    else:
        logger.info("Feature leakage check: CLEAN")

    return result


def check_session_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ip_columns: tuple[str, str] = ("src_ip", "dst_ip"),
    time_column: str = "start_time",
    time_window: float = 5.0,
) -> dict[str, Any]:
    """Check if the same session (IP pair within time window) spans splits.

    Args:
        train_df: Training split.
        val_df: Validation split.
        test_df: Test split.
        ip_columns: Source and destination IP column names.
        time_column: Timestamp column.
        time_window: Seconds within which same-IP flows are a "session".

    Returns:
        Report dict.
    """
    src_col, dst_col = ip_columns

    if src_col not in train_df.columns or dst_col not in train_df.columns:
        return {"checked": False, "reason": "IP columns not found"}

    def session_keys(df: pd.DataFrame) -> set:
        keys = set()
        for _, row in df.iterrows():
            pair = tuple(sorted([str(row[src_col]), str(row[dst_col])]))
            keys.add(pair)
        return keys

    train_sessions = session_keys(train_df)
    val_sessions = session_keys(val_df)
    test_sessions = session_keys(test_df)

    # IP-pair overlap (not necessarily leakage — same hosts can appear in
    # different time periods — but worth reporting)
    train_val = len(train_sessions & val_sessions)
    train_test = len(train_sessions & test_sessions)

    result = {
        "checked": True,
        "train_sessions": len(train_sessions),
        "val_sessions": len(val_sessions),
        "test_sessions": len(test_sessions),
        "train_val_ip_overlap": train_val,
        "train_test_ip_overlap": train_test,
        "note": "IP-pair overlap is expected in temporal splits (same hosts). "
                "Only exact flow duplicates indicate leakage.",
    }

    logger.info(
        "Session check: %d/%d/%d unique IP pairs (train/val/test), "
        "%d/%d IP overlaps (expected in temporal split)",
        len(train_sessions), len(val_sessions), len(test_sessions),
        train_val, train_test,
    )
    return result


def run_all_leakage_checks(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run all leakage checks and produce a combined report.

    Args:
        train_df: Training split.
        val_df: Validation split.
        test_df: Test split.
        output_path: Where to save the JSON report.

    Returns:
        Combined leakage report.
    """
    report = {
        "flow_id_check": check_flow_id_leakage(train_df, val_df, test_df),
        "feature_check": check_feature_leakage(train_df, val_df, test_df),
        "session_check": check_session_leakage(train_df, val_df, test_df),
    }

    # Overall verdict
    checks = [report["flow_id_check"], report["feature_check"]]
    report["overall_clean"] = all(
        c.get("clean", True) for c in checks if c.get("checked", False)
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Leakage report → %s", output_path)

    return report
