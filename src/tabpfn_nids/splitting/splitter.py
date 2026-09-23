"""Leakage-safe train/validation/test splitting.

Network traffic has temporal and session dependencies. A random split risks
placing packets from the same session in both train and test, inflating
metrics. This module implements temporal splitting (preferred) and
group-aware splitting as alternatives.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def temporal_split(
    df: pd.DataFrame,
    time_column: str = "start_time",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time: earliest → train, middle → validation, latest → test.

    This is the recommended split for network traffic. It ensures that the
    model is evaluated on traffic it has never seen, including traffic from
    temporal patterns (time-of-day effects, attack campaigns) that differ
    from training.

    Args:
        df: DataFrame with a timestamp column.
        time_column: Column to sort by.
        train_ratio: Fraction of rows for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for test.

    Returns:
        (train_df, val_df, test_df) tuple.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    sorted_df = df.sort_values(time_column).reset_index(drop=True)
    n = len(sorted_df)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = sorted_df.iloc[:train_end].copy()
    val_df = sorted_df.iloc[train_end:val_end].copy()
    test_df = sorted_df.iloc[val_end:].copy()

    logger.info(
        "Temporal split: train=%d (%.0f%%), val=%d (%.0f%%), test=%d (%.0f%%)",
        len(train_df), 100 * len(train_df) / n,
        len(val_df), 100 * len(val_df) / n,
        len(test_df), 100 * len(test_df) / n,
    )

    return train_df, val_df, test_df


def stratified_random_split(
    df: pd.DataFrame,
    label_column: str = "label",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified random split preserving class balance.

    Less recommended than temporal split for network data, but useful when
    the temporal structure is insufficient.

    Args:
        df: DataFrame with a label column.
        label_column: Column containing class labels.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for test.
        random_seed: Random seed.

    Returns:
        (train_df, val_df, test_df) tuple.
    """
    from sklearn.model_selection import train_test_split

    # First split: train+val vs test
    train_val, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df[label_column],
        random_state=random_seed,
    )

    # Second split: train vs val
    adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val,
        test_size=adjusted_val_ratio,
        stratify=train_val[label_column],
        random_state=random_seed,
    )

    logger.info(
        "Stratified split: train=%d, val=%d, test=%d",
        len(train_df), len(val_df), len(test_df),
    )

    return train_df, val_df, test_df


def split_dataset(
    df: pd.DataFrame,
    strategy: str = "temporal",
    time_column: str = "start_time",
    label_column: str = "label",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataset using the configured strategy.

    Args:
        df: Input DataFrame.
        strategy: "temporal" or "stratified_random".
        time_column: Column for temporal sorting.
        label_column: Column for stratification.
        train_ratio: Training fraction.
        val_ratio: Validation fraction.
        test_ratio: Test fraction.
        random_seed: Random seed.

    Returns:
        (train_df, val_df, test_df) tuple.
    """
    if strategy == "temporal":
        return temporal_split(df, time_column, train_ratio, val_ratio, test_ratio)
    elif strategy == "stratified_random":
        return stratified_random_split(
            df, label_column, train_ratio, val_ratio, test_ratio, random_seed,
        )
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path | str,
    task: str = "binary",
) -> dict[str, str]:
    """Save train/val/test splits to Parquet files.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.
        output_dir: Base directory (data/processed/).
        task: "binary" or "multiclass".

    Returns:
        Dict mapping split name to file path.
    """
    out_dir = Path(output_dir) / task
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        path = out_dir / f"{name}.parquet"
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path, compression="snappy")
        paths[name] = str(path)
        logger.info("Saved %s split: %d rows → %s", name, len(df), path)

    return paths
