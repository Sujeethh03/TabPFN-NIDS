"""Feature engineering pipeline — orchestrates all feature groups.

This module is the single entry point for computing ML features from raw
flow records. It chains the individual feature modules (basic, directional,
timing, packet_stats, tcp_flags) in a deterministic order and produces the
final feature DataFrame ready for labeling and preprocessing.

Feature groups can be enabled/disabled via the pipeline config. The output
separates ML features from flow metadata (IPs, ports, timestamps) so the
downstream preprocessing stage can exclude identifiers without losing
traceability.
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

from tabpfn_nids.features.basic import compute_basic_features, BASIC_FEATURE_NAMES
from tabpfn_nids.features.directional import (
    compute_directional_features,
    DIRECTIONAL_FEATURE_NAMES,
)
from tabpfn_nids.features.timing import compute_timing_features, TIMING_FEATURE_NAMES
from tabpfn_nids.features.packet_stats import (
    compute_packet_stats,
    PACKET_STATS_FEATURE_NAMES,
)
from tabpfn_nids.features.tcp_flags import (
    compute_tcp_flag_features,
    TCP_FLAG_FEATURE_NAMES,
)

logger = logging.getLogger(__name__)

# Metadata columns carried from the flow builder — NOT ML features
METADATA_COLUMNS = [
    "flow_id",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "protocol_name",
    "start_time",
    "end_time",
]

# Columns used only for feature derivation (not kept in final output)
DERIVATION_COLUMNS = [
    "fwd_packet_lengths",
    "bwd_packet_lengths",
    "fwd_timestamps",
    "bwd_timestamps",
]


def compute_all_features(
    df: pd.DataFrame,
    groups: dict[str, bool] | None = None,
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    """Apply all enabled feature engineering groups to a flow DataFrame.

    Args:
        df: Raw flow DataFrame from the flow builder (one row per flow).
        groups: Feature group toggles. Defaults to all enabled.
        epsilon: Denominator guard for rate calculations.

    Returns:
        Enriched DataFrame with ML features and metadata columns.
    """
    if groups is None:
        groups = {
            "basic": True, "directional": True, "rate": True,
            "packet_length": True, "inter_arrival_time": True,
            "tcp_flags": True, "protocol": True,
        }

    started = time.time()
    out = df.copy()

    # --- Feature groups ---------------------------------------------------
    if groups.get("basic", True):
        out = compute_basic_features(out, epsilon=epsilon)
        logger.debug("Added basic features")

    if groups.get("directional", True):
        out = compute_directional_features(out, epsilon=epsilon)
        logger.debug("Added directional features")

    if groups.get("inter_arrival_time", True):
        out = compute_timing_features(out)
        logger.debug("Added timing/IAT features")

    if groups.get("packet_length", True):
        out = compute_packet_stats(out)
        logger.debug("Added packet length statistics")

    if groups.get("tcp_flags", True):
        out = compute_tcp_flag_features(out)
        logger.debug("Added TCP flag features")

    # --- Protocol feature ------------------------------------------------
    if groups.get("protocol", True):
        # Keep protocol_name as a categorical feature
        # It will be encoded in the preprocessing stage
        pass  # protocol_name already exists from flow builder

    # --- Drop derivation columns -----------------------------------------
    # These are the raw lists used to compute IAT and length stats
    cols_to_drop = [c for c in DERIVATION_COLUMNS if c in out.columns]
    out.drop(columns=cols_to_drop, inplace=True)

    elapsed = time.time() - started
    ml_cols = [c for c in out.columns if c not in METADATA_COLUMNS]
    logger.info(
        "Feature engineering complete: %d flows × %d ML features in %.2fs",
        len(out), len(ml_cols), elapsed,
    )

    return out


def get_ml_feature_names(
    groups: dict[str, bool] | None = None,
) -> list[str]:
    """Return the names of all ML features produced by the pipeline.

    Args:
        groups: Feature group toggles.

    Returns:
        Sorted list of ML feature column names.
    """
    if groups is None:
        groups = {
            "basic": True, "directional": True,
            "packet_length": True, "inter_arrival_time": True,
            "tcp_flags": True, "protocol": True,
        }

    names: list[str] = []
    if groups.get("basic"):
        names.extend(BASIC_FEATURE_NAMES)
    if groups.get("directional"):
        names.extend(DIRECTIONAL_FEATURE_NAMES)
    if groups.get("inter_arrival_time"):
        names.extend(TIMING_FEATURE_NAMES)
    if groups.get("packet_length"):
        names.extend(PACKET_STATS_FEATURE_NAMES)
    if groups.get("tcp_flags"):
        names.extend(TCP_FLAG_FEATURE_NAMES)
    if groups.get("protocol"):
        names.append("protocol_name")

    return names


def get_metadata_columns() -> list[str]:
    """Return the list of metadata columns (not ML features)."""
    return list(METADATA_COLUMNS)


def save_features_parquet(
    df: pd.DataFrame,
    output_path: Path | str,
) -> int:
    """Save the feature DataFrame to a Parquet file.

    Args:
        df: Feature DataFrame.
        output_path: Destination path.

    Returns:
        Number of rows written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path, compression="snappy")

    logger.info("Saved %d feature rows to %s", len(df), output_path)
    return len(df)


def build_features_from_flows(
    flows_path: Path | str,
    output_path: Path | str,
    groups: dict[str, bool] | None = None,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    """End-to-end: read flow Parquet, compute features, save result.

    Args:
        flows_path: Input flow Parquet file.
        output_path: Output feature Parquet file.
        groups: Feature group toggles.
        epsilon: Denominator guard.

    Returns:
        Summary dict with row/column counts.
    """
    flows_path = Path(flows_path)
    started = time.time()

    df = pd.read_parquet(flows_path)
    logger.info("Read %d flows from %s", len(df), flows_path)

    featured = compute_all_features(df, groups=groups, epsilon=epsilon)
    save_features_parquet(featured, output_path)

    elapsed = time.time() - started
    ml_cols = [c for c in featured.columns if c not in METADATA_COLUMNS]

    summary = {
        "flows_file": flows_path.name,
        "total_flows": len(featured),
        "ml_features": len(ml_cols),
        "metadata_columns": len(METADATA_COLUMNS),
        "elapsed_seconds": round(elapsed, 2),
    }

    logger.info(
        "Built %d features for %d flows in %.1fs → %s",
        len(ml_cols), len(featured), elapsed, output_path,
    )
    return summary
