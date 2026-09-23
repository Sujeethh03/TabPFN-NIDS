"""Inter-arrival time (IAT) features.

Packet inter-arrival times are the time gaps between consecutive packets
within a flow. Their statistics (mean, std, min, max) are strong indicators
of traffic behaviour:
    - Regular IATs suggest automated traffic (bots, scans)
    - High IAT variance suggests interactive sessions
    - Near-zero IATs suggest floods or bursts
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _parse_timestamp_list(ts_str: str) -> list[float]:
    """Parse a comma-separated timestamp string into a float list."""
    if not ts_str or ts_str == "":
        return []
    return [float(t) for t in ts_str.split(",") if t.strip()]


def _compute_iat_stats(timestamps: list[float]) -> dict[str, float]:
    """Compute IAT statistics from a list of packet timestamps.

    Args:
        timestamps: Sorted epoch timestamps of packets in one direction.

    Returns:
        Dict with mean_iat, std_iat, min_iat, max_iat, median_iat.
    """
    if len(timestamps) < 2:
        return {
            "mean_iat": 0.0,
            "std_iat": 0.0,
            "min_iat": 0.0,
            "max_iat": 0.0,
            "median_iat": 0.0,
        }

    ts = np.array(sorted(timestamps))
    iats = np.diff(ts)

    return {
        "mean_iat": float(np.mean(iats)),
        "std_iat": float(np.std(iats)),
        "min_iat": float(np.min(iats)),
        "max_iat": float(np.max(iats)),
        "median_iat": float(np.median(iats)),
    }


def compute_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute inter-arrival time features for each flow.

    Expects columns ``fwd_timestamps`` and ``bwd_timestamps`` as
    comma-separated strings of epoch timestamps.

    Args:
        df: Flow DataFrame with timestamp columns.

    Returns:
        DataFrame with IAT feature columns added.
    """
    out = df.copy()

    # Overall IAT (combine both directions)
    all_iats = []
    fwd_iats = []
    bwd_iats = []

    for _, row in df.iterrows():
        fwd_ts = _parse_timestamp_list(str(row.get("fwd_timestamps", "")))
        bwd_ts = _parse_timestamp_list(str(row.get("bwd_timestamps", "")))
        all_ts = sorted(fwd_ts + bwd_ts)

        all_iats.append(_compute_iat_stats(all_ts))
        fwd_iats.append(_compute_iat_stats(fwd_ts))
        bwd_iats.append(_compute_iat_stats(bwd_ts))

    # Overall IAT
    for stat in ("mean_iat", "std_iat", "min_iat", "max_iat", "median_iat"):
        out[stat] = [d[stat] for d in all_iats]

    # Forward IAT
    for stat in ("mean_iat", "std_iat"):
        out[f"fwd_{stat}"] = [d[stat] for d in fwd_iats]

    # Backward IAT
    for stat in ("mean_iat", "std_iat"):
        out[f"bwd_{stat}"] = [d[stat] for d in bwd_iats]

    return out


TIMING_FEATURE_NAMES = [
    "mean_iat",
    "std_iat",
    "min_iat",
    "max_iat",
    "median_iat",
    "fwd_mean_iat",
    "fwd_std_iat",
    "bwd_mean_iat",
    "bwd_std_iat",
]
