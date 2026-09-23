"""Basic flow features: duration, total packets, total bytes.

These are the foundational flow-level statistics derived directly from the
reconstructed flow metadata. Every other feature group builds on or extends
these.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_basic_features(df: pd.DataFrame, epsilon: float = 1e-6) -> pd.DataFrame:
    """Compute basic flow-level features.

    Args:
        df: Flow DataFrame with columns: duration, fwd_packets, bwd_packets,
            fwd_bytes, bwd_bytes, total_packets, total_bytes.
        epsilon: Added to zero durations to avoid division by zero.

    Returns:
        DataFrame with basic feature columns added.
    """
    out = df.copy()

    # Ensure duration is non-negative
    out["duration"] = out["duration"].clip(lower=0.0)

    # Safe duration for rate calculations
    out["_safe_duration"] = out["duration"].clip(lower=epsilon)

    # Rate features
    out["packets_per_second"] = out["total_packets"] / out["_safe_duration"]
    out["bytes_per_second"] = out["total_bytes"] / out["_safe_duration"]

    # Drop helper column
    out.drop(columns=["_safe_duration"], inplace=True)

    return out


BASIC_FEATURE_NAMES = [
    "duration",
    "total_packets",
    "total_bytes",
    "packets_per_second",
    "bytes_per_second",
]
