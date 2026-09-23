"""Directional flow features: forward/backward splits and ratios.

These features capture the asymmetry between the initiator (forward) and
responder (backward) directions. Many attack patterns exhibit distinctive
directional signatures — e.g., data exfiltration has high fwd_byte_ratio,
while scanning has high fwd_packet_ratio with low bwd_packets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_directional_features(
    df: pd.DataFrame, epsilon: float = 1e-6
) -> pd.DataFrame:
    """Compute directional (forward/backward) features.

    Args:
        df: Flow DataFrame with fwd_packets, bwd_packets, fwd_bytes,
            bwd_bytes, total_packets, total_bytes, duration.
        epsilon: Denominator guard for rate/ratio calculations.

    Returns:
        DataFrame with directional feature columns added.
    """
    out = df.copy()
    safe_dur = out["duration"].clip(lower=epsilon)

    # Directional counts (already present from flow builder, but ensure they exist)
    for col in ("fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes"):
        if col not in out.columns:
            out[col] = 0

    # Ratios (guarded against total == 0)
    total_pkts = out["total_packets"].clip(lower=1)
    total_bytes = out["total_bytes"].clip(lower=1)

    out["fwd_packet_ratio"] = out["fwd_packets"] / total_pkts
    out["bwd_packet_ratio"] = out["bwd_packets"] / total_pkts
    out["fwd_byte_ratio"] = out["fwd_bytes"] / total_bytes
    out["bwd_byte_ratio"] = out["bwd_bytes"] / total_bytes

    # Directional rates
    out["fwd_packets_per_second"] = out["fwd_packets"] / safe_dur
    out["bwd_packets_per_second"] = out["bwd_packets"] / safe_dur
    out["fwd_bytes_per_second"] = out["fwd_bytes"] / safe_dur
    out["bwd_bytes_per_second"] = out["bwd_bytes"] / safe_dur

    return out


DIRECTIONAL_FEATURE_NAMES = [
    "fwd_packets",
    "bwd_packets",
    "fwd_bytes",
    "bwd_bytes",
    "fwd_packet_ratio",
    "bwd_packet_ratio",
    "fwd_byte_ratio",
    "bwd_byte_ratio",
    "fwd_packets_per_second",
    "bwd_packets_per_second",
    "fwd_bytes_per_second",
    "bwd_bytes_per_second",
]
