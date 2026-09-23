"""Packet length statistical features.

Packet size distributions differ sharply across traffic types:
    - DNS queries have small, uniform packets
    - File transfers have large, bimodal packets (headers vs data)
    - Scans have very small, regular packets
    - Floods often use fixed-size packets
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _parse_length_list(len_str: str) -> list[int]:
    """Parse a comma-separated length string into an int list."""
    if not len_str or len_str == "":
        return []
    return [int(x) for x in len_str.split(",") if x.strip()]


def _compute_length_stats(lengths: list[int], prefix: str = "") -> dict[str, float]:
    """Compute length statistics from a list of packet sizes.

    Args:
        lengths: Packet lengths in bytes.
        prefix: Column name prefix (e.g., "fwd_" or "bwd_").

    Returns:
        Dict with min, max, mean, median, std, variance of packet lengths.
    """
    p = prefix
    if not lengths:
        return {
            f"{p}pkt_len_min": 0.0,
            f"{p}pkt_len_max": 0.0,
            f"{p}pkt_len_mean": 0.0,
            f"{p}pkt_len_median": 0.0,
            f"{p}pkt_len_std": 0.0,
            f"{p}pkt_len_var": 0.0,
        }

    arr = np.array(lengths, dtype=np.float64)
    return {
        f"{p}pkt_len_min": float(np.min(arr)),
        f"{p}pkt_len_max": float(np.max(arr)),
        f"{p}pkt_len_mean": float(np.mean(arr)),
        f"{p}pkt_len_median": float(np.median(arr)),
        f"{p}pkt_len_std": float(np.std(arr)),
        f"{p}pkt_len_var": float(np.var(arr)),
    }


def compute_packet_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute packet length statistics for each flow.

    Expects columns ``fwd_packet_lengths`` and ``bwd_packet_lengths`` as
    comma-separated strings.

    Args:
        df: Flow DataFrame.

    Returns:
        DataFrame with packet length statistics added.
    """
    out = df.copy()

    all_stats = []
    fwd_stats = []
    bwd_stats = []

    for _, row in df.iterrows():
        fwd_lens = _parse_length_list(str(row.get("fwd_packet_lengths", "")))
        bwd_lens = _parse_length_list(str(row.get("bwd_packet_lengths", "")))
        all_lens = fwd_lens + bwd_lens

        all_stats.append(_compute_length_stats(all_lens, ""))
        fwd_stats.append(_compute_length_stats(fwd_lens, "fwd_"))
        bwd_stats.append(_compute_length_stats(bwd_lens, "bwd_"))

    # Merge stats into DataFrame
    for stat_list in (all_stats, fwd_stats, bwd_stats):
        for key in stat_list[0]:
            out[key] = [d[key] for d in stat_list]

    return out


PACKET_STATS_FEATURE_NAMES = [
    "pkt_len_min", "pkt_len_max", "pkt_len_mean",
    "pkt_len_median", "pkt_len_std", "pkt_len_var",
    "fwd_pkt_len_min", "fwd_pkt_len_max", "fwd_pkt_len_mean",
    "fwd_pkt_len_median", "fwd_pkt_len_std", "fwd_pkt_len_var",
    "bwd_pkt_len_min", "bwd_pkt_len_max", "bwd_pkt_len_mean",
    "bwd_pkt_len_median", "bwd_pkt_len_std", "bwd_pkt_len_var",
]
