"""TCP flag features.

TCP flags are powerful indicators of connection state and attack patterns:
    - SYN without ACK → connection initiation or SYN flood
    - RST spikes → port scanning or connection resets
    - FIN without preceding data → connection teardown anomalies
    - PSH flags → data push urgency
    - URG flags → rare, often anomalous

Features include raw counts and meaningful ratios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_tcp_flag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute TCP flag features from per-flow flag counts.

    Args:
        df: Flow DataFrame with fwd_syn, fwd_ack, ..., bwd_syn, bwd_ack, ...
            columns from the flow builder.

    Returns:
        DataFrame with TCP flag features added.
    """
    out = df.copy()

    # Total flag counts (both directions)
    out["syn_count"] = out.get("fwd_syn", 0) + out.get("bwd_syn", 0)
    out["ack_count"] = out.get("fwd_ack", 0) + out.get("bwd_ack", 0)
    out["fin_count"] = out.get("fwd_fin", 0) + out.get("bwd_fin", 0)
    out["rst_count"] = out.get("fwd_rst", 0) + out.get("bwd_rst", 0)
    out["psh_count"] = out.get("fwd_psh", 0) + out.get("bwd_psh", 0)
    out["urg_count"] = out.get("fwd_urg", 0) + out.get("bwd_urg", 0)

    # Derived ratios (guarded)
    tcp_mask = out["protocol_name"] == "tcp"
    total_pkts = out["total_packets"].clip(lower=1)

    # SYN ratio — high values suggest scanning
    out["syn_ratio"] = np.where(tcp_mask, out["syn_count"] / total_pkts, 0.0)

    # ACK ratio — normal TCP has many ACKs
    out["ack_ratio"] = np.where(tcp_mask, out["ack_count"] / total_pkts, 0.0)

    # RST ratio — high values suggest port scanning or errors
    out["rst_ratio"] = np.where(tcp_mask, out["rst_count"] / total_pkts, 0.0)

    return out


TCP_FLAG_FEATURE_NAMES = [
    "syn_count", "ack_count", "fin_count",
    "rst_count", "psh_count", "urg_count",
    "syn_ratio", "ack_ratio", "rst_ratio",
]
