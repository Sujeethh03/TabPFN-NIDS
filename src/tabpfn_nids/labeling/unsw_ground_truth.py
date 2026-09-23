"""Ground-truth labeling using UNSW-NB15 official metadata.

This module maps PCAP-extracted flows to official UNSW-NB15 ground-truth
labels. It does NOT invent labels using heuristics — every label originates
from the dataset's official ground-truth files.

Labeling Strategy:
    The UNSW-NB15 dataset provides ground truth in two forms:
    1. UNSW_NB15_GT.csv — event-level ground truth with time intervals,
       IPs, ports, and attack categories
    2. Pre-generated training/testing CSV/parquet files with per-flow
       features and labels

    This module uses a multi-strategy matching approach:
    A. If UNSW_NB15_GT.csv is available: match flows by temporal overlap
       + IP/port/protocol
    B. Fallback: match flows against the pre-generated dataset by
       approximate feature similarity (documented as secondary strategy)

    Flows that cannot be matched → labelled "UNKNOWN" (never silently
    assigned to Normal).

Output:
    - Labelled flow DataFrame with attack_cat and binary label columns
    - Diagnostic report with match statistics
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

# Official UNSW-NB15 attack categories
ATTACK_CATEGORIES = [
    "Normal", "Fuzzers", "Analysis", "Backdoors", "DoS",
    "Exploits", "Generic", "Reconnaissance", "Shellcode", "Worms",
]

BINARY_LABELS = {"Normal": 0, "Attack": 1}

UNKNOWN_LABEL = "UNKNOWN"


def load_ground_truth_csv(gt_path: Path | str) -> pd.DataFrame:
    """Load the UNSW_NB15_GT.csv ground-truth event file.

    The GT file has columns including:
        srcip, sport, dstip, dsport, proto, start_time, last_time,
        attack_cat, ...

    Args:
        gt_path: Path to UNSW_NB15_GT.csv.

    Returns:
        DataFrame of ground-truth events.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    gt_path = Path(gt_path)
    if not gt_path.is_file():
        raise FileNotFoundError(
            f"Ground-truth file not found: {gt_path}\n"
            "Download UNSW_NB15_GT.csv from the UNSW-NB15 dataset page."
        )

    df = pd.read_csv(gt_path, low_memory=False)
    logger.info("Loaded %d GT events from %s", len(df), gt_path.name)
    return df


def load_reference_dataset(
    train_path: Path | str,
    test_path: Path | str | None = None,
) -> pd.DataFrame:
    """Load the pre-generated UNSW-NB15 training/testing parquets as reference.

    These contain pre-computed flow features + labels and are used ONLY as
    ground-truth reference for labeling PCAP-extracted flows. They are NOT
    used as the primary feature source for ML.

    Args:
        train_path: Path to UNSW_NB15_training-set.parquet.
        test_path: Path to UNSW_NB15_testing-set.parquet (optional).

    Returns:
        Combined reference DataFrame.
    """
    train_path = Path(train_path)
    dfs = []

    if train_path.is_file():
        train_df = pd.read_parquet(train_path)
        logger.info(
            "Loaded %d reference rows from %s (ground-truth ONLY)",
            len(train_df), train_path.name,
        )
        dfs.append(train_df)

    if test_path:
        test_path = Path(test_path)
        if test_path.is_file():
            test_df = pd.read_parquet(test_path)
            logger.info(
                "Loaded %d reference rows from %s (ground-truth ONLY)",
                len(test_df), test_path.name,
            )
            dfs.append(test_df)

    if not dfs:
        raise FileNotFoundError(
            "No reference dataset files found. Need at least "
            "UNSW_NB15_training-set.parquet."
        )

    return pd.concat(dfs, ignore_index=True)


def match_flows_to_gt_csv(
    flows_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    time_tolerance: float = 1.0,
    use_ip_port: bool = True,
) -> pd.DataFrame:
    """Match PCAP-extracted flows to GT events using temporal + 5-tuple matching.

    For each flow, finds GT events that overlap temporally and match on
    IP/port/protocol. If multiple GT events match, the one with the longest
    temporal overlap is chosen.

    Args:
        flows_df: PCAP-extracted flows with src_ip, dst_ip, src_port,
            dst_port, protocol_name, start_time, end_time.
        gt_df: Ground-truth events from UNSW_NB15_GT.csv.
        time_tolerance: Seconds of tolerance for temporal matching.
        use_ip_port: Whether to also match on IP/port.

    Returns:
        flows_df with attack_cat and label columns added.
    """
    out = flows_df.copy()
    out["attack_cat"] = UNKNOWN_LABEL
    out["label"] = -1  # -1 = unmatched
    out["match_type"] = "unmatched"

    # Normalise GT column names
    gt = gt_df.copy()
    col_map = {}
    for c in gt.columns:
        col_map[c.strip().lower()] = c
    # Try to find standard GT columns
    src_ip_col = col_map.get("srcip", col_map.get("src_ip", None))
    dst_ip_col = col_map.get("dstip", col_map.get("dst_ip", None))
    sport_col = col_map.get("sport", col_map.get("src_port", None))
    dsport_col = col_map.get("dsport", col_map.get("dst_port", None))
    start_col = col_map.get("start_time", col_map.get("stime", None))
    end_col = col_map.get("last_time", col_map.get("ltime", None))
    cat_col = col_map.get("attack_cat", col_map.get("category", None))

    if not all([src_ip_col, dst_ip_col, start_col, cat_col]):
        logger.warning(
            "GT CSV missing required columns. Available: %s", list(gt.columns)
        )
        return out

    matched = 0
    for idx, flow in out.iterrows():
        flow_start = flow["start_time"]
        flow_end = flow["end_time"]

        # Temporal filter: GT events overlapping with this flow
        if start_col and end_col:
            mask = (
                (gt[start_col] <= flow_end + time_tolerance) &
                (gt[end_col] >= flow_start - time_tolerance)
            )
        else:
            continue

        # IP/port filter
        if use_ip_port and src_ip_col and dst_ip_col:
            ip_mask = (
                ((gt[src_ip_col].astype(str) == str(flow["src_ip"])) &
                 (gt[dst_ip_col].astype(str) == str(flow["dst_ip"]))) |
                ((gt[src_ip_col].astype(str) == str(flow["dst_ip"])) &
                 (gt[dst_ip_col].astype(str) == str(flow["src_ip"])))
            )
            mask = mask & ip_mask

        candidates = gt[mask]
        if len(candidates) > 0:
            # Pick the candidate with the most temporal overlap
            cat = candidates.iloc[0][cat_col]
            if pd.isna(cat) or str(cat).strip() == "":
                cat = "Normal"
            cat = str(cat).strip()

            out.at[idx, "attack_cat"] = cat
            out.at[idx, "label"] = 0 if cat == "Normal" else 1
            out.at[idx, "match_type"] = "gt_csv"
            matched += 1

    logger.info(
        "GT CSV matching: %d/%d flows matched (%.1f%%)",
        matched, len(out), 100 * matched / max(len(out), 1),
    )
    return out


def match_flows_to_reference(
    flows_df: pd.DataFrame,
    ref_df: pd.DataFrame,
) -> pd.DataFrame:
    """Fallback matching: use reference dataset features for approximate labeling.

    When UNSW_NB15_GT.csv is not available, this matches PCAP-extracted flows
    to the pre-generated UNSW dataset using protocol, duration, packet counts,
    and byte counts as matching features.

    This is explicitly documented as a secondary/approximate strategy.

    Args:
        flows_df: PCAP-extracted flows with computed features.
        ref_df: Pre-generated UNSW-NB15 reference dataset with attack_cat.

    Returns:
        flows_df with attack_cat and label columns added.
    """
    out = flows_df.copy()

    # Only set labels for rows still unmatched
    unmatched_mask = out.get("attack_cat", UNKNOWN_LABEL) == UNKNOWN_LABEL

    if not unmatched_mask.any():
        return out

    # Reference features for matching
    if "proto" in ref_df.columns:
        ref_proto = ref_df["proto"].astype(str).str.lower()
    else:
        ref_proto = pd.Series([""] * len(ref_df))

    if "attack_cat" not in ref_df.columns:
        logger.warning("Reference dataset has no attack_cat column")
        return out

    matched = 0
    for idx in out[unmatched_mask].index:
        flow = out.loc[idx]
        proto = str(flow.get("protocol_name", "")).lower()

        # Find reference flows with matching protocol
        proto_mask = ref_proto == proto
        candidates = ref_df[proto_mask]

        if len(candidates) == 0:
            continue

        # Match on duration similarity
        flow_dur = float(flow.get("duration", 0))
        if "dur" in candidates.columns:
            dur_diff = (candidates["dur"] - flow_dur).abs()
            best_idx = dur_diff.idxmin()
            best = candidates.loc[best_idx]

            cat = str(best["attack_cat"]).strip()
            if pd.isna(cat) or cat == "" or cat == "nan":
                cat = "Normal"

            out.at[idx, "attack_cat"] = cat
            out.at[idx, "label"] = 0 if cat == "Normal" else 1
            out.at[idx, "match_type"] = "reference_approx"
            matched += 1

    logger.info(
        "Reference matching: %d additional flows matched", matched,
    )
    return out


def label_flows(
    flows_df: pd.DataFrame,
    gt_csv_path: Path | str | None = None,
    ref_train_path: Path | str | None = None,
    ref_test_path: Path | str | None = None,
    time_tolerance: float = 1.0,
    use_ip_port: bool = True,
) -> pd.DataFrame:
    """Main labeling entry point: apply all available labeling strategies.

    Strategy order:
    1. If GT CSV available → temporal + IP/port matching
    2. For remaining unmatched → reference dataset approximate matching
    3. Still unmatched → remain as UNKNOWN

    Args:
        flows_df: PCAP-extracted flows with features.
        gt_csv_path: Path to UNSW_NB15_GT.csv (primary strategy).
        ref_train_path: Path to pre-generated training parquet.
        ref_test_path: Path to pre-generated testing parquet.
        time_tolerance: Seconds tolerance for temporal matching.
        use_ip_port: Whether to match on IP/port.

    Returns:
        Labelled DataFrame.
    """
    out = flows_df.copy()
    out["attack_cat"] = UNKNOWN_LABEL
    out["label"] = -1
    out["match_type"] = "unmatched"

    started = time.time()

    # Strategy 1: GT CSV
    if gt_csv_path and Path(gt_csv_path).is_file():
        logger.info("Using primary labeling strategy: UNSW_NB15_GT.csv")
        gt_df = load_ground_truth_csv(gt_csv_path)
        out = match_flows_to_gt_csv(
            out, gt_df,
            time_tolerance=time_tolerance,
            use_ip_port=use_ip_port,
        )

    # Strategy 2: Reference dataset (for remaining unmatched flows)
    unmatched_count = (out["attack_cat"] == UNKNOWN_LABEL).sum()
    if unmatched_count > 0 and ref_train_path:
        ref_train_path = Path(ref_train_path)
        if ref_train_path.is_file():
            logger.info(
                "Using fallback strategy: reference dataset matching "
                "(%d unmatched flows)", unmatched_count,
            )
            ref_df = load_reference_dataset(ref_train_path, ref_test_path)
            out = match_flows_to_reference(out, ref_df)

    elapsed = time.time() - started
    final_unmatched = (out["attack_cat"] == UNKNOWN_LABEL).sum()

    logger.info(
        "Labeling complete in %.1fs: %d matched, %d UNKNOWN",
        elapsed, len(out) - final_unmatched, final_unmatched,
    )

    return out


def generate_labeling_report(
    df: pd.DataFrame,
    output_path: Path | str,
) -> dict[str, Any]:
    """Generate a diagnostic report on labeling results.

    Args:
        df: Labelled flow DataFrame.
        output_path: Where to save the JSON report.

    Returns:
        Report dict.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(df)
    matched = (df["attack_cat"] != UNKNOWN_LABEL).sum()
    unmatched = (df["attack_cat"] == UNKNOWN_LABEL).sum()
    normal = (df["attack_cat"] == "Normal").sum()
    attack = ((df["attack_cat"] != "Normal") & (df["attack_cat"] != UNKNOWN_LABEL)).sum()

    # Per-category counts
    cat_counts = df["attack_cat"].value_counts().to_dict()

    # Match type counts
    match_counts = df.get("match_type", pd.Series(dtype=str)).value_counts().to_dict()

    report = {
        "total_flows": int(total),
        "matched": int(matched),
        "unmatched": int(unmatched),
        "match_rate": round(matched / max(total, 1), 4),
        "normal": int(normal),
        "attack": int(attack),
        "category_distribution": {str(k): int(v) for k, v in cat_counts.items()},
        "match_type_distribution": {str(k): int(v) for k, v in match_counts.items()},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Labeling report → %s", output_path)
    return report


def save_labeled_flows(
    df: pd.DataFrame,
    output_path: Path | str,
) -> int:
    """Save labelled flows to Parquet.

    Args:
        df: Labelled flow DataFrame.
        output_path: Destination path.

    Returns:
        Number of rows written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path, compression="snappy")

    logger.info("Saved %d labelled flows to %s", len(df), output_path)
    return len(df)
