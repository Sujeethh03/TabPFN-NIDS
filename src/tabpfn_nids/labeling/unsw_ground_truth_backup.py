"""
Ground-truth labeling using official UNSW-NB15 per-flow metadata.

This module maps PCAP-extracted flows to the official UNSW-NB15
per-flow dataset using:

1. Exact 5-tuple matching:
   - Source IP
   - Source port
   - Destination IP
   - Destination port
   - Protocol

2. Forward and reverse direction matching.

3. Nearest timestamp matching.

4. Configurable timestamp tolerance.

Only labels originating from the official UNSW-NB15 dataset are used.

Flows that cannot be matched remain UNKNOWN.
No heuristic/approximate labeling is performed.
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


# ---------------------------------------------------------------------
# Official UNSW-NB15 attack categories
# ---------------------------------------------------------------------

ATTACK_CATEGORIES = [
    "Normal",
    "Fuzzers",
    "Analysis",
    "Backdoors",
    "DoS",
    "Exploits",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
]

BINARY_LABELS = {
    "Normal": 0,
    "Attack": 1,
}

UNKNOWN_LABEL = "UNKNOWN"


# ---------------------------------------------------------------------
# Raw UNSW-NB15 flow CSV columns
# ---------------------------------------------------------------------

UNSW_RAW_COLUMNS = [
    "srcip",
    "sport",
    "dstip",
    "dsport",
    "proto",
    "state",
    "dur",
    "sbytes",
    "dbytes",
    "sttl",
    "dttl",
    "sloss",
    "dloss",
    "service",
    "sload",
    "dload",
    "spkts",
    "dpkts",
    "swin",
    "dwin",
    "stcpb",
    "dtcpb",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "sjit",
    "djit",
    "stime",
    "ltime",
    "sintpkt",
    "dintpkt",
    "tcprtt",
    "synack",
    "ackdat",
    "is_sm_ips_ports",
    "ct_state_ttl",
    "ct_flw_http_mthd",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_srv_src",
    "ct_srv_dst",
    "ct_dst_ltm",
    "ct_src_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "attack_cat",
    "label",
]


# ---------------------------------------------------------------------
# Protocol normalization
# ---------------------------------------------------------------------

PROTOCOL_MAP = {
    "1": "icmp",
    "6": "tcp",
    "17": "udp",
    "icmp": "icmp",
    "tcp": "tcp",
    "udp": "udp",
}


def normalize_protocol(value: Any) -> str:
    """
    Convert protocol values to a common representation.

    Examples:
        6       -> tcp
        "6"     -> tcp
        "TCP"   -> tcp
        17      -> udp
        1       -> icmp
    """

    if pd.isna(value):
        return ""

    value_str = str(value).strip().lower()

    # Remove decimal representation such as "6.0"
    if value_str.endswith(".0"):
        value_str = value_str[:-2]

    return PROTOCOL_MAP.get(value_str, value_str)


# ---------------------------------------------------------------------
# Normalize IP
# ---------------------------------------------------------------------

def normalize_ip(value: Any) -> str:
    """Normalize IP address values."""

    if pd.isna(value):
        return ""

    return str(value).strip()


# ---------------------------------------------------------------------
# Normalize port
# ---------------------------------------------------------------------

def normalize_port(value: Any) -> int | None:
    """Normalize port values."""

    if pd.isna(value):
        return None

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------
# Load official UNSW-NB15 flow CSV
# ---------------------------------------------------------------------

def load_ground_truth_csv(gt_path: Path | str) -> pd.DataFrame:
    """
    Load the official UNSW-NB15 per-flow CSV.

    The raw UNSW-NB15 CSV files have:

        - NO header
        - 49 columns
        - label in column 48
        - attack category in column 47
        - start time in column 28
        - last time in column 29

    Example:
        UNSW-NB15_1.csv

    Returns:
        DataFrame containing the official flow records.
    """

    gt_path = Path(gt_path)

    if not gt_path.is_file():
        raise FileNotFoundError(
            f"Ground-truth file not found: {gt_path}"
        )

    logger.info(
        "Loading official UNSW-NB15 flow CSV: %s",
        gt_path,
    )

    # -----------------------------------------------------------------
    # Read raw CSV.
    #
    # The official UNSW-NB15 raw CSV has no header.
    # -----------------------------------------------------------------

    df = pd.read_csv(
        gt_path,
        header=None,
        names=UNSW_RAW_COLUMNS,
        low_memory=False,
        on_bad_lines="skip",
    )

    logger.info(
        "Loaded %d official UNSW-NB15 flow records from %s",
        len(df),
        gt_path.name,
    )

    # -----------------------------------------------------------------
    # Normalize required columns
    # -----------------------------------------------------------------

    df["srcip"] = df["srcip"].map(normalize_ip)
    df["dstip"] = df["dstip"].map(normalize_ip)

    df["sport"] = df["sport"].map(normalize_port)
    df["dsport"] = df["dsport"].map(normalize_port)

    df["proto"] = df["proto"].map(normalize_protocol)

    # Convert timestamps to numeric
    df["stime"] = pd.to_numeric(
        df["stime"],
        errors="coerce",
    )

    df["ltime"] = pd.to_numeric(
        df["ltime"],
        errors="coerce",
    )

    # Convert label
    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce",
    )

    # Normalize attack category
    df["attack_cat"] = (
        df["attack_cat"]
        .fillna("Normal")
        .astype(str)
        .str.strip()
    )

    # Empty / nan attack category means Normal
    df.loc[
        df["attack_cat"].isin(["", "nan", "NaN", "None"]),
        "attack_cat",
    ] = "Normal"

    # Make sure binary label agrees with official label
    df["label"] = df["label"].fillna(
        df["attack_cat"].ne("Normal").astype(int)
    )

    # Remove rows without usable timestamps
    before = len(df)

    df = df.dropna(
        subset=["stime", "ltime"]
    ).reset_index(drop=True)

    removed = before - len(df)

    if removed:
        logger.warning(
            "Removed %d GT rows with invalid timestamps",
            removed,
        )

    return df


# ---------------------------------------------------------------------
# Reference dataset loader
# ---------------------------------------------------------------------

def load_reference_dataset(
    train_path: Path | str,
    test_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Load the pre-generated UNSW-NB15 training/testing parquet files.

    This function is retained for compatibility with the existing
    project, but the current labeling pipeline DOES NOT use these files
    for approximate labeling.

    They remain available for other project functionality.
    """

    train_path = Path(train_path)

    dfs = []

    if train_path.is_file():
        train_df = pd.read_parquet(train_path)

        logger.info(
            "Loaded %d reference rows from %s",
            len(train_df),
            train_path.name,
        )

        dfs.append(train_df)

    if test_path:
        test_path = Path(test_path)

        if test_path.is_file():
            test_df = pd.read_parquet(test_path)

            logger.info(
                "Loaded %d reference rows from %s",
                len(test_df),
                test_path.name,
            )

            dfs.append(test_df)

    if not dfs:
        raise FileNotFoundError(
            "No reference dataset files found."
        )

    return pd.concat(
        dfs,
        ignore_index=True,
    )


# ---------------------------------------------------------------------
# Build lookup dictionary
# ---------------------------------------------------------------------

def build_gt_lookup(
    gt_df: pd.DataFrame,
) -> dict[tuple[str, int | None, str, int | None, str], list[dict]]:
    """
    Build a 5-tuple lookup table.

    Key:

        (
            source_ip,
            source_port,
            destination_ip,
            destination_port,
            protocol
        )

    Each key may have multiple records because UNSW-NB15 can contain
    repeated flows with the same 5-tuple.
    """

    lookup: dict[
        tuple[str, int | None, str, int | None, str],
        list[dict],
    ] = {}

    for row in gt_df.itertuples(index=False):

        key = (
            normalize_ip(row.srcip),
            normalize_port(row.sport),
            normalize_ip(row.dstip),
            normalize_port(row.dsport),
            normalize_protocol(row.proto),
        )

        record = {
            "stime": float(row.stime),
            "ltime": float(row.ltime),
            "attack_cat": str(row.attack_cat).strip(),
            "label": int(row.label),
        }

        lookup.setdefault(key, []).append(record)

    logger.info(
        "Built GT 5-tuple lookup with %d unique keys",
        len(lookup),
    )

    return lookup


# ---------------------------------------------------------------------
# Find nearest GT record
# ---------------------------------------------------------------------

def find_best_gt_match(
    flow: pd.Series,
    lookup: dict,
    time_tolerance: float,
) -> tuple[dict | None, float | None, str | None]:
    """
    Find the nearest official UNSW-NB15 record for a flow.

    Checks:

        1. Forward direction
        2. Reverse direction

    Matching is based on exact 5-tuple + nearest timestamp.

    Returns:

        (record, timestamp_difference, direction)

    or

        (None, None, None)
    """

    src_ip = normalize_ip(flow.get("src_ip"))
    dst_ip = normalize_ip(flow.get("dst_ip"))

    src_port = normalize_port(flow.get("src_port"))
    dst_port = normalize_port(flow.get("dst_port"))

    protocol = normalize_protocol(
        flow.get(
            "protocol_name",
            flow.get("protocol", ""),
        )
    )

    flow_start = float(flow["start_time"])

    # ---------------------------------------------------------------
    # Forward direction
    # ---------------------------------------------------------------

    forward_key = (
        src_ip,
        src_port,
        dst_ip,
        dst_port,
        protocol,
    )

    # ---------------------------------------------------------------
    # Reverse direction
    # ---------------------------------------------------------------

    reverse_key = (
        dst_ip,
        dst_port,
        src_ip,
        src_port,
        protocol,
    )

    candidates = []

    for key, direction in [
        (forward_key, "forward"),
        (reverse_key, "reverse"),
    ]:

        records = lookup.get(key)

        if not records:
            continue

        for record in records:

            # Use the official flow start time as the reference point.
            diff = abs(
                float(record["stime"]) - flow_start
            )

            if diff <= time_tolerance:

                candidates.append(
                    (
                        diff,
                        record,
                        direction,
                    )
                )

    if not candidates:
        return None, None, None

    # Nearest timestamp wins
    candidates.sort(
        key=lambda item: item[0]
    )

    best_diff, best_record, best_direction = candidates[0]

    return (
        best_record,
        best_diff,
        best_direction,
    )


# ---------------------------------------------------------------------
# Match PCAP flows to official UNSW-NB15 records
# ---------------------------------------------------------------------

def match_flows_to_gt_csv(
    flows_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    time_tolerance: float = 5.0,
    use_ip_port: bool = True,
) -> pd.DataFrame:
    """
    Match PCAP-extracted flows to official UNSW-NB15 flow records.

    Matching strategy:

        1. Exact source/destination IP
        2. Exact source/destination port
        3. Exact protocol
        4. Forward or reverse direction
        5. Nearest official start timestamp
        6. Timestamp difference <= time_tolerance

    No approximate feature matching is performed.

    Unmatched flows remain UNKNOWN.
    """

    out = flows_df.copy()

    # ---------------------------------------------------------------
    # Initialize labels
    # ---------------------------------------------------------------

    out["attack_cat"] = UNKNOWN_LABEL
    out["label"] = -1
    out["match_type"] = "unmatched"

    # ---------------------------------------------------------------
    # Validate required flow columns
    # ---------------------------------------------------------------

    required_columns = [
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "start_time",
    ]

    missing = [
        col
        for col in required_columns
        if col not in out.columns
    ]

    if missing:
        raise ValueError(
            f"Flow dataframe missing required columns: {missing}"
        )

    # ---------------------------------------------------------------
    # Build lookup
    # ---------------------------------------------------------------

    lookup = build_gt_lookup(gt_df)

    matched = 0
    unmatched = 0

    forward_matches = 0
    reverse_matches = 0

    timestamp_diffs = []

    # ---------------------------------------------------------------
    # Iterate over PCAP flows
    # ---------------------------------------------------------------

    for idx, flow in out.iterrows():

        try:

            record, diff, direction = find_best_gt_match(
                flow,
                lookup,
                time_tolerance,
            )

        except Exception as exc:

            logger.debug(
                "Error matching flow %s: %s",
                idx,
                exc,
            )

            record = None
            diff = None
            direction = None

        # -----------------------------------------------------------
        # No match
        # -----------------------------------------------------------

        if record is None:

            unmatched += 1

            continue

        # -----------------------------------------------------------
        # Match found
        # -----------------------------------------------------------

        attack_cat = str(
            record["attack_cat"]
        ).strip()

        if not attack_cat or attack_cat.lower() == "nan":
            attack_cat = "Normal"

        official_label = int(
            record["label"]
        )

        out.at[idx, "attack_cat"] = attack_cat
        out.at[idx, "label"] = official_label

        out.at[
            idx,
            "match_type",
        ] = f"gt_csv_{direction}"

        matched += 1

        if direction == "forward":
            forward_matches += 1

        elif direction == "reverse":
            reverse_matches += 1

        if diff is not None:
            timestamp_diffs.append(diff)

    # ---------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------

    total = len(out)

    match_rate = (
        matched / total
        if total
        else 0.0
    )

    logger.info(
        "Official GT matching: %d/%d flows matched (%.2f%%)",
        matched,
        total,
        match_rate * 100,
    )

    logger.info(
        "Forward matches: %d",
        forward_matches,
    )

    logger.info(
        "Reverse matches: %d",
        reverse_matches,
    )

    logger.info(
        "Unmatched flows: %d",
        unmatched,
    )

    if timestamp_diffs:

        diffs = np.asarray(
            timestamp_diffs,
            dtype=float,
        )

        logger.info(
            "Timestamp difference: "
            "min=%.6fs median=%.6fs max=%.6fs",
            float(np.min(diffs)),
            float(np.median(diffs)),
            float(np.max(diffs)),
        )

    return out


# ---------------------------------------------------------------------
# Legacy fallback function
# ---------------------------------------------------------------------

def match_flows_to_reference(
    flows_df: pd.DataFrame,
    ref_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Legacy compatibility function.

    IMPORTANT:
    Approximate reference matching is intentionally disabled.

    The project should never assign a label based only on approximate
    duration/protocol similarity because that can create incorrect
    attack labels.

    Unmatched flows remain UNKNOWN.
    """

    logger.warning(
        "Approximate reference matching is disabled. "
        "Unmatched flows will remain UNKNOWN."
    )

    return flows_df


# ---------------------------------------------------------------------
# Main labeling entry point
# ---------------------------------------------------------------------

def label_flows(
    flows_df: pd.DataFrame,
    gt_csv_path: Path | str | None = None,
    ref_train_path: Path | str | None = None,
    ref_test_path: Path | str | None = None,
    time_tolerance: float = 5.0,
    use_ip_port: bool = True,
) -> pd.DataFrame:
    """
    Main labeling entry point.

    Primary strategy:

        Official UNSW-NB15 per-flow CSV
        + exact 5-tuple
        + forward/reverse matching
        + nearest timestamp

    Unmatched flows remain UNKNOWN.

    Approximate reference matching is NOT performed.
    """

    out = flows_df.copy()

    out["attack_cat"] = UNKNOWN_LABEL
    out["label"] = -1
    out["match_type"] = "unmatched"

    started = time.time()

    # ---------------------------------------------------------------
    # Official GT matching
    # ---------------------------------------------------------------

    if gt_csv_path:

        gt_csv_path = Path(
            gt_csv_path
        )

        if gt_csv_path.is_file():

            logger.info(
                "Using official UNSW-NB15 per-flow ground truth: %s",
                gt_csv_path,
            )

            gt_df = load_ground_truth_csv(
                gt_csv_path
            )

            out = match_flows_to_gt_csv(
                out,
                gt_df,
                time_tolerance=time_tolerance,
                use_ip_port=use_ip_port,
            )

        else:

            logger.warning(
                "Ground-truth CSV not found: %s",
                gt_csv_path,
            )

    else:

        logger.warning(
            "No ground-truth CSV configured."
        )

    # ---------------------------------------------------------------
    # DO NOT use approximate fallback.
    # ---------------------------------------------------------------

    final_unmatched = int(
        (
            out["attack_cat"]
            == UNKNOWN_LABEL
        ).sum()
    )

    final_matched = len(out) - final_unmatched

    elapsed = time.time() - started

    logger.info(
        "Labeling complete in %.1fs: "
        "%d matched, %d UNKNOWN",
        elapsed,
        final_matched,
        final_unmatched,
    )

    # ---------------------------------------------------------------
    # Distribution
    # ---------------------------------------------------------------

    normal_count = int(
        (
            out["attack_cat"]
            == "Normal"
        ).sum()
    )

    attack_count = int(
        (
            (out["attack_cat"] != "Normal")
            & (out["attack_cat"] != UNKNOWN_LABEL)
        ).sum()
    )

    logger.info(
        "Label distribution: Normal=%d Attack=%d UNKNOWN=%d",
        normal_count,
        attack_count,
        final_unmatched,
    )

    return out


# ---------------------------------------------------------------------
# Labeling report
# ---------------------------------------------------------------------

def generate_labeling_report(
    df: pd.DataFrame,
    output_path: Path | str,
) -> dict[str, Any]:
    """
    Generate a diagnostic report for labeling results.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = len(df)

    matched = int(
        (
            df["attack_cat"]
            != UNKNOWN_LABEL
        ).sum()
    )

    unmatched = int(
        (
            df["attack_cat"]
            == UNKNOWN_LABEL
        ).sum()
    )

    normal = int(
        (
            df["attack_cat"]
            == "Normal"
        ).sum()
    )

    attack = int(
        (
            (df["attack_cat"] != "Normal")
            & (df["attack_cat"] != UNKNOWN_LABEL)
        ).sum()
    )

    category_counts = (
        df["attack_cat"]
        .value_counts()
        .to_dict()
    )

    match_counts = (
        df.get(
            "match_type",
            pd.Series(dtype=str),
        )
        .value_counts()
        .to_dict()
    )

    report = {
        "total_flows": total,
        "matched": matched,
        "unmatched": unmatched,
        "match_rate": round(
            matched / max(total, 1),
            4,
        ),
        "normal": normal,
        "attack": attack,
        "category_distribution": {
            str(k): int(v)
            for k, v in category_counts.items()
        },
        "match_type_distribution": {
            str(k): int(v)
            for k, v in match_counts.items()
        },
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    logger.info(
        "Labeling report → %s",
        output_path,
    )

    return report


# ---------------------------------------------------------------------
# Save labeled flows
# ---------------------------------------------------------------------

def save_labeled_flows(
    df: pd.DataFrame,
    output_path: Path | str,
) -> int:
    """
    Save labeled flows to Parquet.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = pa.Table.from_pandas(
        df
    )

    pq.write_table(
        table,
        output_path,
        compression="snappy",
    )

    logger.info(
        "Saved %d labelled flows to %s",
        len(df),
        output_path,
    )

    return len(df)