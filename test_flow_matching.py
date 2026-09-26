import pandas as pd
import numpy as np

FLOW_FILE = r"data/intermediate/flows/1_flows.parquet"
CSV_FILE = r"data/raw/UNSW-NB15/UNSW-NB15_1.csv"

print("Loading extracted flows...")
flows = pd.read_parquet(FLOW_FILE)

print("Flows:", len(flows))

print("Loading official CSV...")

csv = pd.read_csv(
    CSV_FILE,
    header=None,
    usecols=[0, 1, 2, 3, 4, 28, 29, 47, 48],
    names=[
        "srcip",
        "sport",
        "dstip",
        "dsport",
        "proto",
        "stime",
        "ltime",
        "attack_cat",
        "label"
    ],
    on_bad_lines="skip",
    low_memory=False
)

csv["stime"] = pd.to_numeric(csv["stime"], errors="coerce")
csv["ltime"] = pd.to_numeric(csv["ltime"], errors="coerce")
csv["sport"] = pd.to_numeric(csv["sport"], errors="coerce")
csv["dsport"] = pd.to_numeric(csv["dsport"], errors="coerce")
csv["label"] = pd.to_numeric(csv["label"], errors="coerce")

protocol_map = {
    6: "tcp",
    17: "udp",
    1: "icmp"
}

flows["proto_name"] = flows["protocol"].map(protocol_map)

print("Building CSV lookup...")

lookup = {}

for row in csv.itertuples(index=False):

    key = (
        str(row.srcip),
        int(row.sport) if pd.notna(row.sport) else -1,
        str(row.dstip),
        int(row.dsport) if pd.notna(row.dsport) else -1,
        str(row.proto).lower()
    )

    lookup.setdefault(key, []).append(row)

print("Matching flows...")

results = []

for row in flows.itertuples(index=False):

    sport = int(row.src_port) if pd.notna(row.src_port) else -1
    dport = int(row.dst_port) if pd.notna(row.dst_port) else -1

    proto = str(row.proto_name).lower()

    direct_key = (
        str(row.src_ip),
        sport,
        str(row.dst_ip),
        dport,
        proto
    )

    reverse_key = (
        str(row.dst_ip),
        dport,
        str(row.src_ip),
        sport,
        proto
    )

    candidates = lookup.get(direct_key, []) + lookup.get(reverse_key, [])

    if not candidates:
        continue

    best = min(
        candidates,
        key=lambda x: abs(float(x.stime) - float(row.start_time))
    )

    diff = abs(float(best.stime) - float(row.start_time))

    results.append({
        "timestamp_diff": diff,
        "label": int(best.label)
    })

# --------------------------------------------------
# Results
# --------------------------------------------------

matched = pd.DataFrame(results)

print()
print("=" * 70)
print("TIMESTAMP THRESHOLD ANALYSIS")
print("=" * 70)

print("Total extracted flows:", len(flows))
print("Matched flows:", len(matched))
print("Unmatched flows:", len(flows) - len(matched))

print()

thresholds = [0.5, 1, 2, 5, 10, 30, 60]

print(
    f"{'Threshold':<15}"
    f"{'Normal':<12}"
    f"{'Attack':<12}"
    f"{'Total':<12}"
    f"{'Coverage':<12}"
)

print("-" * 63)

for threshold in thresholds:

    subset = matched[
        matched["timestamp_diff"] <= threshold
    ]

    normal = int((subset["label"] == 0).sum())
    attack = int((subset["label"] == 1).sum())
    total = len(subset)
    coverage = total / len(flows) * 100

    print(
        f"≤ {threshold:<12}"
        f"{normal:<12}"
        f"{attack:<12}"
        f"{total:<12}"
        f"{coverage:.2f}%"
    )

print()

print("Overall matched label distribution:")
print(matched["label"].value_counts())

print()

print("Timestamp statistics:")
print("Minimum:", matched["timestamp_diff"].min())
print("Median :", matched["timestamp_diff"].median())
print("Maximum:", matched["timestamp_diff"].max())

print("=" * 70)