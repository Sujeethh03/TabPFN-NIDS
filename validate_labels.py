import pandas as pd
from collections import defaultdict

FLOW_FILE = r"data/intermediate/flows/1_flows.parquet"
CSV_FILE = r"data/raw/UNSW-NB15/UNSW-NB15_1.csv"

print("Loading flows...")
flows = pd.read_parquet(FLOW_FILE)

print("Loading official UNSW-NB15 CSV...")

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

# --------------------------------------------------
# Build 5-tuple lookup
# --------------------------------------------------

print("Building lookup...")

lookup = defaultdict(list)

for row in csv.itertuples(index=False):

    key = (
        str(row.srcip),
        int(row.sport) if pd.notna(row.sport) else -1,
        str(row.dstip),
        int(row.dsport) if pd.notna(row.dsport) else -1,
        str(row.proto).lower()
    )

    lookup[key].append(row)

# --------------------------------------------------
# Match every flow
# --------------------------------------------------

matches = []

print("Checking matches...")

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

    # Find nearest timestamp
    best = min(
        candidates,
        key=lambda x: abs(float(x.stime) - float(row.start_time))
    )

    diff = abs(float(best.stime) - float(row.start_time))

    matches.append({
        "timestamp_diff": diff,
        "label": int(best.label),
        "attack_cat": str(best.attack_cat).strip()
            if pd.notna(best.attack_cat)
            else "Normal"
    })

matched = pd.DataFrame(matches)

print()
print("=" * 70)
print("LABEL VALIDATION")
print("=" * 70)

print("Extracted flows :", len(flows))
print("Matched flows   :", len(matched))
print("Unmatched flows :", len(flows) - len(matched))

# --------------------------------------------------
# Threshold analysis
# --------------------------------------------------

for threshold in [2, 5, 10]:

    subset = matched[
        matched["timestamp_diff"] <= threshold
    ]

    normal = subset[subset["label"] == 0]
    attack = subset[subset["label"] == 1]

    print()
    print("-" * 70)
    print(f"TIMESTAMP THRESHOLD: <= {threshold} seconds")
    print("-" * 70)

    print("Total:", len(subset))
    print("Normal:", len(normal))
    print("Attack:", len(attack))

    if len(attack) > 0:
        print()
        print("Attack categories:")
        print(attack["attack_cat"].value_counts())

# --------------------------------------------------
# Check conflicting labels for identical 5-tuples
# --------------------------------------------------

print()
print("=" * 70)
print("CHECKING CONFLICTING LABELS")
print("=" * 70)

conflicts = 0

for key, rows in lookup.items():

    labels = set(int(r.label) for r in rows if pd.notna(r.label))

    if len(labels) > 1:
        conflicts += 1

print("5-tuples with both NORMAL and ATTACK labels:", conflicts)

# --------------------------------------------------
# Timestamp statistics
# --------------------------------------------------

print()
print("=" * 70)
print("TIMESTAMP STATISTICS")
print("=" * 70)

print("Minimum:", matched["timestamp_diff"].min())
print("Median :", matched["timestamp_diff"].median())
print("Maximum:", matched["timestamp_diff"].max())

print()
print("Percentiles:")
print(matched["timestamp_diff"].quantile(
    [0.50, 0.75, 0.90, 0.95, 0.99]
))

print("=" * 70)