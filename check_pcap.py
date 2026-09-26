import pandas as pd
from scapy.utils import RawPcapReader
from datetime import datetime, timezone

PCAP = r"data/raw/UNSW-NB15/10.pcap"
CSV = r"data/raw/UNSW-NB15/UNSW-NB15_1.csv"

print("=" * 60)
print("CHECKING:", PCAP)
print("=" * 60)

# Read PCAP timestamps
reader = RawPcapReader(PCAP)

first = None
last = None
packets = 0

for packet, metadata in reader:
    timestamp = float(metadata.sec) + float(metadata.usec) / 1_000_000

    if first is None:
        first = timestamp

    last = timestamp
    packets += 1

reader.close()

print("Packets:", packets)
print("First UTC:", datetime.fromtimestamp(first, timezone.utc))
print("Last UTC :", datetime.fromtimestamp(last, timezone.utc))

# Read official UNSW-NB15 CSV
print()
print("Reading official CSV...")

df = pd.read_csv(
    CSV,
    header=None,
    usecols=[0, 1, 2, 3, 4, 28, 29, 47, 48],
    names=[
        "srcip", "sport", "dstip", "dsport",
        "proto", "stime", "ltime",
        "attack_cat", "label"
    ],
    on_bad_lines="skip",
    low_memory=False
)

df["stime"] = pd.to_numeric(df["stime"], errors="coerce")
df["ltime"] = pd.to_numeric(df["ltime"], errors="coerce")
df["label"] = pd.to_numeric(df["label"], errors="coerce")

# Find records whose time interval overlaps this PCAP
candidate = df[
    (df["stime"] <= last) &
    (df["ltime"] >= first)
].copy()

attacks = candidate[candidate["label"] == 1]
normal = candidate[candidate["label"] == 0]

print()
print("=" * 60)
print("RESULT")
print("=" * 60)

print("Candidate records:", len(candidate))
print("Normal records:", len(normal))
print("Attack records:", len(attacks))

print()
print("Attack categories:")

if len(attacks) > 0:
    print(attacks["attack_cat"].value_counts())
else:
    print("NO ATTACK RECORDS FOUND")

print("=" * 60)