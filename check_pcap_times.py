from scapy.all import PcapReader
from datetime import datetime, timezone
import os

base = "data/raw/UNSW-NB15"

files = [
    "23.pcap",
    "24.pcap",
    "25.pcap",
    "26.pcap",
    "27.pcap",
]

for filename in files:

    path = os.path.join(base, filename)

    first = None
    last = None
    count = 0

    with PcapReader(path) as reader:

        for packet in reader:

            ts = float(packet.time)

            if first is None:
                first = ts

            last = ts
            count += 1

    print()
    print(filename)
    print("Packets:", count)
    print(
        "First:",
        datetime.fromtimestamp(first, timezone.utc)
    )
    print(
        "Last :",
        datetime.fromtimestamp(last, timezone.utc)
    )