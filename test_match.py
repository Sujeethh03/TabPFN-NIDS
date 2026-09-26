import pandas as pd

flows = pd.read_parquet(
    "data/intermediate/features/all_features.parquet"
).head(20)

path = "data/raw/UNSW-NB15/UNSW-NB15_1.csv"

cols = [0, 1, 2, 3, 4, 28, 29, 47, 48]

gt = pd.read_csv(
    path,
    header=None,
    usecols=cols,
    names=[
        "src_ip",
        "src_port",
        "dst_ip",
        "dst_port",
        "protocol",
        "start_time",
        "end_time",
        "attack_cat",
        "label",
    ],
    low_memory=False,
)

protocol_map = {
    1: "icmp",
    6: "tcp",
    17: "udp",
}

flows["protocol_name"] = flows["protocol"].map(protocol_map)

print("Investigating 20 flows...")
print()

for i, flow in flows.iterrows():

    proto = str(flow["protocol_name"]).lower()

    direct = (
        (gt["src_ip"].astype(str) == str(flow["src_ip"]))
        & (gt["dst_ip"].astype(str) == str(flow["dst_ip"]))
        & (gt["src_port"].astype(str) == str(flow["src_port"]))
        & (gt["dst_port"].astype(str) == str(flow["dst_port"]))
        & (gt["protocol"].astype(str).str.lower() == proto)
    )

    reverse = (
        (gt["src_ip"].astype(str) == str(flow["dst_ip"]))
        & (gt["dst_ip"].astype(str) == str(flow["src_ip"]))
        & (gt["src_port"].astype(str) == str(flow["dst_port"]))
        & (gt["dst_port"].astype(str) == str(flow["src_port"]))
        & (gt["protocol"].astype(str).str.lower() == proto)
    )

    candidates = gt[direct | reverse].copy()

    if len(candidates) == 0:
        print(f"{i}: NO 5-TUPLE MATCH")
        print(
            f"   {flow['src_ip']}:{flow['src_port']} -> "
            f"{flow['dst_ip']}:{flow['dst_port']} "
            f"{proto}"
        )
        print()
        continue

    candidates["time_diff"] = (
        candidates["start_time"] - flow["start_time"]
    ).abs()

    candidates = candidates.sort_values("time_diff")

    best = candidates.iloc[0]

    print(f"{i}: 5-TUPLE MATCH")
    print(
        f"   Flow time : {flow['start_time']:.3f}"
    )
    print(
        f"   CSV time  : {best['start_time']}"
    )
    print(
        f"   Difference: {best['time_diff']:.3f} seconds"
    )
    print(
        f"   Label     : {best['label']}"
    )
    print(
        f"   Category  : {best['attack_cat']}"
    )
    print()