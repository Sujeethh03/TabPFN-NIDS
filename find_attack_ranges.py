import pandas as pd
import os

base = "data/raw/UNSW-NB15"

files = [
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv",
]

print()
print("UNSW-NB15 ATTACK TIME RANGES")
print("=" * 100)

for filename in files:

    path = os.path.join(base, filename)

    df = pd.read_csv(
        path,
        header=None,
        usecols=[28, 29, 47, 48],
        names=[
            "start_time",
            "end_time",
            "attack_cat",
            "label",
        ],
        low_memory=False,
    )

    attacks = df[df["label"] == 1].copy()

    print()
    print(filename)
    print("-" * 100)

    print("Attack records:", len(attacks))
    print("First attack :", attacks["start_time"].min())
    print("Last attack  :", attacks["end_time"].max())

    print()
    print("Attack categories:")
    print(attacks["attack_cat"].value_counts())

    print()
    print("First 10 attack time ranges:")
    print(
        attacks[
            ["start_time", "end_time", "attack_cat"]
        ].head(10).to_string(index=False)
    )

print()
print("=" * 100)