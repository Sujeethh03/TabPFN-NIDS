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
print("UNSW-NB15 CSV SUMMARY")
print("=" * 90)
print(
    f"{'FILE':28} "
    f"{'ROWS':12} "
    f"{'ATTACKS':12} "
    f"{'FIRST_TIME':17} "
    f"{'LAST_TIME':17}"
)
print("-" * 90)

for filename in files:

    path = os.path.join(base, filename)

    df = pd.read_csv(
        path,
        header=None,
        usecols=[28, 29, 48],
        names=["start", "end", "label"],
        low_memory=False,
    )

    rows = len(df)
    attacks = int((df["label"] == 1).sum())
    first_time = int(df["start"].min())
    last_time = int(df["end"].max())

    print(
        f"{filename:28} "
        f"{rows:<12} "
        f"{attacks:<12} "
        f"{first_time:<17} "
        f"{last_time:<17}"
    )

print("=" * 90)