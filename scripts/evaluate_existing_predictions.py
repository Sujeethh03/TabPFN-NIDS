from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from tabpfn_nids.labeling.unsw_ground_truth import (
    load_ground_truth_csv,
    match_flows_to_gt_csv,
)


# ============================================================
# Paths
# ============================================================

PREDICTIONS = Path(
    "results/pcap_analysis/11/11_predictions.csv"
)

GT_FILES = [
    Path("data/raw/UNSW-NB15/UNSW-NB15_1.csv"),
    Path("data/raw/UNSW-NB15/UNSW-NB15_2.csv"),
    Path("data/raw/UNSW-NB15/UNSW-NB15_3.csv"),
    Path("data/raw/UNSW-NB15/UNSW-NB15_4.csv"),
]

OUTPUT = Path(
    "results/pcap_analysis/11/11_evaluation.csv"
)


# ============================================================
# Load predictions
# ============================================================

print("=" * 70)
print("Loading existing predictions")
print("=" * 70)

pred = pd.read_csv(PREDICTIONS)

print(f"Prediction rows: {len(pred):,}")
print(f"Prediction columns: {len(pred.columns)}")


# ============================================================
# Validate required prediction columns
# ============================================================

required_prediction_columns = [
    "flow_id",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "start_time",
    "prediction",
    "prediction_label",
    "normal_probability",
    "attack_probability",
]

missing = [
    c for c in required_prediction_columns
    if c not in pred.columns
]

if missing:
    raise ValueError(
        f"Prediction CSV is missing required columns: {missing}"
    )


# ============================================================
# Load all official UNSW-NB15 raw flow files
# ============================================================

print()
print("=" * 70)
print("Loading official UNSW-NB15 ground truth")
print("=" * 70)

gt_parts = []

for path in GT_FILES:

    print(f"Loading: {path}")

    if not path.is_file():
        raise FileNotFoundError(
            f"Ground-truth file not found: {path}"
        )

    gt = load_ground_truth_csv(path)

    print(
        f"  Loaded {len(gt):,} official flow records"
    )

    gt_parts.append(gt)


ground_truth = pd.concat(
    gt_parts,
    ignore_index=True,
)

print()
print(
    f"Total official GT records: "
    f"{len(ground_truth):,}"
)


# ============================================================
# Match predictions/flows to official GT
# ============================================================

print()
print("=" * 70)
print("Matching PCAP flows to official ground truth")
print("=" * 70)

matched = match_flows_to_gt_csv(
    pred,
    ground_truth,
    time_tolerance=5.0,
    use_ip_port=True,
)


# ============================================================
# Match statistics
# ============================================================

matched_mask = matched["label"] != -1

matched_count = int(matched_mask.sum())
unmatched_count = int((~matched_mask).sum())

print()
print("=" * 70)
print("MATCHING RESULTS")
print("=" * 70)

print(f"Total predicted flows : {len(matched):,}")
print(f"Matched flows         : {matched_count:,}")
print(f"Unmatched flows       : {unmatched_count:,}")

if len(matched):
    print(
        f"Match rate            : "
        f"{matched_count / len(matched) * 100:.2f}%"
    )


# ============================================================
# Save matched evaluation data
# ============================================================

matched.to_csv(
    OUTPUT,
    index=False,
)

print()
print(f"Evaluation CSV saved to:")
print(OUTPUT)


# ============================================================
# Stop if nothing matched
# ============================================================

if matched_count == 0:

    print()
    print("WARNING: No flows matched the official GT.")
    print("Metrics cannot be calculated.")

    raise SystemExit(0)


# ============================================================
# Prepare labels
# ============================================================

evaluation = matched.loc[
    matched_mask
].copy()

y_true = pd.to_numeric(
    evaluation["label"],
    errors="coerce",
).astype(int)

# prediction is the model's numeric binary prediction
y_pred = pd.to_numeric(
    evaluation["prediction"],
    errors="coerce",
).astype(int)


# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0,
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0,
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0,
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1],
)


# ============================================================
# Display results
# ============================================================

print()
print("=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print(f"Evaluated flows : {len(evaluation):,}")
print(f"Accuracy        : {accuracy:.4f}")
print(f"Precision       : {precision:.4f}")
print(f"Recall          : {recall:.4f}")
print(f"F1 Score        : {f1:.4f}")

print()
print("Confusion Matrix")
print()
print("                  Predicted")
print("                 Normal Attack")
print(
    f"Actual Normal    {cm[0,0]:7d} {cm[0,1]:6d}"
)
print(
    f"Actual Attack    {cm[1,0]:7d} {cm[1,1]:6d}"
)


# ============================================================
# Ground-truth distribution
# ============================================================

print()
print("=" * 70)
print("GROUND-TRUTH DISTRIBUTION")
print("=" * 70)

print(
    evaluation["label"]
    .value_counts()
    .sort_index()
    .rename(
        index={
            0: "Normal",
            1: "Attack",
        }
    )
)


# ============================================================
# Prediction distribution
# ============================================================

print()
print("=" * 70)
print("MODEL PREDICTION DISTRIBUTION")
print("=" * 70)

print(
    evaluation["prediction_label"]
    .value_counts()
)


# ============================================================
# Attack categories
# ============================================================

print()
print("=" * 70)
print("GROUND-TRUTH ATTACK CATEGORIES")
print("=" * 70)

print(
    evaluation["attack_cat"]
    .value_counts()
)


# ============================================================
# Classification report
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        zero_division=0,
    )
)


print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)