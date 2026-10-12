import pickle
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd()

MODEL_PATH = ROOT / "data" / "artifacts" / "models" / "tabpfn_binary_model.pkl"
SCALER_PATH = ROOT / "data" / "artifacts" / "preprocessing" / "fitted_scaler.pkl"
TEST_PATH = ROOT / "data" / "processed" / "binary" / "test.parquet"


metadata_columns = [
    "flow_id",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "protocol_name",
    "start_time",
    "end_time",
]

excluded_columns = set(
    metadata_columns + [
        "label",
        "attack_cat",
        "match_type",
    ]
)


print("=" * 70)
print("Evaluating inference artifact on TEST dataset")
print("=" * 70)

# ---------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

print("Model :", type(model).__name__)
print("Scaler:", type(scaler).__name__)


# ---------------------------------------------------------
# Load test data
# ---------------------------------------------------------

df = pd.read_parquet(TEST_PATH)

feature_names = [
    c for c in df.columns
    if c not in excluded_columns
]

print("\nTest rows:", len(df))
print("Features :", len(feature_names))

y_true = df["label"].astype(int).to_numpy()

X = df[feature_names].copy()
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

X = X.to_numpy(dtype=np.float64)

# ---------------------------------------------------------
# Apply same scaler
# ---------------------------------------------------------

X_scaled = scaler.transform(X)

print("Scaled shape:", X_scaled.shape)


# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------

print("\nRunning predictions...")

probabilities = model.predict_proba(X_scaled)

y_pred = np.argmax(probabilities, axis=1)

attack_probability = probabilities[:, 1]


# ---------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, zero_division=0))
print("Recall   :", recall_score(y_true, y_pred, zero_division=0))
print("F1       :", f1_score(y_true, y_pred, zero_division=0))
print("ROC-AUC  :", roc_auc_score(y_true, attack_probability))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Normal", "Attack"],
    zero_division=0,
))

print("\nProbability statistics:")
print("Minimum attack probability :", attack_probability.min())
print("Maximum attack probability :", attack_probability.max())
print("Mean attack probability    :", attack_probability.mean())

print("\nPredicted class counts:")
print(pd.Series(y_pred).map({
    0: "Normal",
    1: "Attack",
}).value_counts())
