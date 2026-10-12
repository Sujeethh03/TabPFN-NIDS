import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from tabpfn_nids.models.tabpfn_model import TabPFNModel


ROOT = Path.cwd()

CLEANED_PATH = (
    ROOT / "data" / "intermediate" / "labels" / "cleaned_flows.parquet"
)

PREPROCESSED_PATH = (
    ROOT / "data" / "intermediate" / "labels" / "preprocessed_flows.parquet"
)

TRAIN_PATH = (
    ROOT / "data" / "processed" / "binary" / "train.parquet"
)

MODEL_PATH = (
    ROOT / "data" / "artifacts" / "models" / "tabpfn_binary_model.pkl"
)

SCALER_PATH = (
    ROOT / "data" / "artifacts" / "preprocessing" / "fitted_scaler.pkl"
)

SCHEMA_PATH = (
    ROOT / "data" / "artifacts" / "models" / "model_feature_schema.json"
)


# =========================================================
# Project metadata columns
# =========================================================

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
    metadata_columns
    + [
        "label",
        "attack_cat",
        "match_type",
    ]
)


print("=" * 70)
print("CREATING CORRECT TABPFN INFERENCE ARTIFACTS")
print("=" * 70)


# =========================================================
# 1. Load cleaned/raw project data
# =========================================================

print("\n[1/7] Loading cleaned flows...")

cleaned = pd.read_parquet(CLEANED_PATH)

print("Cleaned rows   :", len(cleaned))
print("Cleaned columns:", len(cleaned.columns))


# =========================================================
# 2. Determine numeric model features
# =========================================================

numeric_columns = [
    c
    for c in cleaned.select_dtypes(include=[np.number]).columns
    if c not in metadata_columns
    and c not in ["label"]
]

print("\nNumeric features:", len(numeric_columns))


# =========================================================
# 3. Determine final model feature names
# =========================================================

train = pd.read_parquet(TRAIN_PATH)

feature_names = [
    c
    for c in train.columns
    if c not in excluded_columns
]

print("Model features :", len(feature_names))

if len(feature_names) != 67:
    raise RuntimeError(
        f"Expected 67 model features, found {len(feature_names)}"
    )


# =========================================================
# 4. Fit scaler on CLEANED data
# =========================================================

print("\n[2/7] Fitting RobustScaler on CLEANED data...")

scaler = RobustScaler()

scaler.fit(
    cleaned[numeric_columns]
    .astype(np.float64)
)

print(
    "Scaler fitted on",
    len(numeric_columns),
    "numeric columns"
)


# =========================================================
# 5. Verify preprocessing against existing
#    preprocessed_flows.parquet
# =========================================================

print("\n[3/7] Verifying preprocessing...")

preprocessed = pd.read_parquet(PREPROCESSED_PATH)

check_columns = [
    c
    for c in feature_names
    if c in numeric_columns
]

raw_values = (
    cleaned[check_columns]
    .astype(np.float64)
)

expected_values = scaler.transform(raw_values)

actual_values = (
    preprocessed[check_columns]
    .astype(np.float64)
    .to_numpy()
)

max_difference = np.max(
    np.abs(expected_values - actual_values)
)

print(
    "Maximum difference between",
    "reconstructed preprocessing and",
    "existing preprocessed data:",
    max_difference
)

if max_difference > 1e-6:
    print(
        "\nWARNING: preprocessing does not exactly match "
        "preprocessed_flows.parquet."
    )
    print(
        "The project preprocessing may contain another transformation."
    )
else:
    print("Preprocessing verification: PASSED")


# =========================================================
# 6. Train TabPFN directly on EXISTING preprocessed train
# =========================================================

print("\n[4/7] Preparing TabPFN training data...")

X_train = (
    train[feature_names]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
    .to_numpy(dtype=np.float64)
)

y_train = (
    train["label"]
    .astype(int)
    .to_numpy()
)

print("Training matrix:", X_train.shape)
print("Normal samples :", int((y_train == 0).sum()))
print("Attack samples :", int((y_train == 1).sum()))


# IMPORTANT:
# DO NOT scale X_train here.
#
# train.parquet is already preprocessed.
# Applying RobustScaler again would recreate the bug
# that caused the PCAP probability mismatch.

print("\n[5/7] Training TabPFN...")
print("No additional scaler applied to train.parquet.")

model = TabPFNModel(
    task="binary",
    max_context_samples=2000,
    device="auto",
    n_estimators=2,
    random_state=42,
    use_chunked_ensemble=True,
    predict_batch_size=1000,
)

model.fit(X_train, y_train)


# =========================================================
# 7. Save artifacts
# =========================================================

print("\n[6/7] Saving artifacts...")

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

SCALER_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(
        model,
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

with open(SCALER_PATH, "wb") as f:
    pickle.dump(
        scaler,
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )


schema = {
    "feature_names": feature_names,
    "n_features": len(feature_names),
    "task": "binary",
    "scaler": "RobustScaler",
    "scaler_fit_source": "data/intermediate/labels/cleaned_flows.parquet",
    "training_source": "data/processed/binary/train.parquet",
    "max_context_samples": 2000,
    "n_estimators": 2,
    "use_chunked_ensemble": True,
    "random_state": 42,
}

with open(
    SCHEMA_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        schema,
        f,
        indent=2,
    )


# =========================================================
# 8. Verify everything
# =========================================================

print("\n[7/7] Verifying artifacts...")

with open(MODEL_PATH, "rb") as f:
    loaded_model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    loaded_scaler = pickle.load(f)

with open(
    SCHEMA_PATH,
    "r",
    encoding="utf-8",
) as f:
    loaded_schema = json.load(f)


print("\n" + "=" * 70)
print("ARTIFACT VERIFICATION")
print("=" * 70)

print(
    "Model:",
    MODEL_PATH,
    "| exists =",
    MODEL_PATH.exists(),
)

print(
    "Scaler:",
    SCALER_PATH,
    "| exists =",
    SCALER_PATH.exists(),
)

print(
    "Schema:",
    SCHEMA_PATH,
    "| exists =",
    SCHEMA_PATH.exists(),
)

print(
    "\nLoaded model :",
    type(loaded_model).__name__,
)

print(
    "Loaded scaler:",
    type(loaded_scaler).__name__,
)

print(
    "Schema features:",
    len(loaded_schema["feature_names"]),
)

assert len(feature_names) == 67
assert loaded_schema["feature_names"] == feature_names

print("\nSUCCESS")
print("Correct inference artifacts created.")
print("=" * 70)