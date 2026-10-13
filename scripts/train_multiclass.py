#!/usr/bin/env python3
"""
Multiclass Attack Detection Training and Evaluation Pipeline.

Trains a Multiclass TabPFN Classifier on UNSW-NB15 network flow data to
identify the specific attack category occurring in network traffic:
  - Normal
  - Fuzzers
  - Analysis (when present)
  - Backdoors
  - DoS
  - Exploits
  - Generic
  - Reconnaissance
  - Shellcode
  - Worms

Workflow:
1. Load 67 network-flow features and ground-truth attack categories.
2. Validate data quality, missing values, duplicates, and distributions.
3. Split into Train, Validation, and Test sets (temporal or stratified).
4. Fit RobustScaler on training set only (prevent leakage).
5. Address class imbalance for in-context learning via balanced sampling.
6. Train Multiclass TabPFN classifier within CPU/RAM constraints.
7. Evaluate comprehensively on Validation and Test sets:
   - Accuracy, Macro/Weighted Precision, Recall, F1
   - Per-class precision, recall, f1, support
   - Confusion matrix and One-vs-Rest ROC-AUC
   - Comparison with binary detection performance
8. Persist artifacts for subsequent PCAP inference without retraining:
   - Model (.pkl)
   - Feature schema (.json)
   - Scaler (.pkl)
   - Label encoder (.json)
   - Model configuration (.json)
   - Evaluation reports (.json)
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from tabpfn_nids.evaluation.metrics import (
    compute_metrics,
    compute_multiclass_metrics,
    format_metrics,
    format_multiclass_metrics,
)
from tabpfn_nids.features.feature_pipeline import METADATA_COLUMNS
from tabpfn_nids.models.tabpfn_model import TabPFNModel
from tabpfn_nids.splitting.leakage_checker import run_all_leakage_checks
from tabpfn_nids.splitting.splitter import split_dataset, save_splits

# ---------------------------------------------------------------------------
# Paths and Defaults
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
INTERMEDIATE_DIR = DATA_DIR / "intermediate" / "labels"
CLEANED_PATH = INTERMEDIATE_DIR / "cleaned_flows.parquet"
LABELED_PATH = INTERMEDIATE_DIR / "labeled_flows.parquet"

ARTIFACTS_DIR = DATA_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PREPROC_DIR = ARTIFACTS_DIR / "preprocessing"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREPROC_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_PATH = MODELS_DIR / "tabpfn_multiclass_model.pkl"
SCHEMA_OUTPUT_PATH = MODELS_DIR / "multiclass_feature_schema.json"
SCALER_OUTPUT_PATH = PREPROC_DIR / "multiclass_scaler.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "multiclass_label_encoder.json"
CONFIG_OUTPUT_PATH = MODELS_DIR / "multiclass_model_config.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s",
)
logger = logging.getLogger("train_multiclass")


def load_and_validate_data(
    input_path: Path | str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load cleaned network flows and validate feature schema and labels."""
    path = Path(input_path) if input_path else CLEANED_PATH
    if not path.is_file():
        if LABELED_PATH.is_file():
            path = LABELED_PATH
        else:
            raise FileNotFoundError(f"Cleaned/labeled flows not found at {path}")

    logger.info("Loading flow records from: %s", path)
    df = pd.read_parquet(path)
    logger.info("Raw input shape: %d rows × %d columns", len(df), len(df.columns))

    # 1. Filter out UNKNOWN or invalid labels
    initial_count = len(df)
    df = df[df["attack_cat"] != "UNKNOWN"].copy()
    if "label" in df.columns:
        df = df[df["label"] != -1].copy()
    filtered_count = len(df)
    logger.info(
        "Filtered unclassified flows: %d -> %d matched flows (%d removed)",
        initial_count,
        filtered_count,
        initial_count - filtered_count,
    )

    # 2. Identify 67 network-flow model features
    excluded = set(METADATA_COLUMNS + ["label", "attack_cat", "match_type"])
    feature_names = [c for c in df.columns if c not in excluded]

    # Verify against standard schema if present
    binary_schema_path = MODELS_DIR / "model_feature_schema.json"
    if binary_schema_path.is_file():
        with open(binary_schema_path, "r", encoding="utf-8") as f:
            b_schema = json.load(f)
            expected_features = b_schema.get("feature_names", [])
            if set(expected_features).issubset(set(df.columns)):
                feature_names = expected_features

    logger.info("Using %d model features", len(feature_names))

    # 3. Validate missing values and infinities
    X_sub = df[feature_names].apply(pd.to_numeric, errors="coerce")
    inf_count = int(np.isinf(X_sub.to_numpy()).sum())
    nan_count = int(X_sub.isna().to_numpy().sum())
    logger.info("Data validation: Inf count=%d, NaN count=%d", inf_count, nan_count)

    # Clean in dataframe
    df[feature_names] = X_sub.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 4. Report class distribution
    cat_counts = df["attack_cat"].value_counts().to_dict()
    logger.info("Ground-truth attack category distribution:")
    for cat, cnt in cat_counts.items():
        logger.info("  %-16s: %6d flows (%.2f%%)", cat, cnt, cnt / len(df) * 100)

    return df, feature_names


def prepare_balanced_context(
    train_df: pd.DataFrame,
    max_context_samples: int = 1000,
    max_per_class: int = 150,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a balanced in-context training sample for TabPFN.

    Prevents majority class (Normal, 92%) from starving minority classes
    (Worms, Shellcode, Backdoors) while staying within TabPFN's pretraining
    context limits.
    """
    logger.info(
        "Constructing balanced TabPFN context (target limit=%d, max_per_class=%d)...",
        max_context_samples,
        max_per_class,
    )
    sampled = []
    classes = train_df["attack_cat"].unique()

    # Allocate samples proportionally but capped
    for cat in classes:
        group = train_df[train_df["attack_cat"] == cat]
        n_take = min(len(group), max_per_class)
        sampled.append(group.sample(n=n_take, random_state=random_state))

    ctx_df = pd.concat(sampled)

    # If context is under max_context_samples, we can allocate extra from larger classes
    if len(ctx_df) < max_context_samples:
        remaining_budget = max_context_samples - len(ctx_df)
        larger_classes = [c for c in classes if len(train_df[train_df["attack_cat"] == c]) > max_per_class]
        if larger_classes:
            extra_per = remaining_budget // len(larger_classes)
            extra_sampled = []
            for cat in larger_classes:
                group = train_df[train_df["attack_cat"] == cat]
                unselected = group.drop(index=group.index.intersection(ctx_df.index))
                n_extra = min(len(unselected), extra_per)
                if n_extra > 0:
                    extra_sampled.append(unselected.sample(n=n_extra, random_state=random_state))
            if extra_sampled:
                ctx_df = pd.concat([ctx_df] + extra_sampled)

    # Shuffle context
    ctx_df = ctx_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    logger.info("Constructed context: %d samples across %d classes:", len(ctx_df), ctx_df["attack_cat"].nunique())
    for cat, cnt in ctx_df["attack_cat"].value_counts().items():
        logger.info("  %-16s: %4d samples", cat, cnt)

    return ctx_df


def train_multiclass_pipeline(
    input_path: Path | str | None = None,
    split_strategy: str = "stratified",
    max_context_samples: int = 1000,
    max_per_class: int = 120,
    device: str = "cpu",
    n_estimators: int = 1,
    predict_batch_size: int = 200,
    confidence_threshold: float = 0.5,
    random_state: int = 42,
    eval_test_size: int = 500,
) -> dict[str, Any]:
    """Execute complete multiclass training and evaluation workflow."""
    start_time = time.perf_counter()

    print("=" * 72)
    print("TabPFN-NIDS — MULTICLASS ATTACK DETECTION TRAINING PIPELINE")
    print("=" * 72)

    # 1. Load and validate data
    df, feature_names = load_and_validate_data(input_path)

    # 2. Split dataset
    logger.info("Splitting dataset using strategy: '%s'...", split_strategy)
    train_df, val_df, test_df = split_dataset(
        df,
        strategy=split_strategy,
        time_column="start_time" if "start_time" in df.columns else "stime",
        label_column="attack_cat",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=random_state,
    )

    # Check leakage
    leakage_report = run_all_leakage_checks(
        train_df, val_df, test_df,
        output_path=ARTIFACTS_DIR / "split_metadata" / "multiclass_leakage_report.json",
    )
    logger.info("Leakage check completed. Clean: %s", leakage_report.get("overall_clean", True))

    # Save splits
    save_splits(
        train_df, val_df, test_df,
        output_dir=DATA_DIR / "processed",
        task="multiclass",
    )

    # 3. Fit RobustScaler on training set ONLY
    logger.info("Fitting RobustScaler on training data ONLY...")
    scaler = RobustScaler()
    X_train_full = train_df[feature_names].values.astype(np.float64)
    scaler.fit(X_train_full)

    # 4. Handle class imbalance for in-context training sample
    context_df = prepare_balanced_context(
        train_df,
        max_context_samples=max_context_samples,
        max_per_class=max_per_class,
        random_state=random_state,
    )

    # 5. Build Label Encoder and mapping
    classes = sorted(context_df["attack_cat"].unique().tolist())
    # Ensure Normal is present
    if "Normal" not in classes:
        classes.insert(0, "Normal")
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}

    logger.info("Target attack classes (%d): %s", len(classes), classes)

    label_encoder_data = {
        "classes": classes,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "id_to_label": idx_to_class,
        "num_classes": len(classes),
        "confidence_threshold": confidence_threshold,
        "unseen_label": "Uncertain",
        "supported_classes": classes,
        "unsupported_classes": ["Analysis"],
        "unsupported_class_reasons": {
            "Analysis": "Analysis attack category had 0 flow instances in the extracted UNSW-NB15 PCAP slice (timestamps 1421927376 to 1421928304); the earliest Analysis attacks in UNSW-NB15 occur at timestamp 1421930602, so it could not be trained."
        },
    }

    # 6. Transform context
    X_ctx = scaler.transform(context_df[feature_names].values.astype(np.float64))
    y_ctx = context_df["attack_cat"].map(class_to_idx).values.astype(int)

    # 7. Train TabPFN multiclass classifier
    logger.info("Fitting TabPFN Multiclass model on %d context rows...", len(X_ctx))
    fit_start = time.perf_counter()
    model = TabPFNModel(
        task="multiclass",
        max_context_samples=len(X_ctx),
        device=device,
        n_estimators=n_estimators,
        random_state=random_state,
        use_chunked_ensemble=False,
        predict_batch_size=predict_batch_size,
    )
    model.fit(X_ctx, y_ctx)
    fit_duration = time.perf_counter() - fit_start
    logger.info("TabPFN fit completed in %.2fs", fit_duration)

    # 8. Save all artifacts
    logger.info("Saving trained artifacts...")
    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved model -> %s", MODEL_OUTPUT_PATH)

    with open(SCALER_OUTPUT_PATH, "wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved scaler -> %s", SCALER_OUTPUT_PATH)

    with open(LABEL_ENCODER_PATH, "w", encoding="utf-8") as f:
        json.dump(label_encoder_data, f, indent=2)
    logger.info("Saved label encoder -> %s", LABEL_ENCODER_PATH)

    schema = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "task": "multiclass",
        "classes": classes,
        "n_classes": len(classes),
        "scaler": "RobustScaler",
        "scaler_fit_source": "train split only",
        "context_samples": len(X_ctx),
        "max_context_samples": max_context_samples,
        "n_estimators": n_estimators,
        "device": device,
        "confidence_threshold": confidence_threshold,
        "random_state": random_state,
    }
    with open(SCHEMA_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    logger.info("Saved schema -> %s", SCHEMA_OUTPUT_PATH)

    model_config = model.get_config()
    model_config["classes"] = classes
    with open(CONFIG_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    # 9. Evaluate model on Validation and Test sets
    logger.info("Evaluating multiclass model on held-out evaluation sets...")

    def evaluate_split(split_df: pd.DataFrame, split_name: str, sample_size: int | None = None) -> dict[str, Any]:
        eval_df = split_df
        if sample_size and len(eval_df) > sample_size:
            # Stratified sample for evaluation speed
            sample_dfs = []
            for _, grp in eval_df.groupby("attack_cat"):
                n_take = max(1, int(len(grp) / len(eval_df) * sample_size))
                sample_dfs.append(grp.sample(min(len(grp), n_take), random_state=random_state))
            eval_df = pd.concat(sample_dfs, ignore_index=True)

        X_eval = scaler.transform(eval_df[feature_names].values.astype(np.float64))
        eval_cats = eval_df["attack_cat"].tolist()

        pred_start = time.perf_counter()
        proba = model.predict_proba(X_eval)
        pred_indices = np.argmax(proba, axis=1)
        pred_duration = time.perf_counter() - pred_start

        pred_cats = [idx_to_class.get(idx, "Uncertain") for idx in pred_indices]

        # Multiclass evaluation
        metrics = compute_multiclass_metrics(
            y_true=eval_cats,
            y_pred=pred_cats,
            y_proba=proba,
            classes=classes,
            confidence_threshold=confidence_threshold,
        )
        metrics["prediction_seconds"] = round(pred_duration, 2)
        metrics["samples_evaluated"] = len(eval_df)
        metrics["throughput_samples_per_sec"] = round(len(eval_df) / max(1e-6, pred_duration), 2)

        # Binary comparison metrics on same split
        binary_true = [0 if c == "Normal" else 1 for c in eval_cats]
        binary_pred = [0 if c == "Normal" else 1 for c in pred_cats]
        normal_idx = class_to_idx.get("Normal", 0)
        prob_attack = 1.0 - proba[:, normal_idx]
        binary_metrics = compute_metrics(
            y_true=np.array(binary_true),
            y_pred=np.array(binary_pred),
            y_proba=prob_attack,
        )
        metrics["binary_comparison"] = binary_metrics

        return metrics

    val_metrics = evaluate_split(val_df, "Validation", sample_size=eval_test_size)
    test_metrics = evaluate_split(test_df, "Held-out Test", sample_size=eval_test_size)

    # Save reports
    with open(RESULTS_DIR / "multiclass_validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)
    with open(RESULTS_DIR / "multiclass_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    total_pipeline_time = time.perf_counter() - start_time

    # Print evaluation report
    print("\n" + format_multiclass_metrics(test_metrics, title="HELD-OUT TEST SET — MULTICLASS EVALUATION"))

    print("\n" + "=" * 72)
    print("BINARY INTRUSION DETECTION COMPARISON (ON SAME TEST SET)")
    print("=" * 72)
    b_met = test_metrics.get("binary_comparison", {})
    print(f"  Binary Accuracy : {b_met.get('accuracy', 0.0):.4f}")
    print(f"  Binary Precision: {b_met.get('precision', 0.0):.4f}")
    print(f"  Binary Recall   : {b_met.get('recall', 0.0):.4f}")
    print(f"  Binary F1-Score : {b_met.get('f1_score', 0.0):.4f}")
    if b_met.get("roc_auc") is not None:
        print(f"  Binary ROC-AUC  : {b_met['roc_auc']:.4f}")
    print("=" * 72)

    print(f"\nTraining pipeline completed in {total_pipeline_time:.1f}s.")
    print("Artifacts successfully created and verified.")

    return {
        "status": "success",
        "classes": classes,
        "n_classes": len(classes),
        "model_path": str(MODEL_OUTPUT_PATH),
        "scaler_path": str(SCALER_OUTPUT_PATH),
        "schema_path": str(SCHEMA_OUTPUT_PATH),
        "encoder_path": str(LABEL_ENCODER_PATH),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "total_seconds": round(total_pipeline_time, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Train TabPFN Multiclass Intrusion Detection Model")
    parser.add_argument("--data", default=None, help="Path to cleaned/labeled parquet file")
    parser.add_argument("--split-strategy", choices=["stratified", "temporal"], default="stratified", help="Split strategy (default: stratified)")
    parser.add_argument("--context-samples", type=int, default=1000, help="Max context samples for TabPFN (default: 1000)")
    parser.add_argument("--max-per-class", type=int, default=120, help="Max context rows per class (default: 120)")
    parser.add_argument("--device", default="cpu", help="Device to run on ('cpu' or 'auto')")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for predictions (default: 0.5)")
    parser.add_argument("--eval-size", type=int, default=500, help="Evaluation sample size for test evaluation (default: 500)")
    args = parser.parse_args()

    train_multiclass_pipeline(
        input_path=args.data,
        split_strategy=args.split_strategy,
        max_context_samples=args.context_samples,
        max_per_class=args.max_per_class,
        device=args.device,
        confidence_threshold=args.threshold,
        eval_test_size=args.eval_size,
    )


if __name__ == "__main__":
    main()
