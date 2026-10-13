"""
Analyze a new PCAP using the existing trained TabPFN-NIDS model.

Workflow:

PCAP
  ↓
Flow extraction
  ↓
Feature engineering
  ↓
Existing preprocessing artifacts
  ↓
Existing trained TabPFN model
  ↓
Predictions
  ↓
Prediction CSV
  ↓
Optional ground-truth evaluation
  ↓
JSON + HTML analysis report
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tabpfn_nids.flows.flow_builder import (
    extract_flows_from_pcap,
)

from tabpfn_nids.features.feature_pipeline import (
    compute_all_features,
)

from tabpfn_nids.labeling.unsw_ground_truth import (
    load_ground_truth_csv,
    match_flows_to_gt_csv,
)

from tabpfn_nids.evaluation.metrics import (
    compute_metrics,
    compute_multiclass_metrics,
    decode_attack_category,
    format_multiclass_metrics,
    get_attack_category_mapping,
)

from tabpfn_nids.pcap.analysis_report import (
    build_report_data,
    save_json_report,
    save_html_report,
    save_docx_report,
)

from tabpfn_nids.pipeline_config import (
    load_config,
)

from tabpfn_nids.inference import (
    DynamicInferenceManager,
    ModelRegistry,
    ModelInfo,
)


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

ARTIFACTS_DIR = DATA_DIR / "artifacts"

MODEL_PATH = (
    ARTIFACTS_DIR
    / "models"
    / "tabpfn_binary_model.pkl"
)

FEATURE_SCHEMA_PATH = (
    ARTIFACTS_DIR
    / "models"
    / "model_feature_schema.json"
)

SCALER_PATH = (
    ARTIFACTS_DIR
    / "preprocessing"
    / "fitted_scaler.pkl"
)

MULTICLASS_MODEL_PATH = (
    ARTIFACTS_DIR
    / "models"
    / "tabpfn_multiclass_model.pkl"
)

MULTICLASS_SCHEMA_PATH = (
    ARTIFACTS_DIR
    / "models"
    / "multiclass_feature_schema.json"
)

MULTICLASS_SCALER_PATH = (
    ARTIFACTS_DIR
    / "preprocessing"
    / "multiclass_scaler.pkl"
)

MULTICLASS_LABEL_ENCODER_PATH = (
    ARTIFACTS_DIR
    / "models"
    / "multiclass_label_encoder.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "pcap_analysis"
)


# =========================================================
# Logging
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# =========================================================
# Load trained artifacts
# =========================================================

def load_artifacts(include_multiclass: bool = False):
    """Load binary model (and optionally multiclass model) and preprocessing artifacts."""

    logger.info("Loading binary TabPFN model...")

    with open(MODEL_PATH, "rb") as f:
        binary_model = pickle.load(f)

    logger.info("Loading feature schema...")

    with open(
        FEATURE_SCHEMA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        schema = json.load(f)

    feature_names = schema["feature_names"]

    logger.info("Loading fitted binary scaler...")

    with open(SCALER_PATH, "rb") as f:
        binary_scaler = pickle.load(f)

    if not include_multiclass:
        return binary_model, binary_scaler, feature_names

    multiclass_model = None
    multiclass_scaler = None
    multiclass_label_encoder = None

    if MULTICLASS_MODEL_PATH.exists():
        logger.info("Loading multiclass TabPFN model...")
        with open(MULTICLASS_MODEL_PATH, "rb") as f:
            multiclass_model = pickle.load(f)

        if MULTICLASS_SCALER_PATH.exists():
            logger.info("Loading multiclass scaler...")
            with open(MULTICLASS_SCALER_PATH, "rb") as f:
                multiclass_scaler = pickle.load(f)
        else:
            multiclass_scaler = binary_scaler

        if MULTICLASS_LABEL_ENCODER_PATH.exists():
            logger.info("Loading multiclass label encoder...")
            with open(
                MULTICLASS_LABEL_ENCODER_PATH,
                "r",
                encoding="utf-8",
            ) as f:
                multiclass_label_encoder = json.load(f)

    logger.info(
        "Binary model ready. Multiclass model: %s",
        "Loaded (%d classes)" % len(multiclass_label_encoder["classes"])
        if multiclass_label_encoder is not None
        else "Not available",
    )

    return (
        binary_model,
        binary_scaler,
        feature_names,
        multiclass_model,
        multiclass_scaler,
        multiclass_label_encoder,
    )


# =========================================================
# PCAP → Flows → Features
# =========================================================

def extract_and_build_features(
    pcap_path: Path,
    work_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    """Convert PCAP into the feature representation used by the model."""

    flows_path = work_dir / "flows.parquet"

    logger.info(
        "Extracting flows from %s",
        pcap_path,
    )

    summary = extract_flows_from_pcap(
        pcap_path=pcap_path,
        output_path=flows_path,
        backend="scapy",
        flow_timeout=120.0,
        idle_timeout=60.0,
        batch_size=50_000,
    )

    logger.info(
        "Extracted %d flows from %d packets",
        summary["total_flows"],
        summary["total_packets"],
    )

    flows = pd.read_parquet(
        flows_path
    )

    logger.info(
        "Building ML features..."
    )

    features = compute_all_features(
        flows
    )

    logger.info(
        "Generated %d rows × %d columns",
        len(features),
        len(features.columns),
    )

    return features, summary


# =========================================================
# Prepare model input
# =========================================================

def prepare_model_input(
    features: pd.DataFrame,
    feature_names: list,
    binary_scaler,
    multiclass_scaler=None,
):
    """Prepare features for binary and multiclass models."""

    missing_features = [
        feature
        for feature in feature_names
        if feature not in features.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features: "
            + ", ".join(missing_features)
        )

    metadata_columns = [
        column
        for column in features.columns
        if column not in feature_names
    ]

    metadata = features[
        metadata_columns
    ].copy()

    X = features[
        feature_names
    ].copy()

    # Convert all model columns to numeric.
    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Replace infinite values.
    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Missing/invalid model inputs are replaced by 0.
    X = X.fillna(0)

    logger.info(
        "Applying fitted scaler(s)..."
    )

    X_binary_scaled = binary_scaler.transform(X)

    if multiclass_scaler is not None:
        X_multi_scaled = multiclass_scaler.transform(X)
        return metadata, X_binary_scaled, X_multi_scaled

    return metadata, X_binary_scaled


# =========================================================
# Prediction
# =========================================================

def predict(
    model,
    X,
):
    """Run prediction using the existing trained TabPFN model."""

    logger.info(
        "Running TabPFN inference..."
    )

    logger.info(
        "Predicting probabilities on %d rows...",
        len(X),
    )

    prediction_start = time.perf_counter()

    if hasattr(
        model,
        "predict_proba",
    ):
        probabilities = model.predict_proba(
            X
        )

        predictions = np.argmax(
            probabilities,
            axis=1,
        )

    else:
        predictions = model.predict(
            X
        )

        predictions = np.asarray(
            predictions
        )

        probabilities = None

    prediction_seconds = (
        time.perf_counter()
        - prediction_start
    )

    logger.info(
        "predict_proba completed in %.2fs",
        prediction_seconds,
    )

    logger.info(
        "Predicted %d samples in %.1fs",
        len(predictions),
        prediction_seconds,
    )

    return (
        predictions,
        probabilities,
    )


def predict_multiclass(
    model,
    X,
    batch_size: int = 200,
):
    """Run multiclass prediction in batches to manage CPU/memory."""

    logger.info(
        "Running TabPFN multiclass inference on %d rows...",
        len(X),
    )

    if len(X) == 0:
        return np.empty(0, dtype=int), np.empty((0, 9))

    prediction_start = time.perf_counter()

    probs_list = []
    for i in range(0, len(X), batch_size):
        batch = X[i : i + batch_size]
        batch_probs = model.predict_proba(batch)
        probs_list.append(batch_probs)

    probabilities = np.vstack(probs_list)
    predictions = np.argmax(probabilities, axis=1)

    prediction_seconds = time.perf_counter() - prediction_start
    logger.info(
        "Multiclass inference on %d samples completed in %.2fs",
        len(predictions),
        prediction_seconds,
    )

    return (
        predictions,
        probabilities,
    )


# =========================================================
# Convert prediction labels
# =========================================================

def convert_prediction_label(
    value,
):
    """Convert numeric prediction to human-readable label."""

    if int(value) == 0:
        return "Normal"

    if int(value) == 1:
        return "Attack"

    return str(value)


# =========================================================
# Build prediction output
# =========================================================

def build_prediction_output(
    metadata: pd.DataFrame,
    original_features: pd.DataFrame,
    predictions,
    probabilities,
    inference_meta: dict | None = None,
    per_model_probabilities: dict | None = None,
    predicted_categories=None,
    category_confidences=None,
    prediction_statuses=None,
    multiclass_probabilities=None,
    class_names=None,
    multiclass_predictions=None,
    is_conflicts=None,
    conflict_reasons=None,
):
    """Build the flow-level prediction dataframe with binary and multiclass fields."""

    output = metadata.copy()

    for column in original_features.columns:
        output[column] = (
            original_features[column]
            .values
        )

    output["prediction"] = predictions

    output["prediction_label"] = [
        convert_prediction_label(
            value
        )
        for value in predictions
    ]
    output["binary_prediction"] = output["prediction_label"]

    if probabilities is not None:
        if probabilities.shape[1] >= 2:
            output["normal_probability"] = (
                probabilities[:, 0]
            )
            output["attack_probability"] = (
                probabilities[:, 1]
            )
            output["binary_attack_probability"] = (
                probabilities[:, 1]
            )

    # Multiclass predictions & classification status
    if multiclass_predictions is not None:
        output["predicted_attack_category_id"] = [int(p) for p in multiclass_predictions]

    if predicted_categories is not None:
        output["predicted_attack_category"] = predicted_categories
        output["attack_category_confidence"] = category_confidences
        output["category_confidence"] = category_confidences
        output["prediction_status"] = prediction_statuses
        output["classification_status"] = prediction_statuses
        if is_conflicts is not None:
            output["is_model_conflict"] = is_conflicts
            output["conflict_reason"] = conflict_reasons

    if multiclass_probabilities is not None and class_names is not None:
        for idx, cls_name in enumerate(class_names):
            clean_cls = f"prob_{cls_name.lower().replace(' ', '_')}"
            output[clean_cls] = multiclass_probabilities[:, idx]

    # Optional ensemble provenance columns
    if inference_meta is not None and inference_meta.get("num_models", 1) > 1:
        output["models_used"] = ", ".join(inference_meta.get("models_used", []))
        output["ensemble_size"] = inference_meta.get("num_models", 1)
        output["ensemble_method"] = inference_meta.get("ensemble_method", "Mean probability")
        output["prediction_threshold"] = inference_meta.get("prediction_threshold", 0.5)

    # Optional per-model probabilities
    if per_model_probabilities:
        for m_id, m_probs in per_model_probabilities.items():
            if hasattr(m_probs, "ndim") and m_probs.ndim == 2 and m_probs.shape[1] >= 2:
                output[f"{m_id}_attack_probability"] = m_probs[:, 1]
            elif hasattr(m_probs, "ndim") and m_probs.ndim == 1:
                output[f"{m_id}_attack_probability"] = m_probs

    return output


# =========================================================
# Ground-truth evaluation
# =========================================================

def evaluate_with_ground_truth(
    features: pd.DataFrame,
    predictions,
    probabilities,
    ground_truth_path: Path,
    multiclass_predictions=None,
    multiclass_probabilities=None,
    multiclass_label_encoder=None,
):
    """Match flows against ground truth and calculate binary & multiclass metrics."""

    logger.info(
        "Loading ground truth..."
    )

    ground_truth = load_ground_truth_csv(
        ground_truth_path
    )

    logger.info(
        "Matching flows against ground truth..."
    )

    labeled_flows = match_flows_to_gt_csv(
        features,
        ground_truth,
        time_tolerance=5.0,
        use_ip_port=True,
    )

    matched_mask = (
        labeled_flows["label"] != -1
    )

    matched_count = int(
        matched_mask.sum()
    )

    logger.info(
        "Matched %d/%d flows",
        matched_count,
        len(labeled_flows),
    )

    if matched_count == 0:
        raise ValueError(
            "No flows matched the supplied ground truth."
        )

    y_true = (
        labeled_flows.loc[
            matched_mask,
            "label",
        ]
        .astype(int)
        .to_numpy()
    )

    y_pred = np.asarray(
        predictions
    )[matched_mask.to_numpy()]

    matched_probabilities = None

    if probabilities is not None:
        matched_probabilities = (
            np.asarray(probabilities)[
                matched_mask.to_numpy()
            ]
        )

    metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=matched_probabilities,
    )

    multiclass_metrics = None
    if (
        multiclass_predictions is not None
        and multiclass_label_encoder is not None
        and "attack_cat" in labeled_flows.columns
    ):
        class_to_idx = multiclass_label_encoder["class_to_idx"]
        class_names = multiclass_label_encoder["classes"]

        matched_gt_cats = (
            labeled_flows.loc[matched_mask, "attack_cat"]
            .astype(str)
            .str.strip()
        )

        y_true_multi = []
        valid_indices = []

        for idx, cat_name in enumerate(matched_gt_cats):
            if cat_name in class_to_idx:
                y_true_multi.append(class_to_idx[cat_name])
                valid_indices.append(idx)
            elif cat_name.lower() == "normal" and "Normal" in class_to_idx:
                y_true_multi.append(class_to_idx["Normal"])
                valid_indices.append(idx)

        if len(valid_indices) > 0:
            y_true_multi = np.array(y_true_multi)
            matched_mc_preds = np.asarray(multiclass_predictions)[matched_mask.to_numpy()][valid_indices]
            matched_mc_probs = None
            if multiclass_probabilities is not None:
                matched_mc_probs = np.asarray(multiclass_probabilities)[matched_mask.to_numpy()][valid_indices]

            multiclass_metrics = compute_multiclass_metrics(
                y_true=y_true_multi,
                y_pred=matched_mc_preds,
                y_proba=matched_mc_probs,
                class_names=class_names,
            )

    return (
        labeled_flows,
        metrics,
        multiclass_metrics,
    )


# =========================================================
# Save prediction CSV
# =========================================================

def save_results(
    output: pd.DataFrame,
    pcap_path: Path,
    work_dir: Path,
):
    """Save flow-level prediction CSV."""

    csv_path = (
        work_dir
        / f"{pcap_path.stem}_predictions.csv"
    )

    output.to_csv(
        csv_path,
        index=False,
    )

    logger.info(
        "Prediction CSV saved to:"
    )

    logger.info(
        "%s",
        csv_path,
    )

    return csv_path


# =========================================================
# Print prediction summary
# =========================================================

def print_summary(
    output: pd.DataFrame,
):
    """Print prediction counts, attack categories, and classification status."""

    total = len(output)

    normal_count = int(
        (
            output["prediction"]
            == 0
        ).sum()
    )

    attack_count = int(
        (
            output["prediction"]
            == 1
        ).sum()
    )

    print()
    print("=" * 60)
    print("PCAP ANALYSIS COMPLETE")
    print("=" * 60)

    print(
        f"Total flows : {total}"
    )

    print(
        f"Normal      : {normal_count} ({(normal_count / total * 100):.1f}%)" if total else f"Normal      : {normal_count}"
    )

    print(
        f"Attack      : {attack_count} ({(attack_count / total * 100):.1f}%)" if total else f"Attack      : {attack_count}"
    )

    if "predicted_attack_category" in output.columns:
        print("-" * 60)
        print("ATTACK CATEGORY BREAKDOWN")
        print("-" * 60)
        cat_counts = output["predicted_attack_category"].value_counts()
        for cat, cnt in cat_counts.items():
            pct = (cnt / total * 100) if total else 0.0
            print(f"  {cat:<18}: {cnt:>6} ({pct:>5.1f}%)")

    if "prediction_status" in output.columns:
        print("-" * 60)
        print("CLASSIFICATION RELIABILITY & STATUS")
        print("-" * 60)
        status_counts = output["prediction_status"].value_counts()
        for st, cnt in status_counts.items():
            pct = (cnt / total * 100) if total else 0.0
            print(f"  {st:<18}: {cnt:>6} ({pct:>5.1f}%)")
        if "attack_category_confidence" in output.columns:
            avg_conf = output["attack_category_confidence"].mean()
            print(f"  Average Confidence: {avg_conf * 100:.2f}%")

    print("=" * 60)


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze a new PCAP using "
            "the existing trained TabPFN model."
        )
    )

    parser.add_argument(
        "--pcap",
        required=True,
        help=(
            "Path to the PCAP file "
            "to analyze."
        ),
    )

    parser.add_argument(
        "--ground-truth",
        default=None,
        help=(
            "Optional ground-truth CSV "
            "for evaluation."
        ),
    )

    parser.add_argument(
        "--max-rows-per-worker",
        type=int,
        default=None,
        help="Max rows per worker chunk (default: 10000).",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum parallel inference workers (default: auto).",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification decision threshold for Attack (default: 0.5).",
    )

    parser.add_argument(
        "--enable-ensemble",
        action="store_true",
        default=None,
        help="Enable multi-model ensemble if multiple models are available.",
    )

    parser.add_argument(
        "--include-per-model",
        action="store_true",
        default=False,
        help="Include per-model probability columns in output CSV.",
    )

    parser.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold below which predictions are marked Uncertain (default: 0.5).",
    )

    args = parser.parse_args()

    analysis_start = (
        time.perf_counter()
    )

    pcap_path = (
        Path(args.pcap).resolve()
    )

    if not pcap_path.exists():
        raise FileNotFoundError(
            f"PCAP file not found: "
            f"{pcap_path}"
        )

    work_dir = (
        OUTPUT_DIR
        / pcap_path.stem
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "PCAP: %s",
        pcap_path,
    )

    # -----------------------------------------------------
    # 1. Load already-trained artifacts & configure inference
    # -----------------------------------------------------

    (
        model,
        scaler,
        feature_names,
        multiclass_model,
        multiclass_scaler,
        multiclass_label_encoder,
    ) = load_artifacts(include_multiclass=True)

    try:
        cfg = load_config()
        inf_cfg = cfg.inference
    except Exception as exc:
        logger.warning("Could not load pipeline config (%s), using defaults.", exc)
        from tabpfn_nids.pipeline_config import InferenceConfig
        inf_cfg = InferenceConfig()

    max_rows = args.max_rows_per_worker or inf_cfg.max_rows_per_worker
    max_workers = args.max_workers if args.max_workers is not None else inf_cfg.max_workers
    threshold = args.threshold if args.threshold is not None else inf_cfg.prediction_threshold
    enable_ensemble = args.enable_ensemble if args.enable_ensemble is not None else inf_cfg.enable_ensemble
    include_per_model = args.include_per_model or inf_cfg.include_per_model_probabilities
    executor_type = inf_cfg.executor_type
    aggregation = inf_cfg.probability_aggregation
    weights = inf_cfg.weights

    registry = ModelRegistry(
        models_dir=ARTIFACTS_DIR / "models",
        default_schema_path=FEATURE_SCHEMA_PATH,
    )
    registry.register_model(
        ModelInfo(
            model_id="tabpfn_binary_model",
            model_path=MODEL_PATH,
            schema_path=FEATURE_SCHEMA_PATH,
            feature_count=len(feature_names),
            feature_names=feature_names,
        ),
        model_instance=model,
    )

    inference_manager = DynamicInferenceManager(
        max_rows_per_worker=max_rows,
        max_workers=max_workers,
        executor_type=executor_type,
        probability_aggregation=aggregation,
        prediction_threshold=threshold,
        enable_ensemble=enable_ensemble,
        include_per_model_probabilities=include_per_model,
        weights=weights,
        model_registry=registry,
    )

    # -----------------------------------------------------
    # 2. PCAP → flows → features
    # -----------------------------------------------------

    features, extraction_summary = (
        extract_and_build_features(
            pcap_path,
            work_dir,
        )
    )

    # -----------------------------------------------------
    # 3. Prepare model input
    # -----------------------------------------------------

    metadata, X_binary, X_multi = (
        prepare_model_input(
            features,
            feature_names,
            scaler,
            multiclass_scaler,
        )
    )

    original_features = (
        features[
            feature_names
        ].copy()
    )

    # -----------------------------------------------------
    # 4. Predict
    # -----------------------------------------------------

    inf_result = inference_manager.predict(
        X_binary,
        model_ids=["tabpfn_binary_model"],
    )

    predictions = inf_result.predictions
    probabilities = inf_result.probabilities
    inference_meta = inf_result.metadata
    per_model_probabilities = inf_result.per_model_probabilities

    multi_preds = None
    multi_probs = None
    predicted_categories = None
    category_confidences = None
    prediction_statuses = None
    classes = None

    if multiclass_model is not None and multiclass_label_encoder is not None:
        classes = multiclass_label_encoder.get("classes", [])
        mapping = get_attack_category_mapping(multiclass_label_encoder)

        multi_preds, multi_probs = predict_multiclass(
            multiclass_model,
            X_multi,
            batch_size=200,
        )

        predicted_categories = [
            mapping.get(int(idx), decode_attack_category(idx, multiclass_label_encoder))
            for idx in multi_preds
        ]
        category_confidences = np.max(multi_probs, axis=1)

        uncertainty_thresh = args.uncertainty_threshold
        prediction_statuses = []
        is_conflicts = []
        conflict_reasons = []
        for i, p_cat in enumerate(predicted_categories):
            bin_lbl = "Normal" if int(predictions[i]) == 0 else "Attack"
            conf = float(category_confidences[i])
            has_conflict = False
            reason = None
            if bin_lbl == "Attack" and str(p_cat).lower() == "normal":
                has_conflict = True
                reason = "Binary model predicted Attack, but multiclass model classified flow as Normal."
            elif bin_lbl == "Normal" and str(p_cat).lower() != "normal":
                has_conflict = True
                reason = f"Binary model predicted Normal, but multiclass model detected {p_cat}."

            is_conflicts.append(has_conflict)
            conflict_reasons.append(reason)

            if has_conflict:
                status = "Uncertain"
            elif conf >= uncertainty_thresh:
                status = "Classified"
            else:
                status = "Uncertain"
            prediction_statuses.append(status)
    else:
        is_conflicts = None
        conflict_reasons = None

    output = (
        build_prediction_output(
            metadata,
            original_features,
            predictions,
            probabilities,
            inference_meta=inference_meta,
            per_model_probabilities=per_model_probabilities,
            predicted_categories=predicted_categories,
            category_confidences=category_confidences,
            prediction_statuses=prediction_statuses,
            multiclass_probabilities=multi_probs,
            class_names=classes,
            multiclass_predictions=multi_preds,
            is_conflicts=is_conflicts,
            conflict_reasons=conflict_reasons,
        )
    )

    # -----------------------------------------------------
    # 5. Optional ground-truth evaluation
    # -----------------------------------------------------

    metrics = None
    multiclass_metrics = None
    labeled_flows = None
    ground_truth_path = None

    if args.ground_truth:

        ground_truth_path = (
            Path(
                args.ground_truth
            ).resolve()
        )

        if not ground_truth_path.exists():
            raise FileNotFoundError(
                "Ground-truth file not found: "
                f"{ground_truth_path}"
            )

        (
            labeled_flows,
            metrics,
            multiclass_metrics,
        ) = evaluate_with_ground_truth(
            features=features,
            predictions=predictions,
            probabilities=probabilities,
            ground_truth_path=ground_truth_path,
            multiclass_predictions=multi_preds,
            multiclass_probabilities=multi_probs,
            multiclass_label_encoder=multiclass_label_encoder,
        )

        # Add ground-truth information
        # to the prediction CSV.

        output["actual_label"] = (
            labeled_flows[
                "label"
            ].values
        )

        output[
            "actual_attack_category"
        ] = (
            labeled_flows[
                "attack_cat"
            ].values
        )

        output["match_type"] = (
            labeled_flows[
                "match_type"
            ].values
        )

    # -----------------------------------------------------
    # 6. Save prediction CSV
    # -----------------------------------------------------

    csv_path = save_results(
        output,
        pcap_path,
        work_dir,
    )

    # -----------------------------------------------------
    # 7. Print prediction summary
    # -----------------------------------------------------

    print_summary(
        output
    )

    # -----------------------------------------------------
    # 8. Print evaluation metrics
    # -----------------------------------------------------

    if metrics is not None:

        print()
        print("=" * 60)
        print("BINARY GROUND TRUTH EVALUATION")
        print("=" * 60)

        print(
            f"Accuracy    : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision   : "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall      : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"F1 Score    : "
            f"{metrics['f1_score']:.4f}"
        )

        if (
            metrics["roc_auc"]
            is not None
        ):

            print(
                f"ROC-AUC     : "
                f"{metrics['roc_auc']:.4f}"
            )

        else:

            print(
                "ROC-AUC     : N/A"
            )

        print()
        print(
            "Confusion Matrix"
        )

        print(
            "                 Predicted"
        )

        print(
            "              Normal  Attack"
        )

        print(
            f"Actual Normal  "
            f"{metrics['true_negatives']:>6}  "
            f"{metrics['false_positives']:>6}"
        )

        print(
            f"Actual Attack  "
            f"{metrics['false_negatives']:>6}  "
            f"{metrics['true_positives']:>6}"
        )

        print(
            "=" * 60
        )

    if multiclass_metrics is not None:
        print()
        print("=" * 60)
        print("MULTICLASS ATTACK IDENTIFICATION EVALUATION")
        print("=" * 60)
        print(format_multiclass_metrics(multiclass_metrics))
        print("=" * 60)

    # -----------------------------------------------------
    # 9. Total analysis time
    # -----------------------------------------------------

    total_analysis_seconds = (
        time.perf_counter()
        - analysis_start
    )

    # -----------------------------------------------------
    # 10. Generate detailed reports
    # -----------------------------------------------------

    report_data = build_report_data(
        pcap_path=pcap_path,
        extraction_summary=(
            extraction_summary
        ),
        features=features,
        feature_names=feature_names,
        predictions=predictions,
        probabilities=probabilities,
        total_analysis_seconds=(
            total_analysis_seconds
        ),
        ground_truth_path=(
            ground_truth_path
        ),
        metrics=metrics,
        labeled_flows=labeled_flows,
        inference_meta=inference_meta,
        multiclass_predictions=multi_preds,
        multiclass_probabilities=multi_probs,
        predicted_categories=predicted_categories,
        category_confidences=category_confidences,
        prediction_statuses=prediction_statuses,
        multiclass_metrics=multiclass_metrics,
        class_names=classes,
        uncertainty_threshold=args.uncertainty_threshold,
    )

    json_report_path = (
        work_dir
        / "analysis_report.json"
    )

    html_report_path = (
        work_dir
        / "analysis_report.html"
    )
    
    docx_report_path = (
        work_dir
        / f"{pcap_path.stem}_analysis_report.docx"
    )

    save_json_report(
        report_data,
        json_report_path,
    )

    save_html_report(
        report_data,
        html_report_path,
    )
    
    save_docx_report(
        report_data,
        docx_report_path,
    )

    # -----------------------------------------------------
    # 11. Final output paths
    # -----------------------------------------------------

    print(
        f"CSV file: {csv_path}"
    )

    print(
        f"Total analysis time: "
        f"{total_analysis_seconds:.2f}s"
    )

    print(
        f"JSON report: "
        f"{json_report_path}"
    )

    print(
        f"HTML report: "
        f"{html_report_path}"
    )
    
    print(
        f"Word report: "
        f"{docx_report_path}"
    )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()