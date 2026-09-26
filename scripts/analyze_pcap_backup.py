from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from tabpfn_nids.flows.flow_builder import extract_flows_from_pcap
from tabpfn_nids.features.feature_pipeline import compute_all_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "data" / "artifacts" / "models" / "tabpfn_binary_model.pkl"
SCHEMA_PATH = PROJECT_ROOT / "data" / "artifacts" / "models" / "model_feature_schema.json"
SCALER_PATH = PROJECT_ROOT / "data" / "artifacts" / "preprocessing" / "fitted_scaler.pkl"

OUTPUT_DIR = PROJECT_ROOT / "results" / "pcap_analysis"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_artifacts():
    """Load the trained model, feature schema, and fitted scaler."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Feature schema not found: {SCHEMA_PATH}"
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Fitted scaler not found: {SCALER_PATH}"
        )

    logger.info("Loading trained model...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    logger.info("Loading feature schema...")
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    logger.info("Loading fitted scaler...")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    feature_names = schema["feature_names"]

    logger.info("Model expects %d features", len(feature_names))

    return model, scaler, feature_names


def extract_and_build_features(pcap_path: Path, work_dir: Path) -> pd.DataFrame:
    """Convert PCAP into the same feature representation used by the model."""

    flows_path = work_dir / "flows.parquet"

    logger.info("Extracting flows from %s", pcap_path)

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

    flows = pd.read_parquet(flows_path)

    logger.info("Building ML features...")

    features = compute_all_features(flows)

    logger.info(
        "Generated %d rows × %d columns",
        len(features),
        len(features.columns),
    )

    return features


def prepare_model_input(
    features: pd.DataFrame,
    feature_names: list[str],
    scaler,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the trained feature schema and apply the saved scaler."""

    missing = [
        feature
        for feature in feature_names
        if feature not in features.columns
    ]

    if missing:
        raise ValueError(
            "The new PCAP is missing required model features:\n"
            + "\n".join(f"  - {name}" for name in missing)
        )

    # Preserve metadata so it can be included in the output CSV.
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

    metadata = features[
        [c for c in metadata_columns if c in features.columns]
    ].copy()

    # IMPORTANT:
    # Use exactly the feature order stored in model_feature_schema.json.
    X = features[feature_names].copy()

    # Convert everything to numeric.
    X = X.apply(pd.to_numeric, errors="coerce")

    # Replace invalid values.
    X = X.replace([np.inf, -np.inf], np.nan)

    # For inference, use zero for any remaining invalid values.
    X = X.fillna(0.0)

    logger.info("Applying fitted scaler...")

    X_scaled = scaler.transform(X)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=feature_names,
        index=X.index,
    )

    return metadata, X_scaled


def predict(
    model,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray | None]:

    logger.info("Running TabPFN inference...")

    probabilities = None

    if hasattr(model, "predict_proba"):
        try:
            probabilities = np.asarray(
                model.predict_proba(X)
            )

            if probabilities.ndim == 2:
                predictions = np.argmax(
                    probabilities,
                    axis=1,
                )
            else:
                predictions = np.asarray(
                    model.predict(X)
                )

        except Exception as exc:
            logger.warning(
                "predict_proba failed: %s. Falling back to predict().",
                exc,
            )

            predictions = np.asarray(
                model.predict(X)
            )

    else:
        predictions = np.asarray(
            model.predict(X)
        )

    logger.info(
        "Predicted %d samples",
        len(predictions),
    )

    return predictions, probabilities


def convert_prediction_label(prediction) -> str:
    """Convert model prediction to Normal / Attack."""

    if isinstance(prediction, str):
        value = prediction.lower()

        if value == "normal":
            return "Normal"

        if value == "attack":
            return "Attack"

    try:
        return "Attack" if int(prediction) == 1 else "Normal"
    except (TypeError, ValueError):
        return str(prediction)


def build_prediction_output(
    metadata: pd.DataFrame,
    original_features: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
) -> pd.DataFrame:
    """Build the downloadable CSV using original, unscaled features."""

    output = metadata.copy()

    # Keep ORIGINAL feature values for the downloadable CSV.
    for column in original_features.columns:
        output[column] = original_features[column].values

    output["prediction"] = predictions

    output["prediction_label"] = [
        convert_prediction_label(value)
        for value in predictions
    ]

    if probabilities is not None and probabilities.ndim == 2:
        if probabilities.shape[1] >= 2:
            output["normal_probability"] = probabilities[:, 0]
            output["attack_probability"] = probabilities[:, 1]

    return output


def save_results(
    output: pd.DataFrame,
    pcap_path: Path,
    output_dir: Path,
) -> Path:

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{pcap_path.stem}_predictions.csv"

    output.to_csv(csv_path, index=False)

    logger.info("Prediction CSV saved to:")
    logger.info("%s", csv_path)

    return csv_path


def print_summary(output: pd.DataFrame) -> None:

    counts = output["prediction_label"].value_counts()

    normal_count = int(counts.get("Normal", 0))
    attack_count = int(counts.get("Attack", 0))

    print()
    print("=" * 60)
    print("PCAP ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Total flows : {len(output):,}")
    print(f"Normal      : {normal_count:,}")
    print(f"Attack      : {attack_count:,}")
    print("=" * 60)
    print()


def main():

    parser = argparse.ArgumentParser(
        description="Analyze a new PCAP using the existing trained TabPFN model."
    )

    parser.add_argument(
        "--pcap",
        required=True,
        help="Path to the PCAP file to analyze.",
    )

    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Optional ground-truth CSV for evaluation.",
    )

    args = parser.parse_args()

    pcap_path = Path(args.pcap).resolve()

    if not pcap_path.exists():
        raise FileNotFoundError(
            f"PCAP file not found: {pcap_path}"
        )

    work_dir = OUTPUT_DIR / pcap_path.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info("PCAP: %s", pcap_path)

    # ---------------------------------------------------------
    # 1. Load already-trained artifacts
    # ---------------------------------------------------------
    model, scaler, feature_names = load_artifacts()

    # ---------------------------------------------------------
    # 2. PCAP → flows → features
    # ---------------------------------------------------------
    features = extract_and_build_features(
        pcap_path,
        work_dir,
    )

    # ---------------------------------------------------------
    # 3. Prepare exact model input
    # ---------------------------------------------------------
    metadata, X = prepare_model_input(
        features,
        feature_names,
        scaler,
    )
    original_features = features[feature_names].copy()

    # ---------------------------------------------------------
    # 4. Predict
    # ---------------------------------------------------------
    predictions, probabilities = predict(
        model,
        X,
    )

    # ---------------------------------------------------------
    # 5. Build downloadable CSV
    # ---------------------------------------------------------
    output = build_prediction_output(
        metadata,
        original_features,
        predictions,
        probabilities,
    )

    csv_path = save_results(
        output,
        pcap_path,
        work_dir,
    )

    print_summary(output)

    print(f"CSV file: {csv_path}")


if __name__ == "__main__":
    main()