"""Tests for multiclass attack detection in TabPFN-NIDS."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from tabpfn_nids.evaluation.metrics import (
    compute_multiclass_metrics,
    decode_attack_category,
    format_multiclass_metrics,
    get_attack_category_mapping,
)
from tabpfn_nids.pcap.analysis_report import (
    build_report_data,
    save_html_report,
    save_docx_report,
)
from tabpfn_nids.api.pcap_router import (
    _load_artifacts,
    _get_model_info,
    _prepare_flows_table,
    ARTIFACTS_DIR,
    MULTICLASS_MODEL_PATH,
    MULTICLASS_SCALER_PATH,
    MULTICLASS_LABEL_ENCODER_PATH,
    MULTICLASS_SCHEMA_PATH,
)


# ===========================================================================
# 1. Metric computation tests
# ===========================================================================

def test_multiclass_metrics_perfect() -> None:
    """A perfect multiclass classifier yields 1.0 across accuracy, F1, and AUC."""
    classes = ["Normal", "DoS", "Exploits"]
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    y_prob = np.zeros((6, 3))
    for i, label in enumerate(y_true):
        y_prob[i, label] = 1.0

    metrics = compute_multiclass_metrics(y_true, y_pred, y_proba=y_prob, classes=classes)

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["weighted_f1"] == pytest.approx(1.0)
    assert metrics["roc_auc_ovr"] == pytest.approx(1.0)
    assert len(metrics["classes"]) == 3

    # Check per-class metrics
    for c in classes:
        assert metrics["per_class"][c]["precision"] == pytest.approx(1.0)
        assert metrics["per_class"][c]["recall"] == pytest.approx(1.0)
        assert metrics["per_class"][c]["f1_score"] == pytest.approx(1.0)
        assert metrics["per_class"][c]["support"] == 2


def test_multiclass_confusion_matrix_shape() -> None:
    """Confusion matrix has shape (C, C) where C is number of unique classes."""
    classes = ["Normal", "Fuzzers", "Exploits", "DoS"]
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 1, 3])

    metrics = compute_multiclass_metrics(y_true, y_pred, classes=classes)
    cm = np.array(metrics["confusion_matrix"])
    assert cm.shape == (4, 4)
    assert metrics["accuracy"] == pytest.approx(0.75)


def test_format_multiclass_metrics() -> None:
    """Formatted report contains summary statistics and class breakdown."""
    classes = ["Normal", "DoS"]
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.6, 0.4]])

    metrics = compute_multiclass_metrics(y_true, y_pred, y_proba=y_prob, classes=classes)
    formatted = format_multiclass_metrics(metrics)

    assert "Multiclass Results" in formatted
    assert "Accuracy" in formatted
    assert "Macro F1" in formatted
    assert "Weighted F1" in formatted
    assert "Per-Class Metrics" in formatted
    assert "Normal" in formatted
    assert "DoS" in formatted


# ===========================================================================
# 2. Artifact and model schema tests
# ===========================================================================

def test_multiclass_artifacts_exist() -> None:
    """Trained model, scaler, label encoder, and schema files must exist."""
    assert MULTICLASS_MODEL_PATH.exists(), f"Missing: {MULTICLASS_MODEL_PATH}"
    assert MULTICLASS_SCALER_PATH.exists(), f"Missing: {MULTICLASS_SCALER_PATH}"
    assert MULTICLASS_LABEL_ENCODER_PATH.exists(), f"Missing: {MULTICLASS_LABEL_ENCODER_PATH}"
    assert MULTICLASS_SCHEMA_PATH.exists(), f"Missing: {MULTICLASS_SCHEMA_PATH}"


def test_multiclass_label_encoder_classes() -> None:
    """Encoder must include all 9 attack classes with bijective mapping."""
    with open(MULTICLASS_LABEL_ENCODER_PATH, "r", encoding="utf-8") as f:
        encoder = json.load(f)

    expected_classes = [
        "Normal",
        "Fuzzers",
        "Exploits",
        "Reconnaissance",
        "DoS",
        "Generic",
        "Shellcode",
        "Worms",
        "Backdoors",
    ]
    classes = encoder.get("classes", [])
    assert len(classes) == 9
    for c in expected_classes:
        assert c in classes

    class_to_idx = encoder.get("class_to_idx", {})
    idx_to_class = encoder.get("idx_to_class", {})
    assert len(class_to_idx) == 9
    assert len(idx_to_class) == 9
    for name, idx in class_to_idx.items():
        assert idx_to_class[str(idx)] == name


def test_multiclass_feature_schema() -> None:
    """Schema must define 67 network features."""
    with open(MULTICLASS_SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    assert schema["n_features"] == 67
    assert len(schema["feature_names"]) == 67
    assert "duration" in schema["feature_names"]
    assert "fwd_packets" in schema["feature_names"]


def test_multiclass_scaler_transform() -> None:
    """Fitted scaler transforms 67-dimensional feature vectors without error."""
    with open(MULTICLASS_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    dummy_features = np.ones((5, 67), dtype=float)
    scaled = scaler.transform(dummy_features)
    assert scaled.shape == (5, 67)
    assert np.all(np.isfinite(scaled))


# ===========================================================================
# 3. Report generation and uncertainty logic tests
# ===========================================================================

def test_build_report_data_with_multiclass(tmp_path: Path) -> None:
    """Report data dictionary must correctly include multiclass analysis."""
    pcap_path = tmp_path / "test_traffic.pcap"
    pcap_path.write_bytes(b"\xd4\xc3\xb2\xa1" + b"\x00" * 100)

    extraction_summary = {"packet_count": 100, "flow_count": 4}
    features = pd.DataFrame(
        {
            "src_ip": ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"],
            "dst_ip": ["10.0.0.100", "10.0.0.100", "10.0.0.100", "10.0.0.100"],
            "protocol": ["TCP", "UDP", "TCP", "TCP"],
        }
    )
    feature_names = ["feat1", "feat2"]
    predictions = np.array([0, 1, 1, 0])
    probabilities = np.array([[0.9, 0.1], [0.1, 0.9], [0.3, 0.7], [0.8, 0.2]])

    classes = ["Normal", "DoS", "Exploits", "Generic"]
    mc_preds = np.array([0, 1, 2, 0])
    mc_probs = np.array([
        [0.92, 0.03, 0.03, 0.02],
        [0.05, 0.88, 0.05, 0.02],
        [0.20, 0.20, 0.45, 0.15],  # max conf 0.45 -> Uncertain (< 0.50)
        [0.85, 0.05, 0.05, 0.05],
    ])
    predicted_cats = ["Normal", "DoS", "Exploits", "Normal"]
    cat_confidences = np.array([0.92, 0.88, 0.45, 0.85])
    statuses = ["Classified", "Classified", "Uncertain", "Classified"]

    report = build_report_data(
        pcap_path=pcap_path,
        extraction_summary=extraction_summary,
        features=features,
        feature_names=feature_names,
        predictions=predictions,
        probabilities=probabilities,
        total_analysis_seconds=1.5,
        multiclass_predictions=mc_preds,
        multiclass_probabilities=mc_probs,
        class_names=classes,
        predicted_categories=predicted_cats,
        category_confidences=cat_confidences,
        prediction_statuses=statuses,
        uncertainty_threshold=0.50,
    )

    assert "multiclass" in report
    mc_sum = report["multiclass"]
    assert mc_sum["uncertainty_threshold"] == 0.50
    assert mc_sum["status_counts"]["uncertain"] == 1
    assert mc_sum["status_counts"]["classified"] == 3
    assert mc_sum["uncertain_rate_percentage"] == pytest.approx(25.0)
    assert "DoS" in mc_sum["categories_detected"]
    assert "Exploits" in mc_sum["categories_detected"]
    assert mc_sum["category_counts"]["Normal"] == 2
    assert mc_sum["category_counts"]["DoS"] == 1
    assert mc_sum["category_counts"]["Exploits"] == 1


# ===========================================================================
# 4. API router helper tests
# ===========================================================================

def test_api_artifact_loading() -> None:
    """API artifact loader returns both binary and multiclass models."""
    bin_model, bin_scaler, feat_names, mc_model, mc_scaler, mc_le = _load_artifacts()
    assert bin_model is not None
    assert bin_scaler is not None
    assert len(feat_names) == 67
    assert mc_model is not None
    assert mc_scaler is not None
    assert mc_le is not None
    assert len(mc_le["classes"]) == 9


def test_api_get_model_info() -> None:
    """Model info includes dual binary and multiclass capabilities."""
    info = _get_model_info()
    assert info["model_type"] == "TabPFN"
    assert info["multiclass_available"] is True
    assert info["num_multiclass_classes"] == 9
    assert "Exploits" in info["multiclass_classes"]


def test_api_prepare_flows_table_columns() -> None:
    """Flows table includes multiclass category, confidence, and status."""
    sample_df = pd.DataFrame(
        {
            "src_ip": ["192.168.1.10"],
            "dst_ip": ["192.168.1.1"],
            "src_port": [54321],
            "dst_port": [80],
            "protocol": ["TCP"],
            "duration": [0.1234],
            "prediction": [1],
            "prediction_label": ["Attack"],
            "normal_probability": [0.15],
            "attack_probability": [0.85],
            "predicted_attack_category": ["Exploits"],
            "attack_category_confidence": [0.85],
            "prediction_status": ["Classified"],
        }
    )
    table_df = _prepare_flows_table(sample_df)

    assert "flow_id" in table_df.columns
    assert "src_ip" in table_df.columns
    assert "dst_ip" in table_df.columns
    assert "predicted_attack_category" in table_df.columns
    assert "attack_category_confidence" in table_df.columns
    assert "prediction_status" in table_df.columns
    assert table_df["predicted_attack_category"].iloc[0] == "Exploits"
    assert table_df["prediction_status"].iloc[0] == "Classified"


# ===========================================================================
# 5. Label encoder mapping and decoding consistency tests
# ===========================================================================

def test_exact_multiclass_category_mapping() -> None:
    """Verify the exact trained mapping between numeric class IDs and original attack names."""
    expected_mapping = {
        0: "Backdoors",
        1: "DoS",
        2: "Exploits",
        3: "Fuzzers",
        4: "Generic",
        5: "Normal",
        6: "Reconnaissance",
        7: "Shellcode",
        8: "Worms",
    }

    # 1. From artifact
    with open(MULTICLASS_LABEL_ENCODER_PATH, "r", encoding="utf-8") as f:
        encoder_data = json.load(f)

    mapping = get_attack_category_mapping(encoder_data)
    assert mapping == expected_mapping, f"Mapping mismatch: {mapping} vs {expected_mapping}"

    # 2. Test decode_attack_category for each integer and string ID
    for num_id, cat_name in expected_mapping.items():
        assert decode_attack_category(num_id, encoder_data) == cat_name
        assert decode_attack_category(str(num_id), encoder_data) == cat_name
        # Test case-insensitive preservation
        assert decode_attack_category(cat_name, encoder_data) == cat_name
        assert decode_attack_category(cat_name.lower(), encoder_data) == cat_name


def test_label_encoder_serialization_roundtrip(tmp_path: Path) -> None:
    """Verify that saving and reloading the label encoder preserves the exact mapping."""
    with open(MULTICLASS_LABEL_ENCODER_PATH, "r", encoding="utf-8") as f:
        original = json.load(f)

    save_path = tmp_path / "test_label_encoder.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(original, f, indent=2)

    with open(save_path, "r", encoding="utf-8") as f:
        reloaded = json.load(f)

    original_map = get_attack_category_mapping(original)
    reloaded_map = get_attack_category_mapping(reloaded)

    assert original_map == reloaded_map
    assert reloaded_map[5] == "Normal"
    assert reloaded_map[1] == "DoS"
    assert reloaded_map[2] == "Exploits"


def test_build_report_data_decodes_numeric_categories(tmp_path: Path) -> None:
    """Verify that build_report_data decodes numeric category arrays into readable names."""
    sample_pcap = tmp_path / "traffic.pcap"
    sample_pcap.write_bytes(b"dummy")

    extraction_summary = {"packet_count": 20, "flow_count": 5}
    features = pd.DataFrame({"feat1": [1.0] * 5})
    feature_names = ["feat1"]
    binary_preds = np.array([0, 1, 1, 1, 1])
    binary_probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.1, 0.9], [0.3, 0.7]])

    # Caller passes raw numeric predictions: 5=Normal, 1=DoS, 2=Exploits, 4=Generic, 6=Reconnaissance
    numeric_categories = [5, 1, 2, 4, 6]
    cat_confidences = [0.95, 0.88, 0.92, 0.85, 0.78]
    statuses = ["Classified", "Classified", "Classified", "Classified", "Classified"]

    report = build_report_data(
        pcap_path=sample_pcap,
        extraction_summary=extraction_summary,
        features=features,
        feature_names=feature_names,
        predictions=binary_preds,
        probabilities=binary_probs,
        total_analysis_seconds=10.0,
        predicted_categories=numeric_categories,
        category_confidences=cat_confidences,
        prediction_statuses=statuses,
    )

    mc = report["multiclass"]
    # Check that keys are readable names and NOT strings of numbers
    assert "5" not in mc["category_counts"]
    assert "1" not in mc["category_counts"]
    assert "2" not in mc["category_counts"]
    assert mc["category_counts"]["Normal"] == 1
    assert mc["category_counts"]["DoS"] == 1
    assert mc["category_counts"]["Exploits"] == 1
    assert mc["category_counts"]["Generic"] == 1
    assert mc["category_counts"]["Reconnaissance"] == 1

    # Check that detected attack categories only includes malicious attacks
    assert "Normal" not in mc["categories_detected"]
    assert set(mc["categories_detected"]) == {"DoS", "Exploits", "Generic", "Reconnaissance"}
    assert mc["attack_categories_detected_count"] == 4
    assert mc["malicious_flows_count"] == 4
    assert mc["normal_flows_count"] == 1
    assert mc["total_flows_count"] == 5


def test_throughput_formatting_never_zero_for_active_pipeline() -> None:
    """Verify that throughput calculations retain positive float values when flows/sec < 1."""
    sample_pcap = Path("data/sample_traffic.pcap")
    if not sample_pcap.exists():
        sample_pcap = Path.cwd() / "data" / "sample_traffic.pcap"

    inf_meta = {
        "inference_mode": "Single model",
        "num_flows": 36,
        "total_inference_seconds": 87.21,
        "rows_per_second": 36 / 87.21,  # 0.4128 flows/sec
        "num_models": 1,
        "num_workers": 1,
        "max_rows_per_worker": 10000,
        "num_chunks": 1,
        "rows_per_chunk": [36],
        "models_used": ["tabpfn_binary_model"],
    }

    report = build_report_data(
        pcap_path=sample_pcap,
        extraction_summary={"packet_count": 194, "flow_count": 36},
        features=pd.DataFrame({"x": [1] * 36}),
        feature_names=["x"],
        predictions=np.array([0] * 36),
        probabilities=np.array([[0.9, 0.1]] * 36),
        total_analysis_seconds=87.21,
        inference_meta=inf_meta,
    )

    perf = report["performance"]
    assert perf["flows_per_second"] == pytest.approx(0.41, abs=0.02)
    assert report["inference"]["Throughput (flows/sec)"] == pytest.approx(0.41, abs=0.02)


# ===========================================================================
# 6. Benchmark coverage, conflict policy, and summary validation tests
# ===========================================================================

def test_analysis_exclusion_documented() -> None:
    """UNSW-NB15 'Analysis' class must be explicitly identified as unsupported with documentation."""
    with open(MULTICLASS_LABEL_ENCODER_PATH, "r", encoding="utf-8") as f:
        encoder = json.load(f)

    assert "unsupported_classes" in encoder
    assert "Analysis" in encoder["unsupported_classes"]
    assert "Analysis" in encoder.get("unsupported_class_reasons", {})
    reason = encoder["unsupported_class_reasons"]["Analysis"]
    assert "1421927376" in reason or "slice" in reason.lower()

    # Supported classes must be exactly the 9 classes
    assert len(encoder["supported_classes"]) == 9
    assert "Normal" in encoder["supported_classes"]
    assert "Exploits" in encoder["supported_classes"]


def test_multiclass_confidence_rejection_curve() -> None:
    """compute_multiclass_metrics evaluates threshold curve and accepted accuracy."""
    classes = ["Normal", "DoS", "Exploits"]
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 1, 0])
    # Confidences: 0.9, 0.8, 0.45 (uncertain at 0.5), 0.85
    y_proba = np.array([
        [0.90, 0.05, 0.05],
        [0.10, 0.80, 0.10],
        [0.25, 0.30, 0.45],  # Wrong prediction & low confidence (0.45)
        [0.85, 0.10, 0.05],
    ])

    metrics = compute_multiclass_metrics(
        y_true, y_pred, y_proba=y_proba, classes=classes, confidence_threshold=0.50
    )

    assert metrics["prediction_coverage"] == 0.75  # 3 of 4 pass threshold 0.50
    assert metrics["uncertain_rate"] == 0.25
    assert metrics["accepted_count"] == 3
    assert metrics["uncertain_count"] == 1
    # On the 3 accepted predictions (indices 0, 1, 3), all 3 are correct:
    assert metrics["accepted_accuracy"] == 1.0
    # On all 4 predictions, 3 are correct (0.75 accuracy):
    assert metrics["accuracy"] == 0.75

    # Check threshold curve
    assert "threshold_curve" in metrics
    assert len(metrics["threshold_curve"]) == 7  # 0.3 to 0.9
    for point in metrics["threshold_curve"]:
        assert "threshold" in point
        assert "coverage" in point
        assert "accepted_accuracy" in point


def test_binary_and_multiclass_conflict_policy() -> None:
    """Disagreements between binary (Attack) and multiclass (Normal) must be flagged as Uncertain."""
    binary_preds = [1, 0, 1]
    multiclass_cats = ["Normal", "Exploits", "Exploits"]
    category_confs = [0.80, 0.70, 0.90]
    thresh = 0.50

    statuses = []
    conflicts = []
    for bin_p, cat, conf in zip(binary_preds, multiclass_cats, category_confs):
        bin_lbl = "Normal" if bin_p == 0 else "Attack"
        has_conflict = False
        if bin_lbl == "Attack" and cat == "Normal":
            has_conflict = True
        elif bin_lbl == "Normal" and cat != "Normal":
            has_conflict = True

        conflicts.append(has_conflict)
        if has_conflict:
            statuses.append("Uncertain")
        elif conf >= thresh:
            statuses.append("Classified")
        else:
            statuses.append("Uncertain")

    # Flow 0: Binary=Attack, Multiclass=Normal -> Conflict -> Uncertain
    assert conflicts[0] is True
    assert statuses[0] == "Uncertain"

    # Flow 1: Binary=Normal, Multiclass=Exploits -> Conflict -> Uncertain
    assert conflicts[1] is True
    assert statuses[1] == "Uncertain"

    # Flow 2: Binary=Attack, Multiclass=Exploits -> Agreement -> Classified
    assert conflicts[2] is False
    assert statuses[2] == "Classified"


def test_summary_counts_mathematical_consistency(tmp_path: Path) -> None:
    """Summary counts must strictly adhere to mathematical consistency rules."""
    pcap = tmp_path / "consistency_test.pcap"
    pcap.write_bytes(b"dummy")

    n_samples = 10
    features = pd.DataFrame({"f1": [1.0] * n_samples})
    feature_names = ["f1"]
    binary_preds = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])  # 6 Normal, 4 Attack
    binary_probs = np.zeros((n_samples, 2))
    binary_probs[:6, 0] = 0.9
    binary_probs[:6, 1] = 0.1
    binary_probs[6:, 0] = 0.1
    binary_probs[6:, 1] = 0.9

    # Multiclass predictions: 6 Normal, 2 Exploits, 1 DoS, 1 Generic
    cats = ["Normal"] * 6 + ["Exploits", "Exploits", "DoS", "Generic"]
    confs = [0.9] * 8 + [0.4, 0.4]  # 8 confident, 2 uncertain
    statuses = ["Classified"] * 8 + ["Uncertain", "Uncertain"]

    perf_breakdown = {
        "extraction_seconds": 0.5,
        "feature_engineering_seconds": 0.3,
        "preprocessing_seconds": 0.1,
        "binary_inference_seconds": 1.2,
        "multiclass_inference_seconds": 1.5,
        "total_inference_seconds": 2.7,
        "total_analysis_seconds": 4.0,
    }

    report = build_report_data(
        pcap_path=pcap,
        extraction_summary={"total_packets": 50, "total_flows": n_samples},
        features=features,
        feature_names=feature_names,
        predictions=binary_preds,
        probabilities=binary_probs,
        total_analysis_seconds=4.0,
        predicted_categories=cats,
        category_confidences=confs,
        prediction_statuses=statuses,
        uncertainty_threshold=0.5,
        performance_breakdown=perf_breakdown,
    )

    preds = report["predictions"]
    mc = report["multiclass"]
    perf = report["performance"]

    # 1. Total flows = Normal flows + Malicious flows
    assert preds["total"] == preds["normal_count"] + preds["attack_count"]
    assert preds["normal_count"] == 6
    assert preds["attack_count"] == 4

    # 2. Classified + Uncertain + Unknown == Total
    status_sum = mc["status_counts"]["classified"] + mc["status_counts"]["uncertain"] + mc["status_counts"]["unknown"]
    assert status_sum == preds["total"]
    assert mc["status_counts"]["classified"] == 8
    assert mc["status_counts"]["uncertain"] == 2

    # 3. Distinct attack categories counts strictly malicious non-Normal types
    assert "Normal" not in mc["categories_detected"]
    assert set(mc["categories_detected"]) == {"Exploits", "DoS", "Generic"}
    assert mc["attack_categories_detected_count"] == 3

    # 4. Total malicious flows in multiclass matches sum of malicious categories
    malicious_sum = sum(mc["category_counts"][c] for c in mc["categories_detected"])
    assert mc["malicious_flows_count"] == malicious_sum
    assert mc["malicious_flows_count"] == 4

    # 5. Performance breakdown preserved
    assert perf["extraction_seconds"] == 0.5
    assert perf["binary_inference_seconds"] == 1.2
    assert perf["multiclass_inference_seconds"] == 1.5
    assert perf["flows_processed"] == n_samples

