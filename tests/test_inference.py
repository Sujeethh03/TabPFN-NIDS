"""Tests for dynamic parallel inference architecture (Requirement Section 23).

Tests:
1. Small dataset (5,000 rows, 1 model -> 1 worker)
2. Exactly 10,000 rows -> 1 worker
3. Just above threshold (10,001 rows -> 2 chunks)
4. 20,000 rows -> 2 chunks
5. 20,001 rows -> 3 chunks
6. 30,000 rows, 3 models -> ensemble aggregation
7. Single model fallback
8. Incompatible model (different feature schema) -> clear compatibility error
9. Probability aggregation with deterministic mock probabilities
10. Output preservation (CSV, JSON, HTML, Word)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from tabpfn_nids.inference import (
    AggregatedPredictions,
    DynamicInferenceManager,
    IncompatibleModelError,
    InferenceChunk,
    InferenceManager,
    ModelInfo,
    ModelRegistry,
    ModelWorker,
    PredictionAggregator,
    calculate_worker_count,
)
from tabpfn_nids.pcap.analysis_report import (
    build_report_data,
    save_docx_report,
    save_html_report,
    save_json_report,
)


class MockTabPFNModel:
    """Mock model that returns deterministic probabilities for testing."""

    def __init__(self, attack_prob: float = 0.5, n_classes: int = 2) -> None:
        self.attack_prob = attack_prob
        self.n_classes = n_classes
        self.call_count = 0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.call_count += 1
        n = len(X)
        p_atk = np.full(n, self.attack_prob, dtype=np.float64)
        p_norm = 1.0 - p_atk
        return np.column_stack([p_norm, p_atk])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


# ---------------------------------------------------------------------------
# Test 1 & 2 & 3 & 4 & 5: Worker calculation and chunking
# ---------------------------------------------------------------------------


def test_1_small_dataset_worker_count() -> None:
    """Test 1: 5,000 rows -> 1 worker."""
    workers = calculate_worker_count(5_000, max_rows_per_worker=10_000)
    assert workers == 1

    mgr = DynamicInferenceManager(max_rows_per_worker=10_000)
    chunks = mgr.create_chunks(5_000)
    assert len(chunks) == 1
    assert chunks[0].start_idx == 0
    assert chunks[0].end_idx == 5_000
    assert chunks[0].row_count == 5_000


def test_2_exactly_10000_rows_worker_count() -> None:
    """Test 2: Exactly 10,000 rows -> 1 worker."""
    workers = calculate_worker_count(10_000, max_rows_per_worker=10_000)
    assert workers == 1

    mgr = DynamicInferenceManager(max_rows_per_worker=10_000)
    chunks = mgr.create_chunks(10_000)
    assert len(chunks) == 1
    assert chunks[0].start_idx == 0
    assert chunks[0].end_idx == 10_000


def test_3_just_above_threshold_worker_count() -> None:
    """Test 3: 10,001 rows -> 2 chunks / 2 workers."""
    workers = calculate_worker_count(10_001, max_rows_per_worker=10_000)
    assert workers == 2

    mgr = DynamicInferenceManager(max_rows_per_worker=10_000)
    chunks = mgr.create_chunks(10_001)
    assert len(chunks) == 2
    assert chunks[0].row_count == 10_000
    assert chunks[1].row_count == 1
    assert chunks[0].start_idx == 0 and chunks[0].end_idx == 10_000
    assert chunks[1].start_idx == 10_000 and chunks[1].end_idx == 10_001


def test_4_20000_rows_worker_count() -> None:
    """Test 4: 20,000 rows -> 2 chunks / 2 workers."""
    workers = calculate_worker_count(20_000, max_rows_per_worker=10_000)
    assert workers == 2

    mgr = DynamicInferenceManager(max_rows_per_worker=10_000)
    chunks = mgr.create_chunks(20_000)
    assert len(chunks) == 2
    assert chunks[0].row_count == 10_000
    assert chunks[1].row_count == 10_000


def test_5_20001_rows_worker_count() -> None:
    """Test 5: 20,001 rows -> 3 chunks / 3 workers."""
    workers = calculate_worker_count(20_001, max_rows_per_worker=10_000)
    assert workers == 3

    mgr = DynamicInferenceManager(max_rows_per_worker=10_000)
    chunks = mgr.create_chunks(20_001)
    assert len(chunks) == 3
    assert chunks[0].row_count == 10_000
    assert chunks[1].row_count == 10_000
    assert chunks[2].row_count == 1


def test_worker_count_respects_max_workers() -> None:
    """Worker count is clamped when max_workers is explicitly specified."""
    assert calculate_worker_count(50_000, max_rows_per_worker=10_000, max_workers=2) == 2
    assert calculate_worker_count(5_000, max_rows_per_worker=10_000, max_workers=4) == 1


# ---------------------------------------------------------------------------
# Test 6: 30,000 rows with 3 models (combined large data + ensemble)
# ---------------------------------------------------------------------------


def test_6_30000_rows_3_models_ensemble() -> None:
    """Test 6: 30,000 rows, 3 models -> 3 chunks, true ensemble averaging."""
    X = np.random.randn(30_000, 10)

    model1 = MockTabPFNModel(attack_prob=0.8)
    model2 = MockTabPFNModel(attack_prob=0.6)
    model3 = MockTabPFNModel(attack_prob=0.7)

    models = {
        "m1": model1,
        "m2": model2,
        "m3": model3,
    }

    mgr = DynamicInferenceManager(
        max_rows_per_worker=10_000,
        enable_ensemble=True,
        probability_aggregation="mean",
        prediction_threshold=0.5,
    )

    res = mgr.predict(X, models=models)

    assert len(res.predictions) == 30_000
    assert res.metadata["num_chunks"] == 3
    assert res.metadata["num_models"] == 3
    assert res.metadata["inference_mode"] in ("Parallel Ensemble", "Ensemble")

    # Expected average attack prob = (0.8 + 0.6 + 0.7) / 3 = 0.7
    np.testing.assert_allclose(res.attack_probabilities, 0.7, atol=1e-5)
    np.testing.assert_allclose(res.normal_probabilities, 0.3, atol=1e-5)

    # 0.7 >= 0.5 threshold -> all Attack (1)
    assert np.all(res.predictions == 1)
    assert all(label == "Attack" for label in res.prediction_labels)

    # Verify each model was called for each chunk (3 chunks x 3 models = 9 calls total)
    assert model1.call_count == 3
    assert model2.call_count == 3
    assert model3.call_count == 3


# ---------------------------------------------------------------------------
# Test 7: Single model fallback
# ---------------------------------------------------------------------------


def test_7_single_model_fallback() -> None:
    """Test 7: Only 1 model exists -> executes in single-model mode without false ensembling."""
    X = np.random.randn(25_000, 10)
    model1 = MockTabPFNModel(attack_prob=0.2)

    mgr = DynamicInferenceManager(
        max_rows_per_worker=10_000,
        enable_ensemble=True,
    )

    res = mgr.predict(X, models={"only_model": model1})

    assert len(res.predictions) == 25_000
    assert res.metadata["num_models"] == 1
    assert res.metadata["num_chunks"] == 3
    assert res.metadata["inference_mode"] == "Parallel"
    assert res.metadata["ensemble_method"] == "Single model"

    # Probabilities should match single model directly: attack 0.2, normal 0.8
    np.testing.assert_allclose(res.attack_probabilities, 0.2, atol=1e-5)
    np.testing.assert_allclose(res.normal_probabilities, 0.8, atol=1e-5)
    assert np.all(res.predictions == 0)
    assert all(label == "Normal" for label in res.prediction_labels)


# ---------------------------------------------------------------------------
# Test 8: Incompatible model detection
# ---------------------------------------------------------------------------


def test_8_incompatible_model_schema_raises_error() -> None:
    """Test 8: Models with differing feature count or schemas must raise IncompatibleModelError."""
    registry = ModelRegistry()

    m1 = ModelInfo(
        model_id="model_67_features",
        model_path=Path("dummy_1.pkl"),
        feature_count=67,
        feature_names=[f"f_{i}" for i in range(67)],
        classes=["Normal", "Attack"],
        task="binary",
    )
    m2 = ModelInfo(
        model_id="model_60_features",
        model_path=Path("dummy_2.pkl"),
        feature_count=60,
        feature_names=[f"f_{i}" for i in range(60)],
        classes=["Normal", "Attack"],
        task="binary",
    )

    registry.register_model(m1, model_instance=MockTabPFNModel())
    registry.register_model(m2, model_instance=MockTabPFNModel())

    with pytest.raises(IncompatibleModelError, match="feature count"):
        registry.validate_compatibility([m1, m2])


def test_8_incompatible_class_mapping_raises_error() -> None:
    """Models with mismatched class mapping must raise IncompatibleModelError."""
    registry = ModelRegistry()

    m1 = ModelInfo(
        model_id="m1",
        model_path=Path("d1.pkl"),
        feature_count=67,
        classes=["Normal", "Attack"],
    )
    m2 = ModelInfo(
        model_id="m2",
        model_path=Path("d2.pkl"),
        feature_count=67,
        classes=["Attack", "Normal"],  # Inverted mapping!
    )

    registry.register_model(m1)
    registry.register_model(m2)

    with pytest.raises(IncompatibleModelError, match="class mapping"):
        registry.validate_compatibility([m1, m2])


# ---------------------------------------------------------------------------
# Test 9: Probability aggregation and thresholding
# ---------------------------------------------------------------------------


def test_9_probability_aggregation_mean_and_threshold() -> None:
    """Test 9: Model 1 = 0.8, Model 2 = 0.6, Model 3 = 0.7 -> mean 0.7 -> Attack at 0.5."""
    aggregator = PredictionAggregator(method="mean", threshold=0.5)

    # 5 rows
    m1_probs = np.array([[0.2, 0.8]] * 5)
    m2_probs = np.array([[0.4, 0.6]] * 5)
    m3_probs = np.array([[0.3, 0.7]] * 5)

    model_probs = {
        "m1": m1_probs,
        "m2": m2_probs,
        "m3": m3_probs,
    }

    res: AggregatedPredictions = aggregator.aggregate(model_probs)

    np.testing.assert_allclose(res.attack_probabilities, 0.7, atol=1e-5)
    np.testing.assert_allclose(res.normal_probabilities, 0.3, atol=1e-5)
    assert np.all(res.predictions == 1)
    assert res.prediction_labels == ["Attack"] * 5
    assert res.ensemble_size == 3
    assert res.ensemble_method == "mean"


def test_9_weighted_probability_aggregation() -> None:
    """Test weighted probability aggregation."""
    weights = {"m1": 0.5, "m2": 0.5}
    aggregator = PredictionAggregator(method="weighted", threshold=0.5, weights=weights)

    m1_probs = np.array([[0.9, 0.1]])
    m2_probs = np.array([[0.5, 0.5]])

    res = aggregator.aggregate({"m1": m1_probs, "m2": m2_probs})
    # 0.5 * 0.1 + 0.5 * 0.5 = 0.30 attack prob
    np.testing.assert_allclose(res.attack_probabilities, [0.3], atol=1e-5)
    assert res.predictions[0] == 0
    assert res.prediction_labels[0] == "Normal"


# ---------------------------------------------------------------------------
# Test 10: Output preservation (CSV, JSON, HTML, DOCX)
# ---------------------------------------------------------------------------


def test_10_output_preservation(tmp_path: Path) -> None:
    """Test 10: Ensure CSV, JSON, HTML, and Word outputs are correctly produced."""
    pcap_path = tmp_path / "test.pcap"
    pcap_path.write_bytes(b"dummy pcap content")

    extraction_summary = {
        "total_packets": 100,
        "total_flows": 10,
        "backend": "scapy",
    }

    feature_names = [f"feat_{i}" for i in range(67)]
    features = pd.DataFrame(
        np.zeros((10, 67)),
        columns=feature_names,
    )
    features["flow_id"] = [f"id_{i}" for i in range(10)]

    predictions = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 0])
    probabilities = np.column_stack([1.0 - predictions * 0.8, predictions * 0.8])

    inference_meta = {
        "inference_mode": "Parallel Ensemble",
        "num_flows": 10,
        "num_models": 2,
        "num_workers": 2,
        "max_rows_per_worker": 5,
        "num_chunks": 2,
        "rows_per_chunk": [5, 5],
        "ensemble_method": "Mean probability",
        "prediction_threshold": 0.5,
        "models_used": ["model_1", "model_2"],
        "total_inference_seconds": 1.25,
        "avg_model_inference_seconds": 0.62,
    }

    # 1. Build report data
    report_data = build_report_data(
        pcap_path=pcap_path,
        extraction_summary=extraction_summary,
        features=features,
        feature_names=feature_names,
        predictions=predictions,
        probabilities=probabilities,
        total_analysis_seconds=2.5,
        inference_meta=inference_meta,
    )

    assert "inference" in report_data
    assert report_data["inference"]["Inference mode"] == "Parallel Ensemble"
    assert report_data["inference"]["Number of models"] == 2
    assert report_data["inference"]["Number of workers"] == 2

    # 2. JSON report
    json_path = tmp_path / "report.json"
    save_json_report(report_data, json_path)
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        saved_json = json.load(f)
    assert saved_json["inference"]["Inference mode"] == "Parallel Ensemble"

    # 3. HTML report
    html_path = tmp_path / "report.html"
    save_html_report(report_data, html_path)
    assert html_path.exists()
    html_content = html_path.read_text(encoding="utf-8")
    assert "Inference Architecture &amp; Execution" in html_content
    assert "Parallel Ensemble" in html_content

    # 4. DOCX report
    docx_path = tmp_path / "report.docx"
    save_docx_report(report_data, docx_path)
    assert docx_path.exists()
    assert docx_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Test 11: Worker error handling
# ---------------------------------------------------------------------------


def test_worker_failure_raises_error() -> None:
    """Worker failure is not swallowed; raises descriptive RuntimeError."""
    failing_model = MagicMock()
    failing_model.predict_proba.side_effect = RuntimeError("GPU out of memory")

    mgr = DynamicInferenceManager(max_rows_per_worker=100)
    X = np.random.randn(200, 5)

    with pytest.raises(RuntimeError, match="GPU out of memory"):
        mgr.predict(X, models={"fail_model": failing_model})
