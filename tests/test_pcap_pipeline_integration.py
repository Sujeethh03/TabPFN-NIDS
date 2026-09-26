"""Integration tests for PCAP analysis pipeline with Dynamic Inference Manager.

Validates end-to-end execution from PCAP through flow extraction, feature engineering,
preprocessing, dynamic chunked inference, and report generation (CSV, JSON, HTML, DOCX).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# pyrefly: ignore [missing-import]
from scripts.analyze_pcap import (
    build_prediction_output,
    extract_and_build_features,
    load_artifacts,
    prepare_model_input,
    save_results,
)
from tabpfn_nids.config import PROJECT_ROOT
from tabpfn_nids.inference import (
    DynamicInferenceManager,
    ModelInfo,
    ModelRegistry,
)
from tabpfn_nids.pcap.analysis_report import (
    build_report_data,
    save_docx_report,
    save_html_report,
    save_json_report,
)


@pytest.fixture
def sample_pcap() -> Path:
    pcap = PROJECT_ROOT / "data" / "sample_traffic.pcap"
    assert pcap.exists(), f"Sample PCAP not found at {pcap}"
    return pcap


def test_pcap_pipeline_end_to_end_with_chunking(sample_pcap: Path, tmp_path: Path) -> None:
    """Test full pipeline with forced small chunk size to verify parallel chunking end-to-end."""
    work_dir = tmp_path / "sample_analysis"
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load artifacts
    model, scaler, feature_names = load_artifacts()
    assert len(feature_names) == 67

    # 2. Extract flows and build features
    features, summary = extract_and_build_features(sample_pcap, work_dir)
    assert len(features) > 0
    assert summary["total_flows"] == len(features)

    # 3. Preprocess
    metadata, X = prepare_model_input(features, feature_names, scaler)
    assert X.shape[1] == 67
    assert len(X) == len(features)

    original_features = features[feature_names].copy()

    # 4. Run dynamic inference with forced small chunk size (e.g. 10 rows per chunk)
    # This exercises chunk creation, multi-chunk execution, and reassembly on real PCAP data
    registry = ModelRegistry()
    registry.register_model(
        ModelInfo(
            model_id="tabpfn_binary_model",
            model_path=PROJECT_ROOT / "data" / "artifacts" / "models" / "tabpfn_binary_model.pkl",
            feature_count=67,
            feature_names=feature_names,
        ),
        model_instance=model,
    )

    manager = DynamicInferenceManager(
        max_rows_per_worker=10,
        enable_ensemble=False,
        prediction_threshold=0.5,
        model_registry=registry,
    )

    inf_result = manager.predict(X)

    assert len(inf_result.predictions) == len(X)
    assert inf_result.metadata["num_chunks"] == 4  # 36 rows with chunk_size 10 = 4 chunks
    assert inf_result.metadata["inference_mode"] in ("Parallel", "Single model")
    assert inf_result.probabilities.shape == (len(X), 2)

    # 5. Build output
    output = build_prediction_output(
        metadata=metadata,
        original_features=original_features,
        predictions=inf_result.predictions,
        probabilities=inf_result.probabilities,
        inference_meta=inf_result.metadata,
    )

    # Verify all expected columns exist
    assert "flow_id" in output.columns
    for feat in feature_names:
        assert feat in output.columns
    assert "prediction" in output.columns
    assert "prediction_label" in output.columns
    assert "normal_probability" in output.columns
    assert "attack_probability" in output.columns

    # 6. Save CSV
    csv_path = save_results(output, sample_pcap, work_dir)
    assert csv_path.exists()
    df_saved = pd.read_csv(csv_path)
    assert len(df_saved) == len(output)

    # 7. Generate reports
    report_data = build_report_data(
        pcap_path=sample_pcap,
        extraction_summary=summary,
        features=features,
        feature_names=feature_names,
        predictions=inf_result.predictions,
        probabilities=inf_result.probabilities,
        total_analysis_seconds=5.0,
        inference_meta=inf_result.metadata,
    )

    assert "inference" in report_data
    assert report_data["inference"]["Chunk size"] == 10
    assert report_data["inference"]["Number of chunks"] == 4

    json_path = work_dir / "report.json"
    save_json_report(report_data, json_path)
    assert json_path.exists()

    html_path = work_dir / "report.html"
    save_html_report(report_data, html_path)
    assert html_path.exists()

    docx_path = work_dir / "report.docx"
    save_docx_report(report_data, docx_path)
    assert docx_path.exists()
