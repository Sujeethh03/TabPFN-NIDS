"""Critical PCAP compatibility test: verifies that live streaming pipeline produces

identical feature columns, ordering, preprocessing, and model predictions as the
existing offline PCAP pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_pcap import (
    extract_and_build_features,
    load_artifacts,
    prepare_model_input,
)
from tabpfn_nids.config import PROJECT_ROOT
from tabpfn_nids.detection.window_manager import DetectionWindow
from tabpfn_nids.flows.live_flow_manager import LiveFlowManager
from tabpfn_nids.inference.live_inference_engine import (
    LiveInferenceEngine,
    extract_and_preprocess_window,
)
from tabpfn_nids.pcap.extractor import extract_packets


@pytest.fixture
def sample_pcap() -> Path:
    pcap = PROJECT_ROOT / "data" / "sample_traffic.pcap"
    assert pcap.exists(), f"Sample PCAP not found at {pcap}"
    return pcap


def test_live_pipeline_pcap_feature_and_prediction_compatibility(
    sample_pcap: Path, tmp_path: Path
) -> None:
    """Verify live flow manager & feature extraction matches offline PCAP pipeline."""
    # -------------------------------------------------------------------------
    # 1. Run Existing Offline Pipeline
    # -------------------------------------------------------------------------
    model, scaler, feature_names = load_artifacts()
    assert len(feature_names) == 67

    work_dir = tmp_path / "offline_compat"
    work_dir.mkdir(parents=True, exist_ok=True)

    offline_features, offline_summary = extract_and_build_features(
        sample_pcap, work_dir
    )
    offline_metadata, X_offline = prepare_model_input(
        offline_features, feature_names, scaler
    )

    # Offline model predictions
    if hasattr(model, "predict_proba"):
        offline_probs = model.predict_proba(X_offline)
        offline_preds = np.argmax(offline_probs, axis=1)
    else:
        offline_preds = model.predict(X_offline)

    # -------------------------------------------------------------------------
    # 2. Run Live Streaming Pipeline with the Same Packets
    # -------------------------------------------------------------------------
    live_flow_mgr = LiveFlowManager(
        flow_timeout_seconds=120.0,
        idle_timeout_seconds=60.0,
    )

    # Stream packet records into live flow manager
    total_streamed_packets = 0
    for batch in extract_packets(sample_pcap, backend="scapy", batch_size=10_000):
        for pkt in batch:
            live_flow_mgr.add_packet(pkt)
            total_streamed_packets += 1

    assert total_streamed_packets == offline_summary["total_packets"]

    # Finalize flows
    live_flows = live_flow_mgr.flush_all()
    assert len(live_flows) == offline_summary["total_flows"]

    # Sort both by flow_id to align rows for direct comparison
    live_flows_sorted = sorted(live_flows, key=lambda f: f.flow_id)

    # Extract & preprocess using the live pipeline function
    live_meta, X_live = extract_and_preprocess_window(
        live_flows_sorted, feature_names, scaler
    )

    # -------------------------------------------------------------------------
    # 3. Compare Dimensions, Feature Names & Values
    # -------------------------------------------------------------------------
    assert X_live.shape == X_offline.shape, (
        f"Shape mismatch: live {X_live.shape} vs offline {X_offline.shape}"
    )

    # Sort offline rows by flow_id as well
    offline_order = offline_features.sort_values("flow_id").index
    X_offline_sorted = X_offline[offline_order]

    # Verify numerical compatibility (all values close within floating tolerance)
    np.testing.assert_allclose(
        X_live,
        X_offline_sorted,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Feature numerical values differ between live and offline pipelines!",
    )

    # -------------------------------------------------------------------------
    # 4. Compare Model Predictions
    # -------------------------------------------------------------------------
    # Offline predictions sorted
    offline_preds_sorted = offline_preds[offline_order]

    if hasattr(model, "predict_proba"):
        live_probs = model.predict_proba(X_live)
        live_preds = np.argmax(live_probs, axis=1)
        np.testing.assert_array_equal(
            live_preds,
            offline_preds_sorted,
            err_msg="Predictions differ between live and offline pipelines!",
        )
    else:
        live_preds = model.predict(X_live)
        np.testing.assert_array_equal(live_preds, offline_preds_sorted)

    print("\n[Compatibility Test Passed]")
    print(f"  Packets analyzed: {total_streamed_packets}")
    print(f"  Flows analyzed  : {len(live_flows)}")
    print(f"  Feature columns : {len(feature_names)} (Exact match)")
    print(f"  Predictions     : {len(live_preds)} matching offline predictions")
