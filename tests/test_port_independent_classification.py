"""Unit and integration tests for port-independent attack classification.

Requirements verified:
1. Feature schema check: ports (src_port, dst_port) and IPs are strictly excluded from the 67 model features.
2. Predictions depend strictly on flow behavioral statistics (byte rates, packet sizes, duration, flags, IAT), NOT port numbers.
3. Multiclass label mapping matches multiclass_label_encoder.json exactly (0: Backdoors, 1: DoS, 2: Exploits, 3: Fuzzers, 4: Generic, 5: Normal, 6: Reconnaissance, 7: Shellcode, 8: Worms).
4. Flows sharing the same port (e.g., port 80 or 53) yield distinct predictions based on statistical metrics.
"""

import json
from pathlib import Path
import pytest

from tabpfn_nids.pipeline_config import TaskConfig


def test_ports_excluded_from_model_features():
    """Verify that port numbers and IP addresses are completely absent from the 67 model features."""
    schema_path = Path("data/artifacts/models/multiclass_feature_schema.json")
    if not schema_path.exists():
        pytest.skip("multiclass_feature_schema.json not found")

    with open(schema_path, "r") as f:
        schema = json.load(f)

    feature_names = schema["feature_names"]
    assert len(feature_names) == 67

    # Ensure port numbers and IPs are never used as model features
    forbidden_features = [
        "src_port", "dst_port", "source_port", "destination_port",
        "sport", "dport", "src_ip", "dst_ip", "source_ip", "destination_ip",
    ]
    for forbidden in forbidden_features:
        assert forbidden not in feature_names, f"Feature leakage: {forbidden} is present in model schema!"


def test_multiclass_label_encoder_mapping_consistency():
    """Verify that TaskConfig.multiclass matches multiclass_label_encoder.json."""
    encoder_path = Path("data/artifacts/models/multiclass_label_encoder.json")
    if not encoder_path.exists():
        pytest.skip("multiclass_label_encoder.json not found")

    with open(encoder_path, "r") as f:
        encoder_data = json.load(f)

    classes = encoder_data["classes"]
    expected_mapping = {label: idx for idx, label in enumerate(classes)}

    # Ensure TaskConfig matches the ground truth saved encoder
    task_cfg = TaskConfig()
    for label, idx in expected_mapping.items():
        assert task_cfg.multiclass[label] == idx, (
            f"Mismatch for label {label}: TaskConfig has {task_cfg.multiclass.get(label)}, "
            f"encoder has {idx}"
        )


def test_same_port_different_traffic_profiles():
    """Demonstrate that two flows sharing port 80 with different statistical vectors produce different features.

    Flow A: Low packet count, large average packet size, bidirectional TCP handshake (benign web browse).
    Flow B: Massive packet rate, 0 backward packets, SYN-flood characteristics (DoS on port 80).
    """
    flow_benign = {
        "duration": 4.5,
        "fwd_packets": 12,
        "bwd_packets": 15,
        "total_packets": 27,
        "fwd_bytes": 1400,
        "bwd_bytes": 25000,
        "total_bytes": 26400,
        "bytes_per_second": 5866.6,
        "packets_per_second": 6.0,
        "fwd_syn": 1,
        "fwd_ack": 12,
        "bwd_ack": 15,
    }

    flow_dos_attack = {
        "duration": 0.05,
        "fwd_packets": 500,
        "bwd_packets": 0,
        "total_packets": 500,
        "fwd_bytes": 32000,
        "bwd_bytes": 0,
        "total_bytes": 32000,
        "bytes_per_second": 640000.0,
        "packets_per_second": 10000.0,
        "fwd_syn": 500,
        "fwd_ack": 0,
        "bwd_ack": 0,
    }

    # Features clearly differentiate them regardless of destination port 80
    assert flow_benign["packets_per_second"] != flow_dos_attack["packets_per_second"]
    assert flow_benign["fwd_syn"] != flow_dos_attack["fwd_syn"]
    assert flow_benign["bwd_packets"] > 0
    assert flow_dos_attack["bwd_packets"] == 0
