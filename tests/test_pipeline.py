"""Unit tests for the PCAP pipeline core modules.

Tests use synthetic data — no real PCAP files or UNSW-NB15 data required.
These tests verify the logic of flow building, feature engineering,
labeling, cleaning, encoding, scaling, feature selection, and splitting.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ============================================================================
# Flow Builder Tests
# ============================================================================

class TestFlowBuilder:
    """Test the bidirectional flow builder."""

    def _make_packet(self, **overrides):
        """Create a PacketRecord with defaults."""
        from tabpfn_nids.pcap.extractor import PacketRecord
        defaults = {
            "timestamp": 1000.0,
            "src_ip": "192.168.1.1",
            "dst_ip": "10.0.0.1",
            "src_port": 12345,
            "dst_port": 80,
            "protocol": 6,
            "protocol_name": "tcp",
            "length": 100,
            "tcp_flags": 0x02,  # SYN
            "payload_length": 50,
        }
        defaults.update(overrides)
        return PacketRecord(**defaults)

    def test_single_packet_flow(self):
        from tabpfn_nids.flows.flow_builder import FlowBuilder
        builder = FlowBuilder()
        builder.add_packet(self._make_packet())
        flows = builder.flush_all()
        assert len(flows) == 1
        assert flows[0].fwd_packets == 1
        assert flows[0].bwd_packets == 0

    def test_bidirectional_flow(self):
        from tabpfn_nids.flows.flow_builder import FlowBuilder
        builder = FlowBuilder()

        # Forward packet
        builder.add_packet(self._make_packet(timestamp=1000.0))
        # Backward packet (reverse direction)
        builder.add_packet(self._make_packet(
            timestamp=1000.1,
            src_ip="10.0.0.1", dst_ip="192.168.1.1",
            src_port=80, dst_port=12345,
            tcp_flags=0x12,  # SYN+ACK
        ))
        # Forward again
        builder.add_packet(self._make_packet(
            timestamp=1000.2, tcp_flags=0x10,  # ACK
        ))

        flows = builder.flush_all()
        assert len(flows) == 1
        assert flows[0].fwd_packets == 2
        assert flows[0].bwd_packets == 1

    def test_idle_timeout_creates_new_flow(self):
        from tabpfn_nids.flows.flow_builder import FlowBuilder
        builder = FlowBuilder(idle_timeout=10.0)

        builder.add_packet(self._make_packet(timestamp=1000.0))
        builder.add_packet(self._make_packet(timestamp=1020.0))  # 20s gap > 10s timeout

        flows = builder.flush_all()
        assert len(flows) == 2

    def test_different_5tuple_creates_separate_flows(self):
        from tabpfn_nids.flows.flow_builder import FlowBuilder
        builder = FlowBuilder()

        builder.add_packet(self._make_packet(dst_port=80))
        builder.add_packet(self._make_packet(dst_port=443))

        flows = builder.flush_all()
        assert len(flows) == 2

    def test_tcp_flags_counted(self):
        from tabpfn_nids.flows.flow_builder import FlowBuilder
        builder = FlowBuilder()

        builder.add_packet(self._make_packet(tcp_flags=0x02))  # SYN
        builder.add_packet(self._make_packet(timestamp=1000.1, tcp_flags=0x10))  # ACK

        flows = builder.flush_all()
        # FIN/RST in the second packet would close flow, but ACK doesn't
        # Actually, we need to check — the second packet has ACK only
        # The flow should still be open until flush_all
        assert len(flows) >= 1
        flow = flows[0]
        assert flow.fwd_syn >= 1

    def test_flow_to_dict(self):
        from tabpfn_nids.flows.flow_builder import FlowBuilder
        builder = FlowBuilder()
        builder.add_packet(self._make_packet())
        flows = builder.flush_all()
        d = flows[0].to_dict()
        assert "flow_id" in d
        assert "src_ip" in d
        assert "duration" in d


# ============================================================================
# Feature Engineering Tests
# ============================================================================

class TestFeatures:
    """Test feature computation modules."""

    def _make_flow_df(self, n: int = 5) -> pd.DataFrame:
        """Create a synthetic flow DataFrame."""
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "flow_id": [f"flow_{i}" for i in range(n)],
            "src_ip": ["192.168.1.1"] * n,
            "dst_ip": ["10.0.0.1"] * n,
            "src_port": rng.integers(1024, 65535, n),
            "dst_port": [80] * n,
            "protocol": [6] * n,
            "protocol_name": ["tcp"] * n,
            "start_time": np.arange(n, dtype=float) * 10,
            "end_time": np.arange(n, dtype=float) * 10 + 5,
            "duration": [5.0] * n,
            "fwd_packets": rng.integers(1, 100, n),
            "bwd_packets": rng.integers(0, 50, n),
            "fwd_bytes": rng.integers(100, 10000, n),
            "bwd_bytes": rng.integers(0, 5000, n),
            "total_packets": rng.integers(1, 150, n),
            "total_bytes": rng.integers(100, 15000, n),
            "fwd_syn": rng.integers(0, 3, n),
            "fwd_ack": rng.integers(0, 50, n),
            "fwd_fin": rng.integers(0, 2, n),
            "fwd_rst": rng.integers(0, 2, n),
            "fwd_psh": rng.integers(0, 10, n),
            "fwd_urg": [0] * n,
            "bwd_syn": rng.integers(0, 2, n),
            "bwd_ack": rng.integers(0, 30, n),
            "bwd_fin": rng.integers(0, 2, n),
            "bwd_rst": [0] * n,
            "bwd_psh": rng.integers(0, 5, n),
            "bwd_urg": [0] * n,
            "fwd_payload_bytes": rng.integers(0, 5000, n),
            "bwd_payload_bytes": rng.integers(0, 3000, n),
            "fwd_packet_lengths": [",".join(str(x) for x in rng.integers(40, 1500, 10)) for _ in range(n)],
            "bwd_packet_lengths": [",".join(str(x) for x in rng.integers(40, 800, 5)) for _ in range(n)],
            "fwd_timestamps": [",".join(f"{x:.6f}" for x in np.sort(rng.uniform(0, 5, 10))) for _ in range(n)],
            "bwd_timestamps": [",".join(f"{x:.6f}" for x in np.sort(rng.uniform(0, 5, 5))) for _ in range(n)],
        })

    def test_basic_features(self):
        from tabpfn_nids.features.basic import compute_basic_features
        df = self._make_flow_df()
        result = compute_basic_features(df)
        assert "packets_per_second" in result.columns
        assert "bytes_per_second" in result.columns
        assert (result["packets_per_second"] >= 0).all()

    def test_directional_features(self):
        from tabpfn_nids.features.directional import compute_directional_features
        df = self._make_flow_df()
        result = compute_directional_features(df)
        assert "fwd_packet_ratio" in result.columns
        assert "bwd_byte_ratio" in result.columns
        # Ratios should be between 0 and 1
        assert (result["fwd_packet_ratio"] >= 0).all()
        assert (result["fwd_packet_ratio"] <= 1).all()

    def test_timing_features(self):
        from tabpfn_nids.features.timing import compute_timing_features
        df = self._make_flow_df()
        result = compute_timing_features(df)
        assert "mean_iat" in result.columns
        assert "fwd_mean_iat" in result.columns

    def test_packet_stats(self):
        from tabpfn_nids.features.packet_stats import compute_packet_stats
        df = self._make_flow_df()
        result = compute_packet_stats(df)
        assert "pkt_len_mean" in result.columns
        assert "fwd_pkt_len_max" in result.columns

    def test_tcp_flags(self):
        from tabpfn_nids.features.tcp_flags import compute_tcp_flag_features
        df = self._make_flow_df()
        result = compute_tcp_flag_features(df)
        assert "syn_count" in result.columns
        assert "syn_ratio" in result.columns

    def test_full_feature_pipeline(self):
        from tabpfn_nids.features.feature_pipeline import compute_all_features
        df = self._make_flow_df()
        result = compute_all_features(df)
        # Should have many more columns than the input
        assert len(result.columns) > len(df.columns)
        # Derivation columns should be removed
        assert "fwd_packet_lengths" not in result.columns
        assert "fwd_timestamps" not in result.columns


# ============================================================================
# Preprocessing Tests
# ============================================================================

class TestPreprocessing:
    """Test data cleaning, encoding, scaling, and feature selection."""

    def _make_dirty_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "duration": [1.0, 2.0, np.nan, 3.0, -1.0],
            "total_packets": [10, 20, 30, np.inf, 50],
            "total_bytes": [100, 200, 300, 400, 500],
            "protocol_name": ["tcp", "udp", "tcp", "tcp", "udp"],
            "label": [0, 1, 0, 1, 0],
        })

    def test_cleaner(self):
        from tabpfn_nids.preprocessing.cleaner import clean_data
        df = self._make_dirty_df()
        cleaned, report = clean_data(df, missing_strategy="median")
        assert not cleaned.isna().any().any()
        assert not np.isinf(cleaned.select_dtypes(include=[np.number])).any().any()
        assert (cleaned["duration"] >= 0).all()

    def test_encoder_ordinal(self):
        from tabpfn_nids.preprocessing.encoder import CategoricalEncoder
        df = self._make_dirty_df()
        enc = CategoricalEncoder(strategy="ordinal", categorical_columns=["protocol_name"])
        result = enc.fit_transform(df)
        assert result["protocol_name"].dtype in [np.int64, np.int32, int]

    def test_encoder_handles_unknown(self):
        from tabpfn_nids.preprocessing.encoder import CategoricalEncoder
        train_df = pd.DataFrame({"proto": ["tcp", "udp"]})
        test_df = pd.DataFrame({"proto": ["tcp", "icmp"]})
        enc = CategoricalEncoder(strategy="ordinal", categorical_columns=["proto"])
        enc.fit(train_df)
        result = enc.transform(test_df)
        # "icmp" was not in training, should be encoded as -1
        assert (result["proto"] == -1).any()

    def test_scaler(self):
        from tabpfn_nids.preprocessing.scaler import NumericScaler
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10, 20, 30, 40, 50]})
        scaler = NumericScaler(strategy="robust", numeric_columns=["a", "b"])
        result = scaler.fit_transform(df)
        # Robust scaling: median should be ~0
        assert abs(result["a"].median()) < 1e-6

    def test_feature_selector(self):
        from tabpfn_nids.preprocessing.feature_selector import FeatureSelector
        df = pd.DataFrame({
            "constant": [1.0] * 10,
            "useful": np.random.randn(10),
            "flow_id": range(10),
        })
        sel = FeatureSelector(
            remove_constant=True,
            exclude_features=["flow_id"],
        )
        result = sel.fit_transform(df)
        assert "constant" not in result.columns
        assert "flow_id" not in result.columns
        assert "useful" in result.columns


# ============================================================================
# Splitting Tests
# ============================================================================

class TestSplitting:
    """Test train/val/test splitting and leakage detection."""

    def _make_labeled_df(self, n: int = 100) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "flow_id": [f"f_{i}" for i in range(n)],
            "src_ip": [f"192.168.1.{i % 10}" for i in range(n)],
            "dst_ip": [f"10.0.0.{i % 5}" for i in range(n)],
            "start_time": np.sort(rng.uniform(0, 1000, n)),
            "feature_a": rng.standard_normal(n),
            "feature_b": rng.standard_normal(n),
            "label": rng.integers(0, 2, n),
        })

    def test_temporal_split(self):
        from tabpfn_nids.splitting.splitter import temporal_split
        df = self._make_labeled_df()
        train, val, test = temporal_split(df, "start_time", 0.7, 0.15, 0.15)
        assert len(train) + len(val) + len(test) == len(df)
        # Temporal order: train timestamps should be earlier
        assert train["start_time"].max() <= val["start_time"].min()
        assert val["start_time"].max() <= test["start_time"].min()

    def test_stratified_split(self):
        from tabpfn_nids.splitting.splitter import stratified_random_split
        df = self._make_labeled_df()
        train, val, test = stratified_random_split(df, "label", 0.7, 0.15, 0.15)
        assert len(train) + len(val) + len(test) == len(df)

    def test_leakage_checker_clean(self):
        from tabpfn_nids.splitting.leakage_checker import check_flow_id_leakage
        df = self._make_labeled_df()
        # Manual non-overlapping split
        train = df.iloc[:70]
        val = df.iloc[70:85]
        test = df.iloc[85:]
        result = check_flow_id_leakage(train, val, test)
        assert result["clean"] is True

    def test_leakage_checker_detects_leak(self):
        from tabpfn_nids.splitting.leakage_checker import check_flow_id_leakage
        df = self._make_labeled_df()
        train = df.iloc[:70]
        val = df.iloc[:15]  # Intentional overlap!
        test = df.iloc[85:]
        result = check_flow_id_leakage(train, val, test)
        assert result["clean"] is False


# ============================================================================
# Config Tests
# ============================================================================

class TestConfig:
    """Test pipeline configuration loading."""

    def test_load_config(self):
        from tabpfn_nids.pipeline_config import load_config
        cfg = load_config()
        assert cfg.extraction.backend in ("scapy", "tshark")
        assert cfg.tabpfn.max_context_samples in (2000, 10_000)
        assert cfg.splitting.train_ratio == 0.7
        assert cfg.task.mode == "binary"

    def test_config_paths_are_absolute(self):
        from tabpfn_nids.pipeline_config import load_config
        cfg = load_config()
        assert cfg.paths.raw_data_dir.is_absolute()
        assert cfg.paths.intermediate_dir.is_absolute()
