"""Pipeline configuration loader.

Reads configs/pipeline.yaml and provides typed access to every section.
This module is the single entry point for pipeline configuration — no other
module should read the YAML directly or hard-code configurable values.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tabpfn_nids.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"


@dataclass
class PathsConfig:
    raw_data_dir: Path
    intermediate_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    results_dir: Path
    logs_dir: Path
    unsw_ground_truth_csv: Path
    unsw_training_parquet: Path
    unsw_testing_parquet: Path


@dataclass
class ExtractionConfig:
    backend: str = "scapy"
    tshark_path: str = "tshark"
    packet_batch_size: int = 50_000
    protocols: list[str] = field(default_factory=lambda: ["tcp", "udp", "icmp"])


@dataclass
class FlowConfig:
    flow_timeout_seconds: float = 120.0
    idle_timeout_seconds: float = 60.0
    min_flow_duration: float = 0.0
    max_packets_per_flow: int = 1_000_000


@dataclass
class FeaturesConfig:
    groups: dict[str, bool] = field(default_factory=lambda: {
        "basic": True, "directional": True, "rate": True,
        "packet_length": True, "inter_arrival_time": True,
        "tcp_flags": True, "protocol": True,
    })
    duration_epsilon: float = 1e-6


@dataclass
class LabelingConfig:
    time_tolerance_seconds: float = 1.0
    use_ip_port_matching: bool = True
    unknown_label: str = "UNKNOWN"


@dataclass
class TaskConfig:
    mode: str = "binary"
    binary: dict[str, int] = field(default_factory=lambda: {"normal": 0, "attack": 1})
    multiclass: dict[str, int] = field(default_factory=lambda: {
        "Normal": 0, "Fuzzers": 1, "Analysis": 2, "Backdoors": 3,
        "DoS": 4, "Exploits": 5, "Generic": 6, "Reconnaissance": 7,
        "Shellcode": 8, "Worms": 9,
    })


@dataclass
class CleaningConfig:
    missing_value_strategy: str = "median"
    remove_impossible: bool = True
    remove_duplicates: bool = True
    max_valid_port: int = 65535


@dataclass
class FeatureSelectionConfig:
    remove_constant: bool = True
    variance_threshold: float = 0.0
    remove_duplicates: bool = True
    correlation_report_threshold: float = 0.98
    exclude_features: list[str] = field(default_factory=lambda: [
        "flow_id", "src_ip", "dst_ip", "src_port", "dst_port",
        "start_time", "end_time",
    ])


@dataclass
class EncodingConfig:
    categorical_strategy: str = "ordinal"


@dataclass
class ScalingConfig:
    numeric_strategy: str = "robust"


@dataclass
class SplittingConfig:
    strategy: str = "temporal"
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    check_leakage: bool = True


@dataclass
class BalancingConfig:
    strategy: str = "none"
    apply_to_train_only: bool = True


@dataclass
class TabPFNConfig:
    checkpoint: str = "tabpfn-v2-classifier.ckpt"
    device: str = "auto"
    max_context_samples: int = 10_000
    n_estimators: str | int = "auto"
    max_train_samples: int | None = None
    predict_batch_size: int = 1_000
    use_chunked_ensemble: bool = True
    ensemble_aggregation: str = "weighted_vote"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_to_file: bool = True
    log_format: str = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"


@dataclass
class PerformanceConfig:
    workers: int = 0
    show_progress: bool = True


@dataclass
class ReproducibilityConfig:
    random_seed: int = 42
    record_checksums: bool = True
    record_package_versions: bool = True


@dataclass
class PipelineConfig:
    """Top-level container holding every configuration section."""

    paths: PathsConfig
    extraction: ExtractionConfig
    flows: FlowConfig
    features: FeaturesConfig
    labeling: LabelingConfig
    task: TaskConfig
    cleaning: CleaningConfig
    feature_selection: FeatureSelectionConfig
    encoding: EncodingConfig
    scaling: ScalingConfig
    splitting: SplittingConfig
    balancing: BalancingConfig
    tabpfn: TabPFNConfig
    logging: LoggingConfig
    performance: PerformanceConfig
    reproducibility: ReproducibilityConfig


def _resolve_paths(raw: dict[str, str]) -> PathsConfig:
    """Convert path strings to absolute Paths relative to PROJECT_ROOT."""
    return PathsConfig(**{
        k: PROJECT_ROOT / v for k, v in raw.items()
    })


def _build_section(cls: type, data: dict[str, Any] | None):
    """Construct a dataclass from a YAML dict, using defaults for missing keys."""
    if data is None:
        return cls()
    return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def load_config(path: Path | str | None = None) -> PipelineConfig:
    """Load and parse the pipeline YAML configuration.

    Args:
        path: Path to the YAML file. Defaults to ``configs/pipeline.yaml``
            relative to the project root.

    Returns:
        A fully typed PipelineConfig.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Pipeline config not found at {config_path}. "
            "Create configs/pipeline.yaml or specify a path."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    logger.info("Loaded pipeline config from %s", config_path)

    return PipelineConfig(
        paths=_resolve_paths(raw.get("paths", {})),
        extraction=_build_section(ExtractionConfig, raw.get("extraction")),
        flows=_build_section(FlowConfig, raw.get("flows")),
        features=_build_section(FeaturesConfig, raw.get("features")),
        labeling=_build_section(LabelingConfig, raw.get("labeling")),
        task=_build_section(TaskConfig, raw.get("task")),
        cleaning=_build_section(CleaningConfig, raw.get("cleaning")),
        feature_selection=_build_section(FeatureSelectionConfig, raw.get("feature_selection")),
        encoding=_build_section(EncodingConfig, raw.get("encoding")),
        scaling=_build_section(ScalingConfig, raw.get("scaling")),
        splitting=_build_section(SplittingConfig, raw.get("splitting")),
        balancing=_build_section(BalancingConfig, raw.get("balancing")),
        tabpfn=_build_section(TabPFNConfig, raw.get("tabpfn")),
        logging=_build_section(LoggingConfig, raw.get("logging")),
        performance=_build_section(PerformanceConfig, raw.get("performance")),
        reproducibility=_build_section(ReproducibilityConfig, raw.get("reproducibility")),
    )
