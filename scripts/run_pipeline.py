#!/usr/bin/env python3
"""Master pipeline orchestrator: PCAP → Features → Labels → TabPFN → Evaluation.

Runs the full NIDS pipeline end-to-end, or individual stages. Each stage
produces intermediate outputs in data/intermediate/ and final outputs in
data/processed/ and results/.

Usage:
    python scripts/run_pipeline.py                 # Full pipeline
    python scripts/run_pipeline.py --stage extract  # Single stage
    python scripts/run_pipeline.py --task binary    # Binary classification
    python scripts/run_pipeline.py --task multiclass
    python scripts/run_pipeline.py --force          # Recompute everything

Stages:
    1. validate     — Validate PCAP files
    2. extract      — Extract flows from PCAPs
    3. features     — Compute ML features from flows
    4. label        — Apply ground-truth labels
    5. clean        — Clean data (NaN, Inf, impossible values)
    6. preprocess   — Encode, scale, select features
    7. split        — Train/val/test split with leakage check
    8. train        — Train TabPFN model
    9. evaluate     — Evaluate and generate reports
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

import numpy as np
import pandas as pd

from tabpfn_nids.pipeline_config import load_config, PipelineConfig
from tabpfn_nids import config as base_config

logger = logging.getLogger("pipeline")

STAGES = [
    "validate", "extract", "features", "label",
    "clean", "preprocess", "split", "train", "evaluate",
]


def stage_validate(cfg: PipelineConfig) -> dict:
    """Stage 1: Validate all PCAP files."""
    from tabpfn_nids.pcap.validator import validate_pcap_directory, generate_validation_report

    pcap_dir = cfg.paths.raw_data_dir
    output_dir = cfg.paths.intermediate_dir / "packet_metadata"

    if not pcap_dir.is_dir():
        logger.error("PCAP directory not found: %s", pcap_dir)
        logger.info(
            "Please place UNSW-NB15 PCAP files in: %s", pcap_dir,
        )
        return {"status": "skipped", "reason": "no PCAP directory"}

    results = validate_pcap_directory(
        pcap_dir, output_dir,
        backend=cfg.extraction.backend,
        tshark_path=cfg.extraction.tshark_path,
    )

    report = generate_validation_report(
        results,
        cfg.paths.intermediate_dir / "validation_report.json",
    )

    return {"status": "done", **report}


def stage_extract(cfg: PipelineConfig) -> dict:
    """Stage 2: Extract bidirectional flows from PCAPs."""
    from tabpfn_nids.flows.flow_builder import extract_flows_from_pcap

    pcap_dir = cfg.paths.raw_data_dir
    flows_dir = cfg.paths.intermediate_dir / "flows"
    flows_dir.mkdir(parents=True, exist_ok=True)

    pcap_files = sorted(
        p for p in pcap_dir.iterdir()
        if p.suffix.lower() in (".pcap", ".pcapng", ".cap")
    ) if pcap_dir.is_dir() else []

    if not pcap_files:
        logger.warning("No PCAP files found for extraction")
        return {"status": "skipped", "reason": "no PCAP files"}

    all_summaries = []
    for pcap_path in pcap_files:
        output_path = flows_dir / f"{pcap_path.stem}_flows.parquet"
        summary = extract_flows_from_pcap(
            pcap_path, output_path,
            backend=cfg.extraction.backend,
            flow_timeout=cfg.flows.flow_timeout_seconds,
            idle_timeout=cfg.flows.idle_timeout_seconds,
            batch_size=cfg.extraction.packet_batch_size,
            tshark_path=cfg.extraction.tshark_path,
        )
        all_summaries.append(summary)

    # Merge all flow files into a single combined file
    flow_files = sorted(flows_dir.glob("*_flows.parquet"))
    if flow_files:
        dfs = [pd.read_parquet(f) for f in flow_files]
        combined = pd.concat(dfs, ignore_index=True)
        combined_path = flows_dir / "all_flows.parquet"
        combined.to_parquet(combined_path, compression="snappy")
        logger.info("Merged %d flow files → %s (%d total flows)",
                     len(flow_files), combined_path, len(combined))

    return {"status": "done", "pcap_count": len(pcap_files), "summaries": all_summaries}


def stage_features(cfg: PipelineConfig) -> dict:
    """Stage 3: Compute ML features from flows."""
    from tabpfn_nids.features.feature_pipeline import build_features_from_flows

    flows_path = cfg.paths.intermediate_dir / "flows" / "all_flows.parquet"
    features_path = cfg.paths.intermediate_dir / "features" / "all_features.parquet"

    if not flows_path.is_file():
        logger.error("Flow file not found: %s", flows_path)
        return {"status": "skipped", "reason": "no flow file"}

    summary = build_features_from_flows(
        flows_path, features_path,
        groups=cfg.features.groups,
        epsilon=cfg.features.duration_epsilon,
    )

    return {"status": "done", **summary}


def stage_label(cfg: PipelineConfig) -> dict:
    """Stage 4: Apply ground-truth labels."""
    from tabpfn_nids.labeling.unsw_ground_truth import (
        label_flows, generate_labeling_report, save_labeled_flows,
    )

    features_path = cfg.paths.intermediate_dir / "features" / "all_features.parquet"
    labels_path = cfg.paths.intermediate_dir / "labels" / "labeled_flows.parquet"
    report_path = cfg.paths.intermediate_dir / "labels" / "labeling_report.json"

    if not features_path.is_file():
        return {"status": "skipped", "reason": "no features file"}

    df = pd.read_parquet(features_path)

    labeled_df = label_flows(
        df,
        gt_csv_path=cfg.paths.unsw_ground_truth_csv,
        ref_train_path=cfg.paths.unsw_training_parquet,
        ref_test_path=cfg.paths.unsw_testing_parquet,
        time_tolerance=cfg.labeling.time_tolerance_seconds,
        use_ip_port=cfg.labeling.use_ip_port_matching,
    )

    save_labeled_flows(labeled_df, labels_path)
    report = generate_labeling_report(labeled_df, report_path)

    return {"status": "done", **report}


def stage_clean(cfg: PipelineConfig) -> dict:
    """Stage 5: Clean data."""
    from tabpfn_nids.preprocessing.cleaner import (
        assess_quality, clean_data, save_quality_report,
    )

    labels_path = cfg.paths.intermediate_dir / "labels" / "labeled_flows.parquet"
    if not labels_path.is_file():
        return {"status": "skipped", "reason": "no labeled data"}

    df = pd.read_parquet(labels_path)

    # Before report
    before = assess_quality(df, "before_cleaning")
    save_quality_report(
        before,
        cfg.paths.artifacts_dir / "preprocessing" / "quality_before.json",
    )

    cleaned, cleaning_report = clean_data(
        df,
        missing_strategy=cfg.cleaning.missing_value_strategy,
        remove_impossible=cfg.cleaning.remove_impossible,
        remove_duplicates=cfg.cleaning.remove_duplicates,
        max_valid_port=cfg.cleaning.max_valid_port,
    )

    # After report
    after = assess_quality(cleaned, "after_cleaning")
    save_quality_report(
        after,
        cfg.paths.artifacts_dir / "preprocessing" / "quality_after.json",
    )

    # Save cleaned data
    output_path = cfg.paths.intermediate_dir / "labels" / "cleaned_flows.parquet"
    cleaned.to_parquet(output_path, compression="snappy")

    return {"status": "done", **cleaning_report}


def stage_preprocess(cfg: PipelineConfig) -> dict:
    """Stage 6: Encode, scale, and select features."""
    from tabpfn_nids.preprocessing.encoder import CategoricalEncoder
    from tabpfn_nids.preprocessing.scaler import NumericScaler
    from tabpfn_nids.preprocessing.feature_selector import FeatureSelector
    from tabpfn_nids.features.feature_pipeline import METADATA_COLUMNS

    cleaned_path = cfg.paths.intermediate_dir / "labels" / "cleaned_flows.parquet"
    if not cleaned_path.is_file():
        return {"status": "skipped", "reason": "no cleaned data"}

    df = pd.read_parquet(cleaned_path)

    # Identify columns
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns
                if c not in METADATA_COLUMNS and c not in ["attack_cat", "match_type"]]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in METADATA_COLUMNS and c not in ["label"]]

    # 1. Feature selection
    selector = FeatureSelector(
        remove_constant=cfg.feature_selection.remove_constant,
        variance_threshold=cfg.feature_selection.variance_threshold,
        remove_duplicates=cfg.feature_selection.remove_duplicates,
        exclude_features=cfg.feature_selection.exclude_features,
        correlation_threshold=cfg.feature_selection.correlation_report_threshold,
    )
    selector.fit(df)
    selector.save_report(cfg.paths.artifacts_dir / "feature_metadata" / "selection_report.json")

    # 2. Categorical encoding
    encoder = CategoricalEncoder(
        strategy=cfg.encoding.categorical_strategy,
        categorical_columns=cat_cols,
    )
    encoder.fit(df)
    encoder.save(cfg.paths.artifacts_dir / "preprocessing" / "encoder_config.json")
    df = encoder.transform(df)

    # 3. Numeric scaling
    scaler = NumericScaler(
        strategy=cfg.scaling.numeric_strategy,
        numeric_columns=num_cols,
    )
    scaler.fit(df)
    scaler.save(cfg.paths.artifacts_dir / "preprocessing" / "scaler_config.json")
    df = scaler.transform(df)

    # Save preprocessed data
    output_path = cfg.paths.intermediate_dir / "labels" / "preprocessed_flows.parquet"
    df.to_parquet(output_path, compression="snappy")

    return {
        "status": "done",
        "categorical_cols": len(cat_cols),
        "numeric_cols": len(num_cols),
        "feature_selection": selector.get_report(),
    }


def stage_split(cfg: PipelineConfig) -> dict:
    """Stage 7: Train/val/test split with leakage detection."""
    from tabpfn_nids.splitting.splitter import split_dataset, save_splits
    from tabpfn_nids.splitting.leakage_checker import run_all_leakage_checks

    preprocessed_path = cfg.paths.intermediate_dir / "labels" / "preprocessed_flows.parquet"
    if not preprocessed_path.is_file():
        return {"status": "skipped", "reason": "no preprocessed data"}

    df = pd.read_parquet(preprocessed_path)

    # Remove UNKNOWN labels before splitting
    unknown_count = (df.get("attack_cat", "") == "UNKNOWN").sum()
    if unknown_count > 0:
        logger.warning("Removing %d UNKNOWN-labeled flows before splitting", unknown_count)
        df = df[df["attack_cat"] != "UNKNOWN"].copy()

    # Remove label == -1
    if "label" in df.columns:
        invalid = (df["label"] == -1).sum()
        if invalid > 0:
            logger.warning("Removing %d unlabeled flows (label=-1)", invalid)
            df = df[df["label"] != -1].copy()

    train_df, val_df, test_df = split_dataset(
        df,
        strategy=cfg.splitting.strategy,
        train_ratio=cfg.splitting.train_ratio,
        val_ratio=cfg.splitting.validation_ratio,
        test_ratio=cfg.splitting.test_ratio,
        random_seed=cfg.splitting.random_seed,
    )

    # Leakage check
    if cfg.splitting.check_leakage:
        leakage_report = run_all_leakage_checks(
            train_df, val_df, test_df,
            output_path=cfg.paths.artifacts_dir / "split_metadata" / "leakage_report.json",
        )
        if not leakage_report.get("overall_clean", True):
            logger.error("LEAKAGE DETECTED — review the leakage report")

    # Save splits
    paths = save_splits(
        train_df, val_df, test_df,
        output_dir=cfg.paths.processed_dir,
        task=cfg.task.mode,
    )

    # Save split metadata
    split_meta = {
        "strategy": cfg.splitting.strategy,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_label_dist": train_df["label"].value_counts().to_dict() if "label" in train_df.columns else {},
        "val_label_dist": val_df["label"].value_counts().to_dict() if "label" in val_df.columns else {},
        "test_label_dist": test_df["label"].value_counts().to_dict() if "label" in test_df.columns else {},
        "paths": paths,
    }
    meta_path = cfg.paths.artifacts_dir / "split_metadata" / "split_info.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(split_meta, f, indent=2, default=str)

    return {"status": "done", **split_meta}


def stage_train(cfg: PipelineConfig) -> dict:
    """Stage 8: Train TabPFN model."""
    from tabpfn_nids.models.tabpfn_model import TabPFNModel
    from tabpfn_nids.features.feature_pipeline import METADATA_COLUMNS

    task = cfg.task.mode
    train_path = cfg.paths.processed_dir / task / "train.parquet"
    if not train_path.is_file():
        return {"status": "skipped", "reason": "no training data"}

    train_df = pd.read_parquet(train_path)

    # Separate features and labels
    exclude = list(set(METADATA_COLUMNS + [
        "label", "attack_cat", "match_type",
    ]))
    feature_cols = [c for c in train_df.columns if c not in exclude]
    X_train = train_df[feature_cols].values.astype(np.float64)
    y_train = train_df["label"].values.astype(int)

    model = TabPFNModel(
        task=task,
        max_context_samples=cfg.tabpfn.max_context_samples,
        device=cfg.tabpfn.device,
        n_estimators=cfg.tabpfn.n_estimators,
        random_state=cfg.reproducibility.random_seed,
        use_chunked_ensemble=cfg.tabpfn.use_chunked_ensemble,
        predict_batch_size=cfg.tabpfn.predict_batch_size,
    )

    model.fit(X_train, y_train)

    # Save model config
    config_path = cfg.paths.artifacts_dir / "preprocessing" / "model_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(model.get_config(), f, indent=2, default=str)

    # Predict on validation and test
    for split_name in ["validation", "test"]:
        split_path = cfg.paths.processed_dir / task / f"{split_name}.parquet"
        if not split_path.is_file():
            continue

        split_df = pd.read_parquet(split_path)
        X_split = split_df[feature_cols].values.astype(np.float64)
        y_proba = model.predict_proba(X_split)
        y_pred = np.argmax(y_proba, axis=1)

        # Save predictions
        pred_df = split_df.copy()
        pred_df["y_pred"] = y_pred
        for i in range(y_proba.shape[1]):
            pred_df[f"proba_class_{i}"] = y_proba[:, i]

        pred_path = cfg.paths.results_dir / f"{split_name}_predictions.parquet"
        Path(pred_path).parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_parquet(pred_path, compression="snappy")
        logger.info("Saved %s predictions → %s", split_name, pred_path)

    return {
        "status": "done",
        "model_config": model.get_config(),
    }


def stage_evaluate(cfg: PipelineConfig) -> dict:
    """Stage 9: Evaluate and generate reports."""
    from tabpfn_nids.evaluation import compute_metrics, format_metrics

    task = cfg.task.mode
    results = {}

    for split_name in ["validation", "test"]:
        pred_path = cfg.paths.results_dir / f"{split_name}_predictions.parquet"
        if not pred_path.is_file():
            continue

        df = pd.read_parquet(pred_path)
        y_true = df["label"].values.astype(int)
        y_pred = df["y_pred"].values.astype(int)

        # Get probability columns
        proba_cols = [c for c in df.columns if c.startswith("proba_class_")]
        y_proba = df[proba_cols].values if proba_cols else None

        metrics = compute_metrics(y_true, y_pred, y_proba)

        print(f"\n{format_metrics(metrics, title=f'{split_name.upper()} — {task}')}")

        # Save metrics
        metrics_path = cfg.paths.results_dir / f"{split_name}_metrics.json"
        with open(metrics_path, "w") as f:
            serialisable = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in metrics.items()
            }
            json.dump(serialisable, f, indent=2, default=str)

        results[split_name] = metrics

    return {"status": "done", "splits_evaluated": list(results.keys())}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="UNSW-NB15 PCAP → TabPFN NIDS Pipeline",
    )
    parser.add_argument(
        "--stage", choices=STAGES, default=None,
        help="Run a single stage. Default: run all stages sequentially.",
    )
    parser.add_argument(
        "--task", choices=["binary", "multiclass"], default="binary",
        help="Classification task (default: binary).",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to pipeline YAML config (default: configs/pipeline.yaml).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force recomputation of all stages.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the pipeline."""
    args = parse_args()

    # Load config
    cfg = load_config(args.config)

    # Override task mode from CLI
    cfg.task.mode = args.task

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format=cfg.logging.log_format,
    )
    base_config.set_seed(cfg.reproducibility.random_seed)

    # Pipeline stages
    stage_funcs = {
        "validate": stage_validate,
        "extract": stage_extract,
        "features": stage_features,
        "label": stage_label,
        "clean": stage_clean,
        "preprocess": stage_preprocess,
        "split": stage_split,
        "train": stage_train,
        "evaluate": stage_evaluate,
    }

    stages_to_run = [args.stage] if args.stage else STAGES
    pipeline_started = time.time()

    print("=" * 60)
    print("UNSW-NB15 PCAP → TabPFN NIDS Pipeline")
    print(f"Task: {args.task} | Stages: {', '.join(stages_to_run)}")
    print("=" * 60)

    results = {}
    for stage_name in stages_to_run:
        print(f"\n{'─' * 60}")
        print(f"  Stage: {stage_name}")
        print(f"{'─' * 60}")

        started = time.time()
        try:
            result = stage_funcs[stage_name](cfg)
            elapsed = time.time() - started
            result["elapsed_seconds"] = round(elapsed, 2)
            results[stage_name] = result
            print(f"  ✓ {stage_name}: {result.get('status', 'done')} ({elapsed:.1f}s)")
        except Exception as exc:
            elapsed = time.time() - started
            results[stage_name] = {
                "status": "error",
                "error": str(exc),
                "elapsed_seconds": round(elapsed, 2),
            }
            logger.error("Stage '%s' failed: %s", stage_name, exc, exc_info=True)
            print(f"  ✗ {stage_name}: ERROR — {exc}")

    total_elapsed = time.time() - pipeline_started

    # Save pipeline summary
    summary = {
        "task": args.task,
        "stages": results,
        "total_elapsed_seconds": round(total_elapsed, 2),
    }
    summary_path = cfg.paths.results_dir / "pipeline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {total_elapsed:.1f}s")
    print(f"Summary → {summary_path}")
    print(f"{'=' * 60}")

    # Return non-zero if any stage errored
    has_errors = any(r.get("status") == "error" for r in results.values())
    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
