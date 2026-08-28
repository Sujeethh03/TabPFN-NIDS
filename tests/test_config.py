"""Tests for project configuration and the TabPFN checkpoint pin."""

from __future__ import annotations

from pathlib import Path

from tabpfn_nids import config


def test_checkpoint_is_pinned_and_not_the_gated_default() -> None:
    """Guards the gated-repository failure documented in design-setup.md 2.3.

    model_path="auto" resolves to 'Prior-Labs/tabpfn_3', which is gated and
    fails on any machine without a HuggingFace token. It is also the wrong
    model: this project reproduces TabPFN v2 (Nature 2025).
    """
    assert config.TABPFN_CHECKPOINT != "auto"
    assert "v2" in config.TABPFN_CHECKPOINT


def test_pretraining_limits_match_the_installed_library() -> None:
    """The mirrored limits must not drift from the installed tabpfn package."""
    from tabpfn.inference_config import InferenceConfig  # noqa: PLC0415

    assert config.MAX_CONTEXT_SAMPLES == InferenceConfig.MAX_NUMBER_OF_SAMPLES
    assert config.MAX_FEATURES == InferenceConfig.MAX_NUMBER_OF_FEATURES
    assert config.MAX_CLASSES == InferenceConfig.MAX_NUMBER_OF_CLASSES
    assert config.MAX_CPU_SAMPLES == InferenceConfig.MAX_CPU_SAMPLES


def test_resolve_device_honours_an_explicit_choice() -> None:
    """An explicit device is passed through untouched; auto returns a real one."""
    assert config.resolve_device("cpu") == "cpu"
    assert config.resolve_device("mps") == "mps"
    assert config.resolve_device("auto") in {"mps", "cpu"}


def test_set_seed_is_reproducible() -> None:
    """Seeding twice produces the same draw (RULE 5)."""
    import numpy as np  # noqa: PLC0415

    config.set_seed(config.SEED)
    first = np.random.rand(5)
    config.set_seed(config.SEED)
    assert np.array_equal(first, np.random.rand(5))


def test_capture_environment_records_provenance() -> None:
    """Every field required for the results CSV is populated (RULE 5)."""
    env = config.capture_environment(seed=config.SEED).as_dict()
    for field in ("seed", "python_version", "machine", "torch_version",
                  "torch_backend", "tabpfn_version", "git_commit"):
        assert env[field] not in (None, ""), f"{field} was not captured"


def test_smoke_test_script_exists_and_pins_the_checkpoint() -> None:
    """smoke_test.py must not construct TabPFNClassifier without model_path."""
    source = (config.PROJECT_ROOT / "smoke_test.py").read_text(encoding="utf-8")
    assert "TABPFN_CHECKPOINT" in source
    assert "TabPFNClassifier()" not in source
