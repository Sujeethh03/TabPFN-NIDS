"""Project-wide configuration: paths, seeds, and logging.

This module is the single source of truth for filesystem layout and the
global random seed. Every other module in the pipeline imports its paths
from here so that no absolute path is ever hardcoded (RULE 7) and every
random operation can be traced back to one seed (RULE 5).
"""

from __future__ import annotations

import logging
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Filesystem layout. PROJECT_ROOT is resolved relative to this file, so the
# code is indifferent to the user's home directory or checkout location.
# This module lives at <root>/src/tabpfn_nids/config.py, hence three parents.
# --------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"

NSL_KDD_DIR: Path = RAW_DIR / "nsl-kdd"
UNSW_NB15_DIR: Path = RAW_DIR / "unsw-nb15"
CIC_IDS_2018_DIR: Path = RAW_DIR / "cic-ids-2018"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
CHECKPOINT_DIR: Path = REPORTS_DIR / "checkpoints"
TABLES_DIR: Path = REPORTS_DIR / "tables"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

SEED: int = 42

# The five-class taxonomy used across every dataset in this project. NSL-KDD
# defines it natively; UNSW-NB15 and CIC-IDS-2018 are mapped onto it by
# tabpfn_nids/datasets/taxonomy.py.
CLASS_NAMES: tuple[str, ...] = ("normal", "dos", "probe", "r2l", "u2r")

# --------------------------------------------------------------------------
# TabPFN checkpoint.
#
# The installed tabpfn package defaults to model_path="auto", which resolves
# to the GATED HuggingFace repository 'Prior-Labs/tabpfn_3'. That fails on a
# clean machine with TabPFNHuggingFaceGatedRepoError unless the user has an
# HF token and has accepted the model terms.
#
# We pin the TabPFN v2 classifier instead, which lives in the UNGATED
# repository 'Prior-Labs/TabPFN-v2-clf'. This is also the scientifically
# correct choice: v2 is the model published in Hollmann et al., Nature 2025,
# which is the model this project claims to reproduce and extend. tabpfn_3 is
# a newer, different model and using it would invalidate that claim.
#
# See docs/feature_design/design-setup.md section 2.3.
# --------------------------------------------------------------------------
TABPFN_CHECKPOINT: str = "tabpfn-v2-classifier.ckpt"

# Above this many samples the tabpfn package refuses CPU inference by default
# (inference_config.MAX_CPU_SAMPLES). Mirrored here so callers can warn early.
MAX_CPU_SAMPLES: int = 1_000

# TabPFN v2 pretraining limits, read from tabpfn/inference_config.py. Exceeding
# them is permitted via ignore_pretraining_limits=True but degrades accuracy,
# which is the effect the chunked ensemble (Enhancement 1) exists to avoid.
MAX_CONTEXT_SAMPLES: int = 10_000
MAX_FEATURES: int = 500
MAX_CLASSES: int = 10


def resolve_device(prefer: str = "auto") -> str:
    """Choose the torch device to run TabPFN on.

    Apple Silicon MPS is preferred where available: it measured roughly 3.7x
    faster than CPU on this workload at equal accuracy, and the tabpfn package
    refuses CPU inference above MAX_CPU_SAMPLES rows by default.

    Args:
        prefer: "auto" to detect, or an explicit "mps" / "cpu" to force.

    Returns:
        The device string to pass to TabPFNClassifier.
    """
    if prefer != "auto":
        return prefer
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def set_seed(seed: int = SEED) -> None:
    """Seed every random number generator this project can reach.

    Args:
        seed: The integer seed to apply. Defaults to the project-wide SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:  # torch is optional for pure-pandas unit tests
        pass


def ensure_dirs() -> None:
    """Create the output directories the pipeline writes to, if absent."""
    for directory in (
        REPORTS_DIR,
        FIGURES_DIR,
        CHECKPOINT_DIR,
        TABLES_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure a consistent stdout log format for the whole pipeline.

    Args:
        level: Logging level for the root logger.

    Returns:
        The configured root logger.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger()


def _git_commit() -> str:
    """Return the current git commit hash, or 'unknown' outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _package_version(name: str) -> str:
    """Return an installed package's version string, or 'not-installed'."""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "not-installed"


@dataclass(frozen=True)
class RunEnvironment:
    """Captured provenance for one experiment run (RULE 5).

    Every field here is written into the results CSV so that a reviewer can
    tell exactly which code, library versions and hardware produced a number.
    """

    seed: int
    python_version: str
    platform: str
    machine: str
    cpu_count: int
    torch_version: str
    torch_backend: str
    tabpfn_version: str
    sklearn_version: str
    pandas_version: str
    numpy_version: str
    git_commit: str

    def as_dict(self) -> dict[str, Any]:
        """Return the environment as a flat dict for CSV serialisation."""
        return dict(self.__dict__)


def capture_environment(seed: int = SEED) -> RunEnvironment:
    """Collect hardware and library provenance for the current process.

    Args:
        seed: The seed in force for this run, recorded verbatim.

    Returns:
        A RunEnvironment snapshot.
    """
    torch_version = _package_version("torch")
    backend = "cpu"
    try:
        import torch

        if torch.backends.mps.is_available():
            backend = "mps"
    except ImportError:
        torch_version = "not-installed"

    return RunEnvironment(
        seed=seed,
        python_version=platform.python_version(),
        platform=platform.platform(),
        machine=platform.machine(),
        cpu_count=os.cpu_count() or -1,
        torch_version=torch_version,
        torch_backend=backend,
        tabpfn_version=_package_version("tabpfn"),
        sklearn_version=_package_version("scikit-learn"),
        pandas_version=_package_version("pandas"),
        numpy_version=_package_version("numpy"),
        git_commit=_git_commit(),
    )
