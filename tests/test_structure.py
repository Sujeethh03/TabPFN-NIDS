"""Verifies the package layout from Build Plan step 0.1.

The point of step 0.1 is not that the folders exist, but that the package
imports cleanly from any working directory. These tests assert that
property directly, so a regression in packaging fails the suite rather
than surfacing later as a broken notebook.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import tabpfn_nids
from tabpfn_nids import config

SUBPACKAGES = [
    "tabpfn_nids.datasets",
    "tabpfn_nids.preprocessing",
    "tabpfn_nids.features",
    "tabpfn_nids.models",
    "tabpfn_nids.evaluation",
    "tabpfn_nids.utils",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_is_importable(name: str) -> None:
    """Every declared subpackage imports and carries a module docstring."""
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} is missing its module-level docstring"


def test_project_root_points_at_the_repository() -> None:
    """PROJECT_ROOT resolves to the checkout root, not to src/."""
    assert (config.PROJECT_ROOT / "pyproject.toml").is_file()
    assert (config.PROJECT_ROOT / "src" / "tabpfn_nids").is_dir()


def test_expected_directories_exist() -> None:
    """The directory tree required by the pipeline is present."""
    expected = [
        config.RAW_DIR,
        config.INTERIM_DIR,
        config.PROCESSED_DIR,
        config.REPORTS_DIR,
        config.FIGURES_DIR,
        config.CHECKPOINT_DIR,
        config.TABLES_DIR,
        config.CONFIGS_DIR,
        config.DOCS_DIR,
    ]
    missing = [str(p) for p in expected if not p.is_dir()]
    assert not missing, f"missing directories: {missing}"


def test_no_hardcoded_absolute_paths_in_package() -> None:
    """Guards Build Plan RULE 7 / step 11.1 against regression."""
    offenders = []
    for path in (config.PROJECT_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for marker in ("/Users/", "/home/", "C:\\\\"):
            if marker in text:
                offenders.append(f"{path.name}: {marker}")
    assert not offenders, f"hardcoded absolute paths found: {offenders}"


def test_imports_cleanly_from_another_working_directory(tmp_path: Path) -> None:
    """The whole point of step 0.1: imports do not depend on the cwd.

    A plain ``src/`` on sys.path would pass the tests above and fail here,
    which is exactly the failure mode that breaks notebooks and fresh clones.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import tabpfn_nids; print(tabpfn_nids.__version__)"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == tabpfn_nids.__version__
