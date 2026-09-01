"""Tests for the NSL-KDD loader (Build Plan step 1.5).

Two groups of tests:

- Tests built on a small synthetic fixture written to a tmp_path. These run
  anywhere, including on a fresh clone with no datasets downloaded, and cover
  the loader's contract and its error handling.
- Tests against the real dataset, skipped automatically when the files are
  absent, that assert the published row counts and label vocabulary.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from tabpfn_nids import config
from tabpfn_nids.data_pipeline.loader import (
    NSL_KDD_COLUMNS,
    NSL_KDD_EXPECTED_TEST_ROWS,
    NSL_KDD_EXPECTED_TRAIN_ROWS,
    NSL_KDD_FEATURE_COLUMNS,
    NSL_KDD_TEST_FILE,
    NSL_KDD_TRAIN_FILE,
    load_nsl_kdd,
)

# One real NSL-KDD record, taken verbatim from KDDTrain+.txt, used to build a
# synthetic fixture with the correct 43-field shape.
SAMPLE_ROW = (
    "0,tcp,ftp_data,SF,491,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0.00,0.00,"
    "0.00,0.00,1.00,0.00,0.00,150,25,0.17,0.03,0.17,0.00,0.00,0.00,0.05,0.00,"
    "normal,20"
)


@pytest.fixture
def fake_nsl_kdd(tmp_path: Path) -> Path:
    """Write a minimal but correctly shaped NSL-KDD directory.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        The directory containing the two synthetic split files.
    """
    (tmp_path / NSL_KDD_TRAIN_FILE).write_text("\n".join([SAMPLE_ROW] * 5) + "\n")
    (tmp_path / NSL_KDD_TEST_FILE).write_text("\n".join([SAMPLE_ROW] * 3) + "\n")
    return tmp_path


# --------------------------------------------------------------------------
# Contract, on synthetic data
# --------------------------------------------------------------------------


def test_returns_a_tuple_of_two_dataframes(fake_nsl_kdd: Path) -> None:
    """The loader returns exactly (train_df, test_df)."""
    result = load_nsl_kdd(fake_nsl_kdd)
    assert isinstance(result, tuple)
    assert len(result) == 2
    train_df, test_df = result
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)


def test_both_splits_have_43_columns(fake_nsl_kdd: Path) -> None:
    """41 features + attack label + difficulty."""
    train_df, test_df = load_nsl_kdd(fake_nsl_kdd)
    assert train_df.shape[1] == 43
    assert test_df.shape[1] == 43
    assert len(NSL_KDD_FEATURE_COLUMNS) == 41


def test_column_names_and_order_are_applied(fake_nsl_kdd: Path) -> None:
    """Names come from the .arff attribute order, not from pandas defaults."""
    train_df, test_df = load_nsl_kdd(fake_nsl_kdd)
    assert tuple(train_df.columns) == NSL_KDD_COLUMNS
    assert tuple(test_df.columns) == NSL_KDD_COLUMNS
    assert train_df.columns[0] == "duration"
    assert train_df.columns[40] == "dst_host_srv_rerror_rate"
    assert train_df.columns[41] == "attack"
    assert train_df.columns[42] == "difficulty"


def test_no_header_row_is_consumed_as_data(fake_nsl_kdd: Path) -> None:
    """The files are headerless; all written rows must survive."""
    train_df, test_df = load_nsl_kdd(fake_nsl_kdd)
    assert len(train_df) == 5
    assert len(test_df) == 3


def test_accepts_a_string_path(fake_nsl_kdd: Path) -> None:
    """data_dir may be a str as well as a Path."""
    train_df, _ = load_nsl_kdd(str(fake_nsl_kdd))
    assert len(train_df) == 5


def test_missing_directory_raises_with_a_helpful_message(tmp_path: Path) -> None:
    """A missing directory fails loudly, not with an empty DataFrame."""
    with pytest.raises(FileNotFoundError, match="directory not found"):
        load_nsl_kdd(tmp_path / "does-not-exist")


def test_missing_split_file_raises_with_download_instructions(tmp_path: Path) -> None:
    """A present directory with no files names the file and how to get it."""
    # re.escape: the '+' in "KDDTrain+.txt" is a regex quantifier otherwise.
    with pytest.raises(FileNotFoundError, match=re.escape(NSL_KDD_TRAIN_FILE)):
        load_nsl_kdd(tmp_path)


def test_wrong_column_count_raises(tmp_path: Path) -> None:
    """A file with the wrong shape is rejected rather than mis-labelled."""
    truncated = ",".join(SAMPLE_ROW.split(",")[:20])
    (tmp_path / NSL_KDD_TRAIN_FILE).write_text(truncated + "\n")
    (tmp_path / NSL_KDD_TEST_FILE).write_text(truncated + "\n")
    with pytest.raises(ValueError, match="columns, expected 43"):
        load_nsl_kdd(tmp_path)


# --------------------------------------------------------------------------
# Against the real dataset
# --------------------------------------------------------------------------

real_data = pytest.mark.skipif(
    not (config.NSL_KDD_DIR / NSL_KDD_TRAIN_FILE).is_file(),
    reason=f"NSL-KDD not downloaded to {config.NSL_KDD_DIR}",
)


@real_data
def test_real_row_counts_match_the_published_dataset() -> None:
    """Tavallaee et al. (2009): 125,973 train and 22,544 test records."""
    train_df, test_df = load_nsl_kdd()
    assert len(train_df) == NSL_KDD_EXPECTED_TRAIN_ROWS
    assert len(test_df) == NSL_KDD_EXPECTED_TEST_ROWS


@real_data
def test_real_splits_have_43_columns() -> None:
    """The shape assertion, against the actual files rather than a fixture."""
    train_df, test_df = load_nsl_kdd()
    assert train_df.shape == (NSL_KDD_EXPECTED_TRAIN_ROWS, 43)
    assert test_df.shape == (NSL_KDD_EXPECTED_TEST_ROWS, 43)


@real_data
def test_real_data_has_no_missing_values() -> None:
    """NSL-KDD is complete; any NaN would signal a parsing fault."""
    train_df, test_df = load_nsl_kdd()
    assert train_df.isna().sum().sum() == 0
    assert test_df.isna().sum().sum() == 0


@real_data
def test_difficulty_is_an_integer_in_the_published_range() -> None:
    """The difficulty column counts correct classifiers out of 21."""
    train_df, _ = load_nsl_kdd()
    assert pd.api.types.is_integer_dtype(train_df["difficulty"])
    assert train_df["difficulty"].between(0, 21).all()


@real_data
def test_attack_column_holds_names_not_the_binary_arff_class() -> None:
    """Field 42 of the .txt is the attack name, not {normal, anomaly}."""
    train_df, test_df = load_nsl_kdd()
    train_labels = set(train_df["attack"].unique())
    assert "normal" in train_labels
    assert "neptune" in train_labels
    assert "anomaly" not in train_labels
    # The test split deliberately introduces unseen attack types.
    assert set(test_df["attack"].unique()) - train_labels


@real_data
def test_categorical_columns_load_as_strings() -> None:
    """protocol_type, service and flag are nominal in the .arff."""
    train_df, _ = load_nsl_kdd()
    for column in ("protocol_type", "service", "flag"):
        # pandas 3.x infers StringDtype for text columns where pandas 2.x gave
        # object; accept either so the test is not pinned to one pandas major.
        assert pd.api.types.is_string_dtype(train_df[column])
    assert set(train_df["protocol_type"].unique()) == {"tcp", "udp", "icmp"}
