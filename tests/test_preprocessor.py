"""Tests for the NSL-KDD preprocessor (Build Plan step 1.5).

Synthetic-fixture tests cover the contract and the leakage guarantees and run
on a fresh clone with no data. Tests marked ``real_data`` assert properties of
the actual dataset and skip automatically when it is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabpfn_nids import config
from tabpfn_nids.data_pipeline.loader import (
    NSL_KDD_CATEGORICAL_COLUMNS,
    NSL_KDD_COLUMNS,
    NSL_KDD_TRAIN_FILE,
    load_and_preprocess_nsl_kdd,
    load_nsl_kdd,
)
from tabpfn_nids.data_pipeline.preprocessor import (
    NUMERIC_COLUMNS,
    NSLKDDPreprocessor,
    drop_difficulty,
)

RNG = np.random.default_rng(config.SEED)


def _frame(
    n: int,
    services: list[str],
    attacks: list[str],
    protocols: list[str] | None = None,
) -> pd.DataFrame:
    """Build a synthetic NSL-KDD-shaped frame.

    Args:
        n: Number of rows.
        services: Pool of `service` values to draw from.
        attacks: Pool of `attack` label values to draw from.
        protocols: Pool of `protocol_type` values; defaults to the real three.

    Returns:
        A DataFrame with the full 43-column NSL-KDD schema.
    """
    protocols = protocols or ["tcp", "udp", "icmp"]
    data: dict[str, object] = {
        column: RNG.integers(0, 100, n).astype(float) for column in NUMERIC_COLUMNS
    }
    data["protocol_type"] = RNG.choice(protocols, n)
    data["service"] = RNG.choice(services, n)
    data["flag"] = RNG.choice(["SF", "S0", "REJ"], n)
    data["attack"] = RNG.choice(attacks, n)
    data["difficulty"] = RNG.integers(0, 22, n)
    return pd.DataFrame(data)[list(NSL_KDD_COLUMNS)]


@pytest.fixture
def splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test frames where the test split has unseen categories and attacks."""
    train = _frame(200, ["http", "ftp", "smtp"], ["normal", "neptune", "smurf"])
    test = _frame(
        80,
        ["http", "ftp", "telnet"],  # 'telnet' never appears in train
        ["normal", "neptune", "warezmaster"],  # 'warezmaster' unseen in train
    )
    return train, test


# --------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------


def test_returns_four_numpy_arrays(splits) -> None:
    """fit_transform yields X_train, y_train, X_test, y_test as ndarrays."""
    train, test = splits
    result = NSLKDDPreprocessor().fit_transform(train, test)
    assert len(result) == 4
    assert all(isinstance(a, np.ndarray) for a in result)


def test_row_counts_are_preserved(splits) -> None:
    """No rows are dropped or duplicated."""
    train, test = splits
    X_train, y_train, X_test, y_test = NSLKDDPreprocessor().fit_transform(train, test)
    assert X_train.shape[0] == len(train) == y_train.shape[0]
    assert X_test.shape[0] == len(test) == y_test.shape[0]


def test_train_and_test_have_the_same_number_of_columns(splits) -> None:
    """Test-only categories must not widen the test matrix."""
    train, test = splits
    X_train, _, X_test, _ = NSLKDDPreprocessor().fit_transform(train, test)
    assert X_train.shape[1] == X_test.shape[1]


def test_all_values_are_numeric(splits) -> None:
    """No strings survive: the output dtype is floating point."""
    train, test = splits
    X_train, _, X_test, _ = NSLKDDPreprocessor().fit_transform(train, test)
    assert np.issubdtype(X_train.dtype, np.floating)
    assert np.issubdtype(X_test.dtype, np.floating)


def test_no_nan_or_infinity_in_output(splits) -> None:
    """Non-finite values would be rejected by TabPFN's input validation."""
    train, test = splits
    X_train, _, X_test, _ = NSLKDDPreprocessor().fit_transform(train, test)
    assert np.isfinite(X_train).all()
    assert np.isfinite(X_test).all()


def test_labels_are_binary(splits) -> None:
    """Only 0 and 1 appear, and the dtype is integer."""
    train, test = splits
    _, y_train, _, y_test = NSLKDDPreprocessor().fit_transform(train, test)
    assert set(np.unique(y_train)) <= {0, 1}
    assert set(np.unique(y_test)) <= {0, 1}
    assert np.issubdtype(y_train.dtype, np.integer)


def test_only_normal_maps_to_zero() -> None:
    """Every non-'normal' label becomes 1, including unseen attack names."""
    attacks = ["normal", "neptune", "smurf", "warezmaster", "apache2", "mscan"]
    frame = _frame(60, ["http"], attacks)
    pre = NSLKDDPreprocessor().fit(frame)
    _, y = pre.transform(frame)
    expected = (frame["attack"] != "normal").astype(int).to_numpy()
    assert np.array_equal(y, expected)
    assert (y[frame["attack"].to_numpy() == "normal"] == 0).all()
    assert (y[frame["attack"].to_numpy() != "normal"] == 1).all()


def test_difficulty_column_is_dropped(splits) -> None:
    """The metadata column must not reach the feature matrix."""
    train, test = splits
    pre = NSLKDDPreprocessor()
    pre.fit_transform(train, test)
    assert "difficulty" not in pre.get_feature_names()
    assert "attack" not in pre.get_feature_names()


def test_drop_difficulty_helper(splits) -> None:
    """The standalone helper removes the column and tolerates its absence."""
    train, _ = splits
    trimmed = drop_difficulty(train)
    assert "difficulty" not in trimmed.columns
    assert "difficulty" not in drop_difficulty(trimmed).columns


# --------------------------------------------------------------------------
# Encoding, scaling and leakage
# --------------------------------------------------------------------------


def test_unknown_test_categories_encode_as_all_zeros(splits) -> None:
    """'telnet' is absent from train, so its indicator block must be zeroed."""
    train, test = splits
    pre = NSLKDDPreprocessor()
    pre.fit_transform(train, test)
    names = pre.get_feature_names()

    assert not any(name.endswith("telnet") for name in names)

    service_cols = [i for i, n in enumerate(names) if n.startswith("service_")]
    X_test, _ = pre.transform(test)
    unseen_rows = (test["service"] == "telnet").to_numpy()
    assert unseen_rows.any(), "fixture should contain unseen-service rows"
    assert (X_test[np.ix_(unseen_rows, service_cols)] == 0).all()


def test_feature_count_is_numeric_plus_train_categories(splits) -> None:
    """Width equals the numeric columns plus the training category vocabulary."""
    train, test = splits
    pre = NSLKDDPreprocessor()
    X_train, _, _, _ = pre.fit_transform(train, test)
    expected_onehot = sum(train[c].nunique() for c in NSL_KDD_CATEGORICAL_COLUMNS)
    assert X_train.shape[1] == len(NUMERIC_COLUMNS) + expected_onehot


def test_scaler_is_fitted_on_train_only(splits) -> None:
    """Training columns standardise to mean 0; test columns need not."""
    train, test = splits
    X_train, _, _, _ = NSLKDDPreprocessor(scale=True).fit_transform(train, test)
    names = NSLKDDPreprocessor().fit(train).get_feature_names()
    numeric_idx = [i for i, n in enumerate(names) if n in NUMERIC_COLUMNS]
    means = X_train[:, numeric_idx].mean(axis=0)
    assert np.allclose(means, 0, atol=1e-8)


def test_scaling_can_be_disabled(splits) -> None:
    """scale=False passes numeric columns through unchanged."""
    train, test = splits
    X_train, _, _, _ = NSLKDDPreprocessor(scale=False).fit_transform(train, test)
    pre = NSLKDDPreprocessor(scale=False).fit(train)
    idx = pre.get_feature_names().index("duration")
    assert np.allclose(np.sort(X_train[:, idx]), np.sort(train["duration"].to_numpy()))


def test_transform_before_fit_raises(splits) -> None:
    """Using an unfitted preprocessor is an error, not a silent default."""
    train, _ = splits
    with pytest.raises(RuntimeError, match="must be fitted"):
        NSLKDDPreprocessor().transform(train)


def test_missing_columns_raise(splits) -> None:
    """A frame that did not come from load_nsl_kdd is rejected."""
    train, _ = splits
    with pytest.raises(KeyError, match="missing"):
        NSLKDDPreprocessor().fit(train.drop(columns=["service"]))


def test_repeated_transform_is_deterministic(splits) -> None:
    """Transforming twice yields identical output (RULE 5)."""
    train, test = splits
    pre = NSLKDDPreprocessor().fit(train)
    first, y_first = pre.transform(test)
    second, y_second = pre.transform(test)
    assert np.array_equal(first, second)
    assert np.array_equal(y_first, y_second)


# --------------------------------------------------------------------------
# Against the real dataset
# --------------------------------------------------------------------------

real_data = pytest.mark.skipif(
    not (config.NSL_KDD_DIR / NSL_KDD_TRAIN_FILE).is_file(),
    reason=f"NSL-KDD not downloaded to {config.NSL_KDD_DIR}",
)


@real_data
def test_real_pipeline_shapes_and_purity() -> None:
    """End-to-end on the real data: shapes align and output is clean."""
    X_train, y_train, X_test, y_test = load_and_preprocess_nsl_kdd()
    assert X_train.shape[0] == 125_973 == y_train.shape[0]
    assert X_test.shape[0] == 22_544 == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    assert np.isfinite(X_train).all() and np.isfinite(X_test).all()
    assert set(np.unique(y_train)) == {0, 1}
    assert set(np.unique(y_test)) == {0, 1}


@real_data
def test_real_unseen_attack_types_all_map_to_the_attack_class() -> None:
    """NSL-KDD's test split introduces 17 attack types absent from training.

    This is the benchmark's defining property: it measures detection of
    novel attacks, not memorisation. Every one must binarise to 1.
    """
    train_df, test_df = load_nsl_kdd()
    unseen = set(test_df["attack"].unique()) - set(train_df["attack"].unique())
    assert len(unseen) == 17, f"expected 17 unseen attack types, got {len(unseen)}"
    assert "normal" not in unseen

    _, _, _, y_test = load_and_preprocess_nsl_kdd()
    is_unseen = test_df["attack"].isin(unseen).to_numpy()
    assert (y_test[is_unseen] == 1).all()


@real_data
def test_real_categorical_vocabularies_do_not_widen_the_test_matrix() -> None:
    """Test categories are a subset of train, so no unknown handling fires here.

    handle_unknown="ignore" is retained as a guard for other releases and for
    the cross-dataset work, but on this release it is never exercised: the
    difference runs the other way, with six services present only in training.
    """
    train_df, test_df = load_nsl_kdd()
    for column in NSL_KDD_CATEGORICAL_COLUMNS:
        train_values = set(train_df[column].unique())
        test_values = set(test_df[column].unique())
        assert test_values <= train_values
    train_only = set(train_df["service"].unique()) - set(test_df["service"].unique())
    assert len(train_only) == 6


@real_data
def test_real_class_balance_matches_published_counts() -> None:
    """67,343 of 125,973 training records are normal (Tavallaee et al., 2009)."""
    _, y_train, _, y_test = load_and_preprocess_nsl_kdd()
    assert int((y_train == 0).sum()) == 67_343
    assert int((y_train == 1).sum()) == 58_630
    # The test split is deliberately attack-heavy relative to train.
    assert y_test.mean() > y_train.mean()
