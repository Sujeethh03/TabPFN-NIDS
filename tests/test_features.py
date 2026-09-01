"""Tests for domain-aware feature engineering (Build Plan step 4.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabpfn_nids import config
from tabpfn_nids.data_pipeline.loader import (
    NSL_KDD_COLUMNS,
    NSL_KDD_TRAIN_FILE,
    load_nsl_kdd,
)
from tabpfn_nids.data_pipeline.preprocessor import (
    NUMERIC_COLUMNS,
    NSLKDDPreprocessor,
)
from tabpfn_nids.features.engineered import (
    ENGINEERED_COLUMNS,
    ENGINEERED_NUMERIC_COLUMNS,
    NIDSFeatureEngineer,
    engineer_features,
)


def make_frame(rows: list[dict] | None = None, n: int = 20) -> pd.DataFrame:
    """Build an NSL-KDD-shaped frame, optionally with explicit rows.

    Args:
        rows: Explicit partial rows; missing columns are filled with zeros.
        n: Number of random rows when ``rows`` is None.

    Returns:
        A DataFrame with the full 43-column NSL-KDD schema.
    """
    rng = np.random.default_rng(config.SEED)
    if rows is None:
        data: dict[str, object] = {
            column: rng.integers(0, 100, n).astype(float)
            for column in NUMERIC_COLUMNS
        }
        data["protocol_type"] = rng.choice(["tcp", "udp", "icmp"], n)
        data["service"] = rng.choice(["http", "ftp", "private"], n)
        data["flag"] = rng.choice(["SF", "S0", "REJ"], n)
        data["attack"] = rng.choice(["normal", "neptune"], n)
        data["difficulty"] = rng.integers(0, 22, n)
        return pd.DataFrame(data)[list(NSL_KDD_COLUMNS)]

    # Numeric columns are float, not int: pandas 3 raises rather than
    # upcasting when a float is assigned into an int64 column.
    frame = pd.DataFrame(
        {column: [0.0] * len(rows) for column in NSL_KDD_COLUMNS}
    )
    for column in ("protocol_type", "service", "flag", "attack"):
        frame[column] = "tcp" if column == "protocol_type" else "x"
    frame["difficulty"] = 0
    for index, row in enumerate(rows):
        for key, value in row.items():
            frame.loc[index, key] = value
    return frame


# --------------------------------------------------------------------------
# Toggle behaviour
# --------------------------------------------------------------------------


def test_disabled_returns_input_unchanged() -> None:
    """enabled=False must be a true no-op, which the ablation depends on."""
    frame = make_frame()
    result = NIDSFeatureEngineer(enabled=False).fit_transform(frame)
    assert result.shape == frame.shape
    assert list(result.columns) == list(frame.columns)
    pd.testing.assert_frame_equal(result, frame)


def test_enabled_adds_exactly_six_columns() -> None:
    """Output has the original columns plus the six engineered ones."""
    frame = make_frame()
    result = NIDSFeatureEngineer(enabled=True).fit_transform(frame)
    assert result.shape[1] == frame.shape[1] + 6
    assert result.shape[1] > frame.shape[1]
    assert result.shape[0] == frame.shape[0]
    for column in ENGINEERED_COLUMNS:
        assert column in result.columns


def test_original_columns_are_preserved_unmodified() -> None:
    """Engineering appends; it must never alter an input column."""
    frame = make_frame()
    result = NIDSFeatureEngineer(enabled=True).fit_transform(frame)
    pd.testing.assert_frame_equal(result[list(frame.columns)], frame)


def test_convenience_wrapper_honours_the_toggle() -> None:
    """engineer_features mirrors the class-level flag."""
    frame = make_frame()
    assert engineer_features(frame, enabled=False).shape == frame.shape
    assert engineer_features(frame, enabled=True).shape[1] == frame.shape[1] + 6


# --------------------------------------------------------------------------
# Correctness of the derivations
# --------------------------------------------------------------------------


def test_feature_formulas() -> None:
    """Each engineered value matches its documented definition."""
    frame = make_frame([
        {"src_bytes": 1000, "dst_bytes": 99, "duration": 9,
         "serror_rate": 0.5, "srv_serror_rate": 0.4},
    ])
    row = NIDSFeatureEngineer().fit_transform(frame).iloc[0]
    assert row["bytes_ratio"] == pytest.approx(1000 / 100)
    assert row["total_bytes"] == pytest.approx(1099)
    assert row["bytes_per_second"] == pytest.approx(1099 / 10)
    assert row["is_short_session"] == 0  # duration 9 >= 5
    assert row["error_rate_composite"] == pytest.approx(0.20)


def test_common_service_flag_combines_three_columns() -> None:
    """The interaction key concatenates protocol, service and flag."""
    frame = make_frame([{"protocol_type": "tcp", "service": "private", "flag": "S0"}])
    result = NIDSFeatureEngineer().fit_transform(frame)
    assert result.iloc[0]["common_service_flag"] == "tcp_private_S0"


def test_is_short_session_threshold() -> None:
    """The boundary is duration < 5, so exactly 5 is not short."""
    frame = make_frame([{"duration": d} for d in (0, 4, 5, 6, 100)])
    values = NIDSFeatureEngineer().fit_transform(frame)["is_short_session"].tolist()
    assert values == [1, 1, 0, 0, 0]


# --------------------------------------------------------------------------
# Edge cases and finiteness
# --------------------------------------------------------------------------


def test_zero_duration_and_zero_bytes_do_not_divide_by_zero() -> None:
    """The +1 denominators make the degenerate all-zero row safe."""
    frame = make_frame([{"src_bytes": 0, "dst_bytes": 0, "duration": 0}])
    row = NIDSFeatureEngineer().fit_transform(frame).iloc[0]
    assert row["bytes_ratio"] == 0.0
    assert row["total_bytes"] == 0.0
    assert row["bytes_per_second"] == 0.0
    assert row["is_short_session"] == 1


def test_zero_dst_bytes_gives_a_finite_ratio() -> None:
    """The classic Inf case: a pure upload with no response."""
    frame = make_frame([{"src_bytes": 500, "dst_bytes": 0, "duration": 0}])
    row = NIDSFeatureEngineer().fit_transform(frame).iloc[0]
    assert row["bytes_ratio"] == pytest.approx(500.0)
    assert np.isfinite(row["bytes_ratio"])


def test_extreme_nsl_kdd_magnitudes_stay_finite() -> None:
    """NSL-KDD's largest real values must not overflow."""
    frame = make_frame([
        {"src_bytes": 1_379_963_888, "dst_bytes": 0, "duration": 0},
        {"src_bytes": 0, "dst_bytes": 1_309_937_401, "duration": 42_908},
    ])
    result = NIDSFeatureEngineer().fit_transform(frame)
    for column in ENGINEERED_NUMERIC_COLUMNS:
        assert np.isfinite(result[column].to_numpy(dtype=np.float64)).all()


@pytest.mark.parametrize("n", [1, 5, 50])
def test_no_nan_or_inf_in_engineered_columns(n) -> None:
    """The core guarantee, across dataset sizes."""
    result = NIDSFeatureEngineer().fit_transform(make_frame(n=n))
    for column in ENGINEERED_NUMERIC_COLUMNS:
        values = result[column].to_numpy(dtype=np.float64)
        assert np.isfinite(values).all()
        assert not np.isnan(values).any()


def test_missing_source_column_raises() -> None:
    """A frame lacking src_bytes cannot be engineered."""
    with pytest.raises(KeyError, match="missing source column"):
        NIDSFeatureEngineer().fit_transform(make_frame().drop(columns=["src_bytes"]))


# --------------------------------------------------------------------------
# Category capping
# --------------------------------------------------------------------------


def test_max_categories_folds_rare_combinations_into_other() -> None:
    """Capping bounds the one-hot width against TabPFN's feature limit."""
    frame = make_frame(n=200)
    engineer = NIDSFeatureEngineer(max_categories=2).fit(frame)
    assert len(engineer.kept_categories_) == 2
    values = set(engineer.transform(frame)["common_service_flag"].unique())
    assert len(values) <= 3  # the two kept categories plus "other"
    assert "other" in values or values <= engineer.kept_categories_


def test_unseen_categories_at_transform_time_become_other() -> None:
    """A combination absent from training must not create a new category."""
    train = make_frame([{"protocol_type": "tcp", "service": "http", "flag": "SF"}])
    test = make_frame([{"protocol_type": "udp", "service": "dns", "flag": "REJ"}])
    engineer = NIDSFeatureEngineer(max_categories=1).fit(train)
    assert engineer.transform(test).iloc[0]["common_service_flag"] == "other"


# --------------------------------------------------------------------------
# Integration with the preprocessor
# --------------------------------------------------------------------------


def test_preprocessor_flag_changes_feature_count() -> None:
    """use_engineered_features widens the matrix; off leaves it alone."""
    train, test = make_frame(n=60), make_frame(n=30)
    off = NSLKDDPreprocessor(use_engineered_features=False)
    on = NSLKDDPreprocessor(use_engineered_features=True)
    X_off, _, Xt_off, _ = off.fit_transform(train, test)
    X_on, _, Xt_on, _ = on.fit_transform(train, test)

    assert X_on.shape[1] > X_off.shape[1]
    assert X_on.shape[1] == Xt_on.shape[1]
    assert X_off.shape[1] == Xt_off.shape[1]
    assert np.isfinite(X_on).all() and np.isfinite(Xt_on).all()


def test_preprocessor_output_stays_finite_with_engineering_on() -> None:
    """Engineered values must survive scaling without producing NaN."""
    train, test = make_frame(n=80), make_frame(n=40)
    X_train, y_train, X_test, y_test = NSLKDDPreprocessor(
        use_engineered_features=True
    ).fit_transform(train, test)
    assert np.isfinite(X_train).all()
    assert np.isfinite(X_test).all()
    assert set(np.unique(y_train)) <= {0, 1}
    assert set(np.unique(y_test)) <= {0, 1}


# --------------------------------------------------------------------------
# Against the real dataset
# --------------------------------------------------------------------------

real_data = pytest.mark.skipif(
    not (config.NSL_KDD_DIR / NSL_KDD_TRAIN_FILE).is_file(),
    reason=f"NSL-KDD not downloaded to {config.NSL_KDD_DIR}",
)


@real_data
def test_real_data_engineering_is_finite_and_widens_the_frame() -> None:
    """End-to-end on the real splits."""
    train_df, test_df = load_nsl_kdd()
    engineer = NIDSFeatureEngineer(enabled=True).fit(train_df)
    for frame in (train_df, test_df):
        enriched = engineer.transform(frame)
        assert enriched.shape[1] == frame.shape[1] + 6
        for column in ENGINEERED_NUMERIC_COLUMNS:
            assert np.isfinite(
                enriched[column].to_numpy(dtype=np.float64)
            ).all()


@real_data
def test_real_feature_count_stays_within_tabpfn_limit() -> None:
    """463 engineered features must remain under the 500-feature limit."""
    train_df, test_df = load_nsl_kdd()
    pre = NSLKDDPreprocessor(use_engineered_features=True)
    X_train, _, X_test, _ = pre.fit_transform(train_df, test_df)
    assert X_train.shape[1] == X_test.shape[1]
    assert X_train.shape[1] <= config.MAX_FEATURES, (
        f"{X_train.shape[1]} features exceeds TabPFN's limit of "
        f"{config.MAX_FEATURES}; set max_service_flag_categories."
    )
