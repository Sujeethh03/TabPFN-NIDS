"""Dataset loaders for the NIDS corpora used in this project.

This module is the only place that touches raw dataset files. It reads them
into pandas DataFrames with correct column names and performs no cleaning,
encoding, scaling or splitting -- those belong to the preprocessing stage.
Keeping the boundary sharp means a loader change cannot silently alter model
inputs.

Currently implemented:
    load_nsl_kdd  -- NSL-KDD (Tavallaee et al., 2009)

TODO: load_unsw_nb15 and load_cic_ids_2018 (Build Plan phases 6.1 and 7.1).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tabpfn_nids import config

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# NSL-KDD column names.
#
# NSL-KDD's .txt files ship without a header row. The 41 feature names and
# their order are taken verbatim from the @attribute declarations in
# KDDTrain+.arff, which is distributed alongside the .txt files in the same
# archive -- they are read from the dataset itself, not reproduced from
# memory. Verified against data/raw/nsl-kdd/KDDTrain+.arff.
#
# The .arff declares 42 attributes: the 41 features below plus a 'class'
# attribute. The .txt files carry 43 comma-separated fields. The two extra
# columns relative to the .arff feature list are:
#
#   field 42  the attack label, e.g. 'normal', 'neptune', 'satan'. Note this
#             is the specific attack name, NOT the .arff's binary
#             {normal, anomaly} 'class' attribute.
#   field 43  the difficulty level, an integer 0-21 giving the number of
#             learners (out of 21) that classified the record correctly.
#             It is dataset metadata, not a feature, and must never be fed
#             to a model.
# --------------------------------------------------------------------------
NSL_KDD_FEATURE_COLUMNS: tuple[str, ...] = (
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
)

NSL_KDD_LABEL_COLUMN: str = "attack"
NSL_KDD_DIFFICULTY_COLUMN: str = "difficulty"

NSL_KDD_COLUMNS: tuple[str, ...] = (
    *NSL_KDD_FEATURE_COLUMNS,
    NSL_KDD_LABEL_COLUMN,
    NSL_KDD_DIFFICULTY_COLUMN,
)

# The three nominal features, per the .arff declarations. Recorded here so the
# preprocessing stage does not have to rediscover them by dtype inspection.
NSL_KDD_CATEGORICAL_COLUMNS: tuple[str, ...] = ("protocol_type", "service", "flag")

NSL_KDD_TRAIN_FILE: str = "KDDTrain+.txt"
NSL_KDD_TEST_FILE: str = "KDDTest+.txt"

# Canonical row counts from Tavallaee et al. (2009). Used only to warn on a
# mismatch -- the loader reports what it read and never silently truncates.
NSL_KDD_EXPECTED_TRAIN_ROWS: int = 125_973
NSL_KDD_EXPECTED_TEST_ROWS: int = 22_544


def _read_split(path: Path) -> pd.DataFrame:
    """Read one headerless NSL-KDD .txt file into a DataFrame.

    Args:
        path: Path to a KDDTrain+.txt / KDDTest+.txt style file.

    Returns:
        A DataFrame with the 43 NSL-KDD columns.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not have exactly 43 columns.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"NSL-KDD file not found: {path}\n"
            "Download the dataset from https://www.kaggle.com/datasets/hassan06/nslkdd "
            f"and place the .txt files in {path.parent}"
        )

    # Read without `names` first. Passing `names` up front makes pandas pad
    # short rows with NaN, so a file of the wrong width would load silently
    # with empty trailing columns instead of failing.
    frame = pd.read_csv(path, header=None, index_col=False)

    if frame.shape[1] != len(NSL_KDD_COLUMNS):
        raise ValueError(
            f"{path.name} has {frame.shape[1]} columns, expected "
            f"{len(NSL_KDD_COLUMNS)} (41 features + label + difficulty). "
            "The file may be a different NSL-KDD variant."
        )

    frame.columns = list(NSL_KDD_COLUMNS)

    logger.info("Loaded %s: %d rows x %d columns", path.name, *frame.shape)
    return frame


def load_nsl_kdd(
    data_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the NSL-KDD training and test splits.

    NSL-KDD ships as headerless comma-separated .txt files. Column names are
    assigned from the @attribute declarations in the accompanying .arff file
    (see NSL_KDD_FEATURE_COLUMNS). The returned frames are raw: no cleaning,
    encoding or scaling is applied, and the difficulty column is retained so
    that callers can decide what to do with it. It is metadata rather than a
    feature and must be dropped before model fitting.

    The predefined train/test split is preserved as published. NSL-KDD's test
    set deliberately contains attack types absent from the training set, which
    is the point of the benchmark; re-splitting the union would destroy that
    property.

    Args:
        data_dir: Directory containing KDDTrain+.txt and KDDTest+.txt.
            Defaults to config.NSL_KDD_DIR (``<project root>/data/raw/nsl-kdd``).

    Returns:
        A ``(train_df, test_df)`` tuple. Each frame has 43 columns: the 41
        NSL-KDD features, an ``attack`` label column holding the specific
        attack name (``normal``, ``neptune``, ``satan``, ...), and an integer
        ``difficulty`` column in the range 0-21.

    Raises:
        FileNotFoundError: If either split file is missing.
        ValueError: If a file does not have exactly 43 columns.

    Example:
        >>> train_df, test_df = load_nsl_kdd()
        >>> train_df.shape
        (125973, 43)
    """
    directory = Path(data_dir) if data_dir is not None else config.NSL_KDD_DIR

    if not directory.is_dir():
        raise FileNotFoundError(
            f"NSL-KDD directory not found: {directory}\n"
            "Create it and place KDDTrain+.txt and KDDTest+.txt inside."
        )

    train_df = _read_split(directory / NSL_KDD_TRAIN_FILE)
    test_df = _read_split(directory / NSL_KDD_TEST_FILE)

    for name, frame, expected in (
        ("train", train_df, NSL_KDD_EXPECTED_TRAIN_ROWS),
        ("test", test_df, NSL_KDD_EXPECTED_TEST_ROWS),
    ):
        if len(frame) != expected:
            logger.warning(
                "NSL-KDD %s split has %d rows, expected %d. The file may be a "
                "different release than the one this project was verified on.",
                name,
                len(frame),
                expected,
            )

    return train_df, test_df


def load_and_preprocess_nsl_kdd(
    data_dir: Path | str | None = None,
    scale: bool = True,
    use_engineered_features: bool = False,
    max_service_flag_categories: int | None = None,
    return_preprocessor: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]
):
    """Load NSL-KDD and preprocess it in one call.

    Convenience wrapper over ``load_nsl_kdd`` followed by
    ``NSLKDDPreprocessor.fit_transform``. The preprocessor is fitted on the
    training split only and then applied to both splits.

    Args:
        data_dir: Directory containing the NSL-KDD .txt files. Defaults to
            ``config.NSL_KDD_DIR``.
        scale: Whether to standardise the numeric columns.
        use_engineered_features: Whether to add the Enhancement 2 domain
            features before encoding.
        max_service_flag_categories: Cap on ``common_service_flag``
            cardinality; None keeps all 336 training categories.
        return_preprocessor: If True, also return the fitted preprocessor so
            the caller can inspect feature names.

    Returns:
        ``(X_train, y_train, X_test, y_test)``, plus the fitted
        ``NSLKDDPreprocessor`` when ``return_preprocessor`` is True.

    Example:
        >>> X_train, y_train, X_test, y_test = load_and_preprocess_nsl_kdd()
    """
    # Imported here rather than at module scope: loading and preprocessing are
    # separate layers, and a top-level import would make every consumer of the
    # loader pull in scikit-learn.
    from tabpfn_nids.data_pipeline.preprocessor import NSLKDDPreprocessor

    train_df, test_df = load_nsl_kdd(data_dir)
    preprocessor = NSLKDDPreprocessor(
        scale=scale,
        use_engineered_features=use_engineered_features,
        max_service_flag_categories=max_service_flag_categories,
    )
    X_train, y_train, X_test, y_test = preprocessor.fit_transform(train_df, test_df)

    if return_preprocessor:
        return X_train, y_train, X_test, y_test, preprocessor
    return X_train, y_train, X_test, y_test
