"""Preprocessing for NSL-KDD.

Turns the raw DataFrames produced by ``loader.py`` into the numeric matrices a
model consumes: one-hot encoded categoricals, standardised numerics, and a
binary label.

The central discipline of this module is that **every transformer is fitted on
the training split only**. The NSL-KDD test set deliberately contains service
values and attack types absent from training; fitting on the union would leak
test information into the encoders and inflate the reported scores. The
encoder is configured with ``handle_unknown="ignore"`` so unseen categories
become an all-zero indicator block rather than an error.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tabpfn_nids.data_pipeline.loader import (
    NSL_KDD_CATEGORICAL_COLUMNS,
    NSL_KDD_DIFFICULTY_COLUMN,
    NSL_KDD_FEATURE_COLUMNS,
    NSL_KDD_LABEL_COLUMN,
)
from tabpfn_nids.features.engineered import (
    ENGINEERED_CATEGORICAL_COLUMNS,
    ENGINEERED_NUMERIC_COLUMNS,
    NIDSFeatureEngineer,
)

logger = logging.getLogger(__name__)

# The single label value that counts as benign traffic. Every other value in
# the `attack` column -- neptune, smurf, warezmaster, and the unseen attack
# types that appear only in the test split -- is an attack.
NORMAL_LABEL: str = "normal"

NUMERIC_COLUMNS: tuple[str, ...] = tuple(
    column
    for column in NSL_KDD_FEATURE_COLUMNS
    if column not in NSL_KDD_CATEGORICAL_COLUMNS
)


class NSLKDDPreprocessor:
    """Encode, scale and binarise NSL-KDD into model-ready arrays.

    The transformer follows the scikit-learn fit/transform contract: ``fit``
    learns the category vocabulary and the scaler statistics from the training
    split alone, and ``transform`` applies them unchanged to any split.

    Attributes:
        scale: Whether numeric columns are standardised.
        feature_names_: Output column names, available after fitting.
        n_features_out_: Number of output columns, available after fitting.

    Example:
        >>> pre = NSLKDDPreprocessor()
        >>> X_train, y_train, X_test, y_test = pre.fit_transform(train_df, test_df)
        >>> X_train.shape[1] == X_test.shape[1]
        True
    """

    def __init__(
        self,
        scale: bool = True,
        use_engineered_features: bool = False,
        max_service_flag_categories: int | None = None,
    ) -> None:
        """Initialise the preprocessor.

        Args:
            scale: If True, standardise numeric columns with StandardScaler.
                TabPFN applies its own internal normalisation, so this is
                mainly needed for the classical baselines; it is kept
                configurable so the two can be compared on identical inputs.
            use_engineered_features: If True, apply NIDSFeatureEngineer before
                encoding (Enhancement 2). The flag is the ablation control:
                with everything else held fixed, toggling it isolates the
                contribution of the engineered features.
            max_service_flag_categories: Passed to the feature engineer to cap
                the cardinality of ``common_service_flag``. It has 336 values
                in NSL-KDD's training split, which one-hot encodes to 463 total
                features against TabPFN's 500-feature pretraining limit.
        """
        self.scale = scale
        self.use_engineered_features = use_engineered_features
        self.max_service_flag_categories = max_service_flag_categories

        self._engineer = NIDSFeatureEngineer(
            enabled=use_engineered_features,
            max_categories=max_service_flag_categories,
        )
        self._column_transformer: ColumnTransformer | None = None
        self.feature_names_: list[str] | None = None
        self.n_features_out_: int | None = None

    # -- internals ---------------------------------------------------------

    @property
    def categorical_columns(self) -> tuple[str, ...]:
        """Categorical columns to one-hot encode, engineered ones included."""
        if self.use_engineered_features:
            return (*NSL_KDD_CATEGORICAL_COLUMNS, *ENGINEERED_CATEGORICAL_COLUMNS)
        return NSL_KDD_CATEGORICAL_COLUMNS

    @property
    def numeric_columns(self) -> tuple[str, ...]:
        """Numeric columns to scale, engineered ones included."""
        if self.use_engineered_features:
            return (*NUMERIC_COLUMNS, *ENGINEERED_NUMERIC_COLUMNS)
        return NUMERIC_COLUMNS

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering if enabled, else pass the frame through.

        Args:
            frame: A raw frame from ``load_nsl_kdd``.

        Returns:
            The frame, enriched when the engineered-feature flag is set.
        """
        if not self.use_engineered_features:
            return frame
        return self._engineer.transform(frame)

    def _build(self) -> ColumnTransformer:
        """Construct the unfitted column transformer."""
        # handle_unknown="ignore" is what makes the test split safe. It is
        # genuinely load-bearing once engineering is on: 9 common_service_flag
        # combinations occur in the test split but never in training.
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float64,
        )
        numeric = StandardScaler() if self.scale else "passthrough"
        return ColumnTransformer(
            transformers=[
                ("categorical", encoder, list(self.categorical_columns)),
                ("numeric", numeric, list(self.numeric_columns)),
            ],
            remainder="drop",  # drops `difficulty` and the label column
            verbose_feature_names_out=False,
        )

    @staticmethod
    def _check_columns(frame: pd.DataFrame) -> None:
        """Raise if the frame is missing columns this preprocessor needs.

        Args:
            frame: A frame as produced by ``load_nsl_kdd``.

        Raises:
            KeyError: If any required feature or label column is absent.
        """
        required = {*NSL_KDD_FEATURE_COLUMNS, NSL_KDD_LABEL_COLUMN}
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(
                f"input frame is missing {len(missing)} required column(s): "
                f"{sorted(missing)}. Was it produced by load_nsl_kdd()?"
            )

    @staticmethod
    def _binarise(frame: pd.DataFrame) -> np.ndarray:
        """Map the attack column to 0 for normal traffic and 1 for attacks.

        Args:
            frame: A frame containing the ``attack`` column.

        Returns:
            An int64 array of 0/1 labels.
        """
        labels = frame[NSL_KDD_LABEL_COLUMN].astype(str).str.strip()
        return (labels != NORMAL_LABEL).astype(np.int64).to_numpy()

    # -- public API --------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> NSLKDDPreprocessor:
        """Learn encoders and scaler statistics from the training split only.

        Args:
            train_df: The raw training frame from ``load_nsl_kdd``.

        Returns:
            self, to allow chaining.
        """
        self._check_columns(train_df)
        # The engineer learns its category vocabulary from the training split
        # only, exactly like the encoders below it.
        if self.use_engineered_features:
            self._engineer.fit(train_df)
        prepared = self._prepare(train_df)

        self._column_transformer = self._build()
        self._column_transformer.fit(prepared)

        self.feature_names_ = list(
            self._column_transformer.get_feature_names_out()
        )
        self.n_features_out_ = len(self.feature_names_)

        n_numeric = len(self.numeric_columns)
        n_categorical = self.n_features_out_ - n_numeric
        logger.info(
            "Fitted on %d rows: %d numeric + %d one-hot = %d features "
            "(engineered features %s)",
            len(train_df),
            n_numeric,
            n_categorical,
            self.n_features_out_,
            "ON" if self.use_engineered_features else "OFF",
        )
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Apply the fitted transformers to a split.

        Args:
            frame: A raw frame from ``load_nsl_kdd``.

        Returns:
            An ``(X, y)`` tuple of float64 features and int64 binary labels.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if self._column_transformer is None:
            raise RuntimeError(
                "NSLKDDPreprocessor must be fitted before transform(). "
                "Call fit(train_df) or use fit_transform(train_df, test_df)."
            )
        self._check_columns(frame)
        prepared = self._prepare(frame)

        X = np.asarray(
            self._column_transformer.transform(prepared), dtype=np.float64
        )
        y = self._binarise(frame)

        # The pipeline should not be able to emit values TabPFN will reject:
        # its input validation rejects infinities outright.
        if not np.isfinite(X).all():
            n_bad = int((~np.isfinite(X)).sum())
            raise ValueError(
                f"preprocessing produced {n_bad} non-finite value(s). This "
                "indicates a fault in the input data rather than a normal "
                "condition -- NSL-KDD contains no missing or infinite values."
            )
        return X, y

    def fit_transform(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit on the training split, then transform both splits.

        This is the intended entry point. It makes the fit-on-train-only rule
        structural rather than a convention a caller has to remember.

        Args:
            train_df: The raw training frame.
            test_df: The raw test frame.

        Returns:
            ``(X_train, y_train, X_test, y_test)`` as numpy arrays. Both
            feature matrices have identical column counts and column meaning.
        """
        self.fit(train_df)
        X_train, y_train = self.transform(train_df)
        X_test, y_test = self.transform(test_df)

        logger.info(
            "train %s / test %s | attack rate train %.1f%%, test %.1f%%",
            X_train.shape,
            X_test.shape,
            100 * y_train.mean(),
            100 * y_test.mean(),
        )
        return X_train, y_train, X_test, y_test

    def get_feature_names(self) -> list[str]:
        """Return the output feature names.

        Returns:
            One name per output column, in column order.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if self.feature_names_ is None:
            raise RuntimeError("NSLKDDPreprocessor must be fitted first.")
        return list(self.feature_names_)


def drop_difficulty(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the frame without the difficulty column.

    The preprocessor already drops it via ``remainder="drop"``. This helper
    exists for callers that want the raw DataFrame minus the metadata column,
    for example when exploring the data in a notebook.

    Args:
        frame: A frame from ``load_nsl_kdd``.

    Returns:
        The frame without ``difficulty``; unchanged if it was already absent.
    """
    return frame.drop(columns=[NSL_KDD_DIFFICULTY_COLUMN], errors="ignore")
