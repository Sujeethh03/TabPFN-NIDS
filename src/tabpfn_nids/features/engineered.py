"""Domain-aware feature engineering for NSL-KDD (Enhancement 2).

Vanilla TabPFN treats every input column as an unrelated variable. Network
flow records are not unrelated: bytes, packets and duration compose into
rates and ratios that a security analyst reads directly, and protocol,
service and flag interact to describe a connection's shape.

This module adds six such features. Five are numeric derivations, one is a
categorical interaction:

``bytes_ratio``            src_bytes / (dst_bytes + 1). Upload-to-download
                           asymmetry. Data exfiltration skews high; a normal
                           HTTP fetch skews low.
``total_bytes``            src_bytes + dst_bytes. Overall flow volume.
``bytes_per_second``       total_bytes / (duration + 1). Throughput; volumetric
                           floods sit at the extreme.
``is_short_session``       1 when duration < 5s. Scans and probes are short.
``error_rate_composite``   serror_rate * srv_serror_rate. High only when both
                           host-level and service-level SYN errors coincide,
                           which is the signature of a SYN flood rather than
                           of one misbehaving service.
``common_service_flag``    protocol_type + service + flag as one categorical.
                           Captures interactions a per-column encoding cannot,
                           e.g. tcp/private/S0 (a half-open scan) versus
                           tcp/http/SF (an ordinary web request).

Every denominator has +1 added, so no division by zero is possible and the
output is finite by construction. The result is asserted finite before it is
returned rather than trusted, because TabPFN rejects infinities at input
validation and a silent Inf would surface much later and far less clearly.

Feature-count warning: ``common_service_flag`` has 336 distinct values in the
NSL-KDD training split. One-hot encoded alongside the existing columns that
totals 463 features, against TabPFN's pretraining limit of 500. Use
``max_categories`` to fold rare combinations into an "other" bucket if the
budget is needed elsewhere; the 100 most frequent combinations already cover
97.5% of training rows.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ENGINEERED_NUMERIC_COLUMNS: tuple[str, ...] = (
    "bytes_ratio",
    "total_bytes",
    "bytes_per_second",
    "is_short_session",
    "error_rate_composite",
)

ENGINEERED_CATEGORICAL_COLUMNS: tuple[str, ...] = ("common_service_flag",)

ENGINEERED_COLUMNS: tuple[str, ...] = (
    *ENGINEERED_NUMERIC_COLUMNS,
    *ENGINEERED_CATEGORICAL_COLUMNS,
)

REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "src_bytes",
    "dst_bytes",
    "duration",
    "serror_rate",
    "srv_serror_rate",
    "protocol_type",
    "service",
    "flag",
)

SHORT_SESSION_SECONDS: float = 5.0
RARE_CATEGORY_LABEL: str = "other"


class NIDSFeatureEngineer:
    """Add domain-aware derived features to NSL-KDD flow records.

    The transformer follows the scikit-learn fit/transform contract. ``fit``
    is only meaningful when ``max_categories`` is set, in which case it learns
    which ``common_service_flag`` values are frequent enough to keep; every
    other value is mapped to "other" at transform time.

    Attributes:
        enabled: When False, ``transform`` returns its input unchanged.
        short_session_threshold: Duration below which a session is "short".
        max_categories: Optional cap on common_service_flag cardinality.
        kept_categories_: The retained category vocabulary, after fitting.

    Example:
        >>> engineer = NIDSFeatureEngineer(enabled=True)
        >>> enriched = engineer.fit_transform(train_df)
        >>> enriched.shape[1] - train_df.shape[1]
        6
    """

    def __init__(
        self,
        enabled: bool = True,
        short_session_threshold: float = SHORT_SESSION_SECONDS,
        max_categories: int | None = None,
    ) -> None:
        """Initialise the feature engineer.

        Args:
            enabled: Master toggle. When False the transformer is a no-op,
                which is what makes the ablation in
                scripts/run_feature_ablation.py a controlled comparison.
            short_session_threshold: Seconds below which ``is_short_session``
                is 1.
            max_categories: If set, keep only this many of the most frequent
                ``common_service_flag`` values and map the rest to "other".
        """
        self.enabled = enabled
        self.short_session_threshold = short_session_threshold
        self.max_categories = max_categories
        self.kept_categories_: set[str] | None = None

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _check_source_columns(frame: pd.DataFrame) -> None:
        """Raise if the columns the derivations need are absent.

        Args:
            frame: The input frame.

        Raises:
            KeyError: If any required source column is missing.
        """
        missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in frame.columns]
        if missing:
            raise KeyError(
                f"cannot engineer features: missing source column(s) {missing}. "
                "Was the frame produced by load_nsl_kdd()?"
            )

    @staticmethod
    def _service_flag_combo(frame: pd.DataFrame) -> pd.Series:
        """Build the protocol/service/flag interaction key.

        Args:
            frame: The input frame.

        Returns:
            A string Series of combined categorical values.
        """
        return (
            frame["protocol_type"].astype(str)
            + "_"
            + frame["service"].astype(str)
            + "_"
            + frame["flag"].astype(str)
        )

    # -- public API --------------------------------------------------------

    def fit(self, frame: pd.DataFrame) -> NIDSFeatureEngineer:
        """Learn the category vocabulary from the training split.

        Args:
            frame: The raw training frame.

        Returns:
            self, to allow chaining.
        """
        if not self.enabled:
            return self

        self._check_source_columns(frame)
        combos = self._service_flag_combo(frame)

        if self.max_categories is None:
            self.kept_categories_ = set(combos.unique())
        else:
            counts = combos.value_counts()
            self.kept_categories_ = set(counts.head(self.max_categories).index)
            coverage = counts.head(self.max_categories).sum() / len(frame)
            logger.info(
                "Keeping %d of %d common_service_flag categories (%.2f%% of rows)",
                len(self.kept_categories_),
                len(counts),
                100 * coverage,
            )

        logger.info(
            "Feature engineer fitted: %d common_service_flag categories",
            len(self.kept_categories_),
        )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the frame with engineered features appended.

        Args:
            frame: A raw NSL-KDD frame.

        Returns:
            A new DataFrame with the original columns plus the six engineered
            ones. When ``enabled`` is False, a copy of the input is returned
            unchanged.

        Raises:
            KeyError: If a required source column is missing.
            ValueError: If any engineered value is non-finite.
        """
        if not self.enabled:
            return frame

        self._check_source_columns(frame)
        out = frame.copy()

        src = out["src_bytes"].astype(np.float64)
        dst = out["dst_bytes"].astype(np.float64)
        duration = out["duration"].astype(np.float64)

        # Every denominator is +1, so division by zero cannot occur even when
        # dst_bytes or duration is 0, which is the common case in NSL-KDD.
        out["bytes_ratio"] = src / (dst + 1.0)
        out["total_bytes"] = src + dst
        out["bytes_per_second"] = (src + dst) / (duration + 1.0)
        out["is_short_session"] = (
            duration < self.short_session_threshold
        ).astype(np.int64)
        out["error_rate_composite"] = (
            out["serror_rate"].astype(np.float64)
            * out["srv_serror_rate"].astype(np.float64)
        )

        combos = self._service_flag_combo(out)
        if self.kept_categories_ is not None:
            combos = combos.where(
                combos.isin(self.kept_categories_), RARE_CATEGORY_LABEL
            )
        out["common_service_flag"] = combos

        self._assert_finite(out)
        return out

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fit on a frame and transform it in one call.

        Args:
            frame: The training frame.

        Returns:
            The enriched frame.
        """
        return self.fit(frame).transform(frame)

    @staticmethod
    def _assert_finite(frame: pd.DataFrame) -> None:
        """Verify no engineered numeric column holds NaN or Inf.

        Args:
            frame: The enriched frame.

        Raises:
            ValueError: If a non-finite value is present.
        """
        for column in ENGINEERED_NUMERIC_COLUMNS:
            values = frame[column].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                n_bad = int((~np.isfinite(values)).sum())
                raise ValueError(
                    f"engineered column {column!r} contains {n_bad} non-finite "
                    "value(s). This should be impossible given the +1 "
                    "denominators; check the input data for overflow."
                )

    def get_feature_names(self) -> list[str]:
        """Return the names of the columns this transformer adds.

        Returns:
            The engineered column names, or an empty list when disabled.
        """
        return list(ENGINEERED_COLUMNS) if self.enabled else []


def engineer_features(
    frame: pd.DataFrame, enabled: bool = True
) -> pd.DataFrame:
    """Convenience wrapper for one-off use on a single frame.

    Args:
        frame: A raw NSL-KDD frame.
        enabled: Whether to add the engineered features.

    Returns:
        The enriched frame, or the input unchanged when disabled.
    """
    return NIDSFeatureEngineer(enabled=enabled).fit_transform(frame)
