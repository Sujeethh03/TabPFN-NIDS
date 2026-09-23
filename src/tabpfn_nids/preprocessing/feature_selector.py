"""Feature selection: remove constant, duplicate, and identifier features.

Fitted on training data only. Produces a report documenting every decision.
Does NOT auto-remove correlated features (only reports them) — domain
knowledge is needed to decide which to keep.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Select features by removing constants, duplicates, and identifiers.

    Attributes:
        remove_constant: Whether to remove zero-variance features.
        variance_threshold: Minimum variance to keep a feature.
        remove_duplicates: Whether to remove duplicate columns.
        exclude_features: Columns to always exclude (identifiers, etc).
        correlation_threshold: Report pairs above this (don't auto-remove).
    """

    def __init__(
        self,
        remove_constant: bool = True,
        variance_threshold: float = 0.0,
        remove_duplicates: bool = True,
        exclude_features: list[str] | None = None,
        correlation_threshold: float = 0.98,
    ) -> None:
        self.remove_constant = remove_constant
        self.variance_threshold = variance_threshold
        self.remove_duplicates = remove_duplicates
        self.exclude_features = exclude_features or []
        self.correlation_threshold = correlation_threshold

        self._selected_features: list[str] | None = None
        self._removed_features: dict[str, str] = {}
        self._correlation_report: list[dict[str, Any]] = []
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> FeatureSelector:
        """Learn which features to keep from the training set.

        Args:
            df: Training DataFrame.

        Returns:
            self for chaining.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = df.columns.tolist()
        keep = set(all_cols)

        # 1. Exclude identifiers
        for col in self.exclude_features:
            if col in keep:
                keep.discard(col)
                self._removed_features[col] = "identifier/metadata"

        # 2. Remove constants
        if self.remove_constant:
            for col in numeric_cols:
                if col in keep and df[col].var() <= self.variance_threshold:
                    keep.discard(col)
                    self._removed_features[col] = f"constant (var={df[col].var():.6f})"

        # 3. Remove duplicate columns
        if self.remove_duplicates:
            seen_hashes: dict[int, str] = {}
            for col in sorted(keep):
                if col not in df.columns:
                    continue
                col_hash = hash(df[col].values.tobytes()) if df[col].dtype != object else hash(tuple(df[col]))
                if col_hash in seen_hashes:
                    keep.discard(col)
                    self._removed_features[col] = f"duplicate of {seen_hashes[col_hash]}"
                else:
                    seen_hashes[col_hash] = col

        # 4. Correlation report (informational only)
        kept_numeric = [c for c in numeric_cols if c in keep]
        if len(kept_numeric) > 1:
            corr = df[kept_numeric].corr().abs()
            for i, col_a in enumerate(kept_numeric):
                for col_b in kept_numeric[i + 1:]:
                    r = corr.loc[col_a, col_b]
                    if r >= self.correlation_threshold:
                        self._correlation_report.append({
                            "feature_a": col_a,
                            "feature_b": col_b,
                            "correlation": round(float(r), 4),
                        })

        self._selected_features = sorted(keep)
        self._fitted = True

        logger.info(
            "Feature selection: %d → %d features (%d removed, %d correlated pairs reported)",
            len(all_cols), len(self._selected_features),
            len(self._removed_features), len(self._correlation_report),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection.

        Args:
            df: DataFrame to filter.

        Returns:
            DataFrame with only selected features.
        """
        if not self._fitted or self._selected_features is None:
            raise RuntimeError("FeatureSelector must be fitted first.")

        cols = [c for c in self._selected_features if c in df.columns]
        return df[cols].copy()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform."""
        return self.fit(df).transform(df)

    def get_report(self) -> dict[str, Any]:
        """Return the feature selection report."""
        return {
            "selected_features": self._selected_features,
            "removed_features": self._removed_features,
            "highly_correlated_pairs": self._correlation_report,
            "n_selected": len(self._selected_features or []),
            "n_removed": len(self._removed_features),
        }

    def save_report(self, path: Path | str) -> None:
        """Save feature selection report to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_report(), f, indent=2)
        logger.info("Feature selection report → %s", path)
