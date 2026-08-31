"""Domain-aware feature engineering (Enhancement 2).

Engineered features are added behind a single boolean flag so that the
with/without comparison is a controlled experiment on identical splits.
"""

from __future__ import annotations

from tabpfn_nids.features.engineered import (
    ENGINEERED_CATEGORICAL_COLUMNS,
    ENGINEERED_COLUMNS,
    ENGINEERED_NUMERIC_COLUMNS,
    NIDSFeatureEngineer,
    engineer_features,
)

__all__ = [
    "ENGINEERED_CATEGORICAL_COLUMNS",
    "ENGINEERED_COLUMNS",
    "ENGINEERED_NUMERIC_COLUMNS",
    "NIDSFeatureEngineer",
    "engineer_features",
]
