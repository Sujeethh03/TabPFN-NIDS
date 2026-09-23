"""Numeric scaling for ML features.

Fitted ONLY on the training set, then applied identically to validation/test.
TabPFN has internal normalisation, so "none" is often appropriate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)


class NumericScaler:
    """Scale numeric features with train-only fitting.

    Attributes:
        strategy: "standard", "robust", "minmax", or "none".
        numeric_columns: Columns to scale.
    """

    def __init__(
        self,
        strategy: str = "robust",
        numeric_columns: list[str] | None = None,
    ) -> None:
        self.strategy = strategy
        self.numeric_columns = numeric_columns or []
        self._scaler: Any = None
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> NumericScaler:
        """Fit scaler on training data only.

        Args:
            df: Training DataFrame.

        Returns:
            self for chaining.
        """
        if self.strategy == "none":
            self._fitted = True
            return self

        cols = [c for c in self.numeric_columns if c in df.columns]
        if not cols:
            logger.warning("No numeric columns to scale")
            self._fitted = True
            return self

        self.numeric_columns = cols

        if self.strategy == "standard":
            self._scaler = StandardScaler()
        elif self.strategy == "robust":
            self._scaler = RobustScaler()
        elif self.strategy == "minmax":
            self._scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling strategy: {self.strategy}")

        self._scaler.fit(df[cols].values.astype(np.float64))
        self._fitted = True

        logger.info("Fitted %s scaler on %d columns", self.strategy, len(cols))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaler.

        Args:
            df: DataFrame to transform.

        Returns:
            Scaled DataFrame.
        """
        if not self._fitted:
            raise RuntimeError("NumericScaler must be fitted first.")

        if self.strategy == "none" or self._scaler is None:
            return df

        out = df.copy()
        cols = [c for c in self.numeric_columns if c in out.columns]
        if cols:
            out[cols] = self._scaler.transform(out[cols].values.astype(np.float64))
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one call."""
        return self.fit(df).transform(df)

    def save(self, path: Path | str) -> None:
        """Save scaler config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "strategy": self.strategy,
            "numeric_columns": self.numeric_columns,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
