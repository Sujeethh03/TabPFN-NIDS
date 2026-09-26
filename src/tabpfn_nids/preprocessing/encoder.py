"""Categorical encoding for PCAP-extracted flow features.

Supports ordinal encoding (for TabPFN, which handles categoricals natively)
and one-hot encoding (for classical models). The encoder is fitted on the
training set only and applied identically to validation/test sets.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

logger = logging.getLogger(__name__)


class CategoricalEncoder:
    """Encode categorical features with train-only fitting discipline.

    Attributes:
        strategy: "ordinal" or "onehot".
        categorical_columns: Columns to encode.
        encoders_: Fitted encoders per column (after fit).
    """

    def __init__(
        self,
        strategy: str = "ordinal",
        categorical_columns: list[str] | None = None,
    ) -> None:
        self.strategy = strategy
        self.categorical_columns = categorical_columns or []
        self.encoders_: dict[str, Any] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> CategoricalEncoder:
        """Fit encoders on the training set.

        Args:
            df: Training DataFrame.

        Returns:
            self for chaining.
        """
        for col in self.categorical_columns:
            if col not in df.columns:
                logger.warning("Column '%s' not in DataFrame, skipping", col)
                continue

            values = df[col].astype(str).to_numpy().reshape(-1, 1)

            if self.strategy == "ordinal":
                enc = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )
                enc.fit(values)
            else:  # onehot
                enc = OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                )
                enc.fit(values)

            self.encoders_[col] = enc

        self._fitted = True
        logger.info(
            "Fitted %s encoder on %d categorical columns",
            self.strategy, len(self.encoders_),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted encoders.

        Args:
            df: DataFrame to transform.

        Returns:
            DataFrame with categorical columns encoded.

        Raises:
            RuntimeError: If not fitted.
        """
        if not self._fitted:
            raise RuntimeError("CategoricalEncoder must be fitted first.")

        out = df.copy()

        for col, enc in self.encoders_.items():
            if col not in out.columns:
                continue

            values = out[col].astype(str).to_numpy().reshape(-1, 1)

            if self.strategy == "ordinal":
                encoded = enc.transform(values).ravel()
                out[col] = encoded.astype(int)
            else:  # onehot
                encoded = enc.transform(values)
                feature_names = enc.get_feature_names_out([col])
                encoded_df = pd.DataFrame(
                    encoded, columns=feature_names, index=out.index,
                )
                out = out.drop(columns=[col])
                out = pd.concat([out, encoded_df], axis=1)

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on data and transform it."""
        return self.fit(df).transform(df)

    def save(self, path: Path | str) -> None:
        """Save encoder config to JSON (not the sklearn objects)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "strategy": self.strategy,
            "categorical_columns": self.categorical_columns,
            "categories": {
                col: enc.categories_[0].tolist()
                for col, enc in self.encoders_.items()
                if hasattr(enc, "categories_")
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        logger.info("Encoder config saved to %s", path)
