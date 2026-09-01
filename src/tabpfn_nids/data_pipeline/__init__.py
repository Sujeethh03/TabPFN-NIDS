"""Dataset loading and preprocessing.

``loader`` reads raw NIDS dataset files into pandas DataFrames with correct
column names and does nothing else. ``preprocessor`` turns those frames into
model-ready numeric arrays, fitting every transformer on the training split
alone.
"""

from __future__ import annotations

from tabpfn_nids.data_pipeline.loader import (
    NSL_KDD_CATEGORICAL_COLUMNS,
    NSL_KDD_COLUMNS,
    NSL_KDD_DIFFICULTY_COLUMN,
    NSL_KDD_FEATURE_COLUMNS,
    NSL_KDD_LABEL_COLUMN,
    load_and_preprocess_nsl_kdd,
    load_nsl_kdd,
)
from tabpfn_nids.data_pipeline.preprocessor import (
    NORMAL_LABEL,
    NUMERIC_COLUMNS,
    NSLKDDPreprocessor,
    drop_difficulty,
)

__all__ = [
    "NORMAL_LABEL",
    "NSL_KDD_CATEGORICAL_COLUMNS",
    "NSL_KDD_COLUMNS",
    "NSL_KDD_DIFFICULTY_COLUMN",
    "NSL_KDD_FEATURE_COLUMNS",
    "NSL_KDD_LABEL_COLUMN",
    "NSLKDDPreprocessor",
    "NUMERIC_COLUMNS",
    "drop_difficulty",
    "load_and_preprocess_nsl_kdd",
    "load_nsl_kdd",
]
