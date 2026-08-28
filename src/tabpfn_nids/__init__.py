"""TabPFN-NIDS: scaling tabular foundation models for network intrusion detection.

The package is organised as one subpackage per pipeline stage:

- ``datasets``      loading raw NIDS datasets into tidy DataFrames
- ``preprocessing`` cleaning, encoding, splitting, cross-dataset alignment
- ``features``      domain-aware feature engineering (Enhancement 2)
- ``models``        TabPFN wrapper, stratified chunking, chunked ensemble
                    (Enhancement 1) and classical baselines
- ``evaluation``    metrics, significance testing, reporting and figures
- ``utils``         logging and checkpointing shared across the pipeline

Only lightweight names are re-exported here. Heavy dependencies (torch,
tabpfn) are imported lazily inside the modules that need them, so that
importing this package stays fast for tests and notebooks.
"""

from __future__ import annotations

from tabpfn_nids.config import (
    CLASS_NAMES,
    PROJECT_ROOT,
    SEED,
    capture_environment,
    ensure_dirs,
    set_seed,
    setup_logging,
)

__version__ = "0.1.0"

__all__ = [
    "CLASS_NAMES",
    "PROJECT_ROOT",
    "SEED",
    "__version__",
    "capture_environment",
    "ensure_dirs",
    "set_seed",
    "setup_logging",
]
