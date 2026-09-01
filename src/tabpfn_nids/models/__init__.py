"""TabPFN inference, stratified chunking and the chunked ensemble.

Enhancement 1 lives here: TabPFN v2 caps its in-context training set at
10,000 samples, so larger datasets are split into stratified chunks whose
per-chunk predictions are aggregated by weighted voting.
"""

from __future__ import annotations

from tabpfn_nids.models.chunked_ensemble import (
    AGGREGATION_STRATEGIES,
    ChunkedTabPFNEnsemble,
)
from tabpfn_nids.models.chunker import (
    describe_chunks,
    make_chunks,
    n_chunks_for,
    random_chunk,
    stratified_chunk,
)
from tabpfn_nids.models.tabpfn_wrapper import TabPFNWrapper

__all__ = [
    "AGGREGATION_STRATEGIES",
    "ChunkedTabPFNEnsemble",
    "TabPFNWrapper",
    "describe_chunks",
    "make_chunks",
    "n_chunks_for",
    "random_chunk",
    "stratified_chunk",
]
