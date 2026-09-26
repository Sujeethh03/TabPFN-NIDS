"""Inference package for TabPFN NIDS.

Provides dynamic, parallel inference for large PCAP captures and true ensemble
support for multiple TabPFN models.
"""

from __future__ import annotations

from tabpfn_nids.inference.inference_manager import (
    DynamicInferenceManager,
    InferenceResult,
    calculate_worker_count,
)
from tabpfn_nids.inference.model_registry import (
    IncompatibleModelError,
    ModelInfo,
    ModelLoadError,
    ModelRegistry,
)
from tabpfn_nids.inference.model_worker import (
    ChunkResult,
    InferenceChunk,
    ModelWorker,
)
from tabpfn_nids.inference.prediction_aggregator import (
    AggregatedPredictions,
    PredictionAggregator,
)

# Alias for convenience
InferenceManager = DynamicInferenceManager

__all__ = [
    "DynamicInferenceManager",
    "InferenceManager",
    "InferenceResult",
    "calculate_worker_count",
    "ModelRegistry",
    "ModelInfo",
    "ModelLoadError",
    "IncompatibleModelError",
    "ModelWorker",
    "InferenceChunk",
    "ChunkResult",
    "PredictionAggregator",
    "AggregatedPredictions",
]
