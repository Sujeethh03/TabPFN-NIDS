"""Model Registry for TabPFN NIDS models.

Discovers, registers, loads, caches, and validates TabPFN model artifacts.
Supports single-model deployments and multi-model ensemble setups.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tabpfn_nids.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = PROJECT_ROOT / "data" / "artifacts" / "models"
DEFAULT_SCHEMA_PATH = DEFAULT_MODELS_DIR / "model_feature_schema.json"


class ModelLoadError(Exception):
    """Raised when a model artifact cannot be loaded."""
    pass


class IncompatibleModelError(Exception):
    """Raised when models in an ensemble have incompatible schemas or class mappings."""
    pass


@dataclass
class ModelInfo:
    """Metadata describing a registered TabPFN model artifact."""

    model_id: str
    model_path: Path
    schema_path: Path | None = None
    feature_count: int = 67
    feature_names: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=lambda: ["Normal", "Attack"])
    task: str = "binary"
    available: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    loaded_model: Any = None

    def summary(self) -> dict[str, Any]:
        """Return a serializable dictionary summary."""
        return {
            "model_id": self.model_id,
            "model_path": str(self.model_path),
            "schema_path": str(self.schema_path) if self.schema_path else None,
            "feature_count": self.feature_count,
            "classes": self.classes,
            "task": self.task,
            "available": self.available,
        }


class ModelRegistry:
    """Manages TabPFN model artifact discovery, loading, and compatibility checks.

    Supports:
    1. Single-model layout:
       models/
           tabpfn_binary_model.pkl
           model_feature_schema.json
    2. Multi-model subdirectories:
       models/
           model_1/
               tabpfn_binary_model.pkl
               model_feature_schema.json (optional)
           model_2/
               ...
    3. Multiple model files in models directory:
       models/
           model_1.pkl
           model_2.pkl
    """

    def __init__(
        self,
        models_dir: Path | str | None = None,
        default_schema_path: Path | str | None = None,
    ) -> None:
        self.models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        self.default_schema_path = (
            Path(default_schema_path) if default_schema_path else DEFAULT_SCHEMA_PATH
        )
        self._models: dict[str, ModelInfo] = {}
        self._load_cache: dict[str, Any] = {}

    def register_model(
        self,
        model_info: ModelInfo,
        model_instance: Any = None,
    ) -> None:
        """Register a model explicitly (useful for testing or runtime injection)."""
        if model_instance is not None:
            model_info.loaded_model = model_instance
            self._load_cache[model_info.model_id] = model_instance
        self._models[model_info.model_id] = model_info
        logger.debug("Registered model '%s' (%s)", model_info.model_id, model_info.model_path)

    def discover_models(self) -> list[ModelInfo]:
        """Scan models directory and discover available TabPFN models.

        Returns:
            List of discovered ModelInfo instances.
        """
        if not self.models_dir.exists():
            logger.warning("Models directory does not exist: %s", self.models_dir)
            return list(self._models.values())

        # Load default schema if available
        default_feature_names: list[str] = []
        default_classes: list[str] = ["Normal", "Attack"]
        default_task: str = "binary"
        if self.default_schema_path.exists():
            try:
                with open(self.default_schema_path, "r", encoding="utf-8") as f:
                    schema = json.load(f)
                default_feature_names = schema.get("feature_names", [])
                default_classes = schema.get("classes", ["Normal", "Attack"])
                default_task = schema.get("task", "binary")
            except Exception as e:
                logger.warning("Failed to parse default schema %s: %s", self.default_schema_path, e)

        discovered: dict[str, ModelInfo] = {}

        # 1. Check for subdirectories (e.g. models/model_1/, models/model_2/)
        subdirs = [d for d in self.models_dir.iterdir() if d.is_dir()]
        for subdir in sorted(subdirs):
            pkl_files = list(subdir.glob("*.pkl"))
            if not pkl_files:
                continue
            model_file = pkl_files[0]
            model_id = subdir.name

            schema_file = subdir / "model_feature_schema.json"
            if not schema_file.exists():
                schema_file = subdir / "metadata.json"
            if not schema_file.exists():
                schema_file = self.default_schema_path

            feature_names = list(default_feature_names)
            classes = list(default_classes)
            task = default_task

            if schema_file.exists():
                try:
                    with open(schema_file, "r", encoding="utf-8") as f:
                        sc = json.load(f)
                    feature_names = sc.get("feature_names", feature_names)
                    classes = sc.get("classes", classes)
                    task = sc.get("task", task)
                except Exception as e:
                    logger.warning("Error reading schema %s: %s", schema_file, e)

            info = ModelInfo(
                model_id=model_id,
                model_path=model_file,
                schema_path=schema_file if schema_file.exists() else None,
                feature_count=len(feature_names) if feature_names else 67,
                feature_names=feature_names,
                classes=classes,
                task=task,
                available=True,
            )
            discovered[model_id] = info

        # 2. Check for root model files if no subdirectories with models were found
        # or alongside subdirectories
        root_pkls = [
            f for f in self.models_dir.glob("*.pkl")
            if f.is_file() and not f.name.startswith(".")
        ]

        # If subdirectories exist, we consider subdirectories as the models.
        # But if no subdirectories exist, or if we want single-model root support:
        if not discovered:
            for pkl_file in sorted(root_pkls):
                model_id = pkl_file.stem
                info = ModelInfo(
                    model_id=model_id,
                    model_path=pkl_file,
                    schema_path=self.default_schema_path if self.default_schema_path.exists() else None,
                    feature_count=len(default_feature_names) if default_feature_names else 67,
                    feature_names=default_feature_names,
                    classes=default_classes,
                    task=default_task,
                    available=True,
                )
                discovered[model_id] = info

        # Merge with any manually registered models
        for m_id, info in self._models.items():
            if m_id not in discovered:
                discovered[m_id] = info

        self._models = discovered
        logger.info("ModelRegistry discovered %d model(s): %s", len(self._models), list(self._models.keys()))
        return list(self._models.values())

    def get_model_info(self, model_id: str) -> ModelInfo:
        """Get ModelInfo for a given model ID."""
        if model_id not in self._models:
            self.discover_models()
        if model_id not in self._models:
            raise KeyError(f"Model '{model_id}' not found in registry. Available: {list(self._models.keys())}")
        return self._models[model_id]

    def list_models(self) -> list[ModelInfo]:
        """Return all known models (discovering if necessary)."""
        if not self._models:
            self.discover_models()
        return list(self._models.values())

    def get_model(self, model_id: str) -> Any:
        """Load and return the model object by model_id with caching.

        Raises:
            ModelLoadError: If loading fails or model cannot be found.
        """
        if model_id in self._load_cache:
            return self._load_cache[model_id]

        info = self.get_model_info(model_id)
        if not info.model_path.exists():
            raise ModelLoadError(
                f"Model file for '{model_id}' does not exist at {info.model_path}"
            )

        logger.info("Loading model artifact for '%s' from %s", model_id, info.model_path)
        try:
            with open(info.model_path, "rb") as f:
                model = pickle.load(f)
            self._load_cache[model_id] = model
            info.loaded_model = model
            return model
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load TabPFN model '{model_id}' from {info.model_path}: {exc}"
            ) from exc

    def load_all(self, model_ids: list[str] | None = None) -> dict[str, Any]:
        """Load and cache all specified models (or all discovered models)."""
        if model_ids is None:
            if not self._models:
                self.discover_models()
            model_ids = list(self._models.keys())

        loaded = {}
        for m_id in model_ids:
            loaded[m_id] = self.get_model(m_id)
        return loaded

    def validate_compatibility(
        self,
        models: list[ModelInfo | str] | None = None,
    ) -> bool:
        """Validate that all specified models are compatible for ensembling.

        Checks:
        1. Feature count matches
        2. Feature names and ordering match
        3. Class mapping matches (Normal=0, Attack=1)
        4. Classification task matches

        Raises:
            IncompatibleModelError: If any incompatibility is detected.
            ValueError: If no models are available to validate.
        """
        if models is None:
            models_info = self.list_models()
        else:
            models_info = [
                self.get_model_info(m) if isinstance(m, str) else m
                for m in models
            ]

        if not models_info:
            raise IncompatibleModelError("No compatible TabPFN model available in registry.")

        if len(models_info) == 1:
            return True

        ref = models_info[0]
        for m in models_info[1:]:
            # 1. Feature count check
            if m.feature_count != ref.feature_count:
                raise IncompatibleModelError(
                    f"Model '{m.model_id}' feature count ({m.feature_count}) does not "
                    f"match reference model '{ref.model_id}' feature count ({ref.feature_count})."
                )

            # 2. Feature names check
            if ref.feature_names and m.feature_names:
                if ref.feature_names != m.feature_names:
                    # Find differences
                    diffs = [
                        f"idx {i}: '{ref.feature_names[i]}' vs '{m.feature_names[i]}'"
                        for i in range(min(len(ref.feature_names), len(m.feature_names)))
                        if ref.feature_names[i] != m.feature_names[i]
                    ]
                    raise IncompatibleModelError(
                        f"Model '{m.model_id}' feature schema does not match "
                        f"reference model '{ref.model_id}'. Differences: {', '.join(diffs[:5])}"
                    )

            # 3. Class mapping check (0 = Normal, 1 = Attack)
            if ref.classes != m.classes:
                raise IncompatibleModelError(
                    f"Model '{m.model_id}' class mapping ({m.classes}) does not match "
                    f"reference model '{ref.model_id}' class mapping ({ref.classes}). "
                    f"Expected ['Normal', 'Attack']."
                )

            # 4. Task check
            if ref.task != m.task:
                raise IncompatibleModelError(
                    f"Model '{m.model_id}' task '{m.task}' does not match "
                    f"reference model '{ref.model_id}' task '{ref.task}'."
                )

        logger.info(
            "Compatibility validated successfully across %d models: %s",
            len(models_info),
            [m.model_id for m in models_info],
        )
        return True
