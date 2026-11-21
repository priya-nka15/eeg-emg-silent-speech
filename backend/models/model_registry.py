"""
URF/A-SSR Model Registry
PRD §46 — Lightweight registry with EXPERIMENTAL/VALIDATED/SELECTED/RETIRED statuses.
Only SELECTED models can be used in final real-time mode.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class ModelStatus(str, Enum):
    EXPERIMENTAL = "EXPERIMENTAL"
    VALIDATED    = "VALIDATED"
    SELECTED     = "SELECTED"
    RETIRED      = "RETIRED"


@dataclass
class ModelEntry:
    model_id: str
    modality: str              # "EMG" | "EEG"
    task: str                  # e.g. "intent_classification"
    architecture: str          # e.g. "SVM" | "GRU" | "CNN-GRU" | "1D-CNN"
    dataset_version: str
    preprocessing_version: str
    experiment_id: str
    status: ModelStatus = ModelStatus.EXPERIMENTAL
    validation_metrics: dict = field(default_factory=dict)
    artifact_path: str = ""
    registered: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelEntry":
        d = dict(d)
        d["status"] = ModelStatus(d["status"])
        return cls(**d)


class ModelRegistryError(Exception):
    pass


class ModelRegistry:
    """
    Tracks all model artifacts with version compatibility checking.
    Only SELECTED models can be loaded in final mode.
    """

    def __init__(self, registry_path: Path) -> None:
        self.registry_path = Path(registry_path)
        self._entries: Dict[str, ModelEntry] = {}
        if self.registry_path.exists():
            self._load()

    def register(self, entry: ModelEntry) -> None:
        """Register a new model. Raises if model_id already exists."""
        if entry.model_id in self._entries:
            raise ModelRegistryError(
                f"Model {entry.model_id!r} already registered. "
                "Use a new model_id or retire the existing one first."
            )
        entry.registered = datetime.utcnow().isoformat()
        self._entries[entry.model_id] = entry
        self._save()

    def update_status(self, model_id: str, status: ModelStatus) -> None:
        if model_id not in self._entries:
            raise ModelRegistryError(f"Model {model_id!r} not found in registry.")
        self._entries[model_id].status = status
        self._save()

    def get(self, model_id: str) -> ModelEntry:
        if model_id not in self._entries:
            raise ModelRegistryError(f"Model {model_id!r} not found.")
        return self._entries[model_id]

    def get_selected(self, modality: str) -> Optional[ModelEntry]:
        """Return the SELECTED model for a given modality, or None."""
        selected = [
            e for e in self._entries.values()
            if e.status == ModelStatus.SELECTED and e.modality == modality
        ]
        if len(selected) > 1:
            raise ModelRegistryError(
                f"Multiple SELECTED models for {modality}. "
                "Only one model may be SELECTED per modality."
            )
        return selected[0] if selected else None

    def list_all(self) -> List[ModelEntry]:
        return list(self._entries.values())

    def can_load_in_final_mode(self, model_id: str) -> bool:
        """
        Check whether a model can be loaded in final real-time mode.
        Only SELECTED models are permitted. Version compatibility is checked.
        """
        if model_id not in self._entries:
            return False
        entry = self._entries[model_id]
        return entry.status == ModelStatus.SELECTED

    def check_version_compatibility(
        self,
        model_id: str,
        current_preprocessing_version: str,
        current_dataset_version: str,
    ) -> List[str]:
        """
        Check that a model's recorded versions match the current pipeline.
        Returns list of incompatibilities (empty = compatible).
        """
        entry = self.get(model_id)
        issues = []
        if entry.preprocessing_version != current_preprocessing_version:
            issues.append(
                f"Preprocessing version mismatch: "
                f"model was trained with '{entry.preprocessing_version}', "
                f"current is '{current_preprocessing_version}'"
            )
        if entry.dataset_version != current_dataset_version:
            issues.append(
                f"Dataset version mismatch: "
                f"model was trained on '{entry.dataset_version}', "
                f"current is '{current_dataset_version}'"
            )
        return issues

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "registry_version": "1",
            "updated": datetime.utcnow().isoformat(),
            "models": {k: v.to_dict() for k, v in self._entries.items()},
        }
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        with open(self.registry_path) as f:
            data = json.load(f)
        self._entries = {
            k: ModelEntry.from_dict(v)
            for k, v in data.get("models", {}).items()
        }
