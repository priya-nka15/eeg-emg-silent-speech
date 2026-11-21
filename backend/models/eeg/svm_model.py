"""
URF/A-SSR EEG SVM Model
PRD §26 — Frozen model set: EEG SVM.
Input: engineered EEG feature vector.
EEG labels are TBD until EEG pilot is complete (PRD §21, DL-006).
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from backend.models.base import BaseModel

# Placeholder labels — NOT frozen. Must be replaced after EEG pilot (Gate 6).
# These are used only for structural testing. A model carrying these labels
# must never be treated as a final EEG classifier.
_PLACEHOLDER_LABELS = frozenset({"CONDITION_A", "CONDITION_B"})


def _warn_unfrozen_labels(label_map: Dict[str, int], context: str) -> None:
    """Issue a clear warning when a model carries unfrozen placeholder labels."""
    if set(label_map.keys()) & _PLACEHOLDER_LABELS:
        warnings.warn(
            f"EEGSVMModel [{context}]: label map contains placeholder labels "
            f"{sorted(set(label_map.keys()) & _PLACEHOLDER_LABELS)}. "
            "EEG labels are NOT frozen — this model must not be used in final "
            "mode or for any research result (PRD §21, DL-006, Gate 6).",
            UserWarning,
            stacklevel=3,
        )


class EEGSVMModel(BaseModel):
    """
    EEG SVM classifier.
    Input: 1D engineered EEG feature vector.

    IMPORTANT: Label map is not frozen. EEG labels require pilot validation
    (PRD §21, DL-006). Placeholder labels CONDITION_A/CONDITION_B are used
    only for structural/software testing.

    A model with unfrozen placeholder labels:
      - may be trained and used in preliminary experiments
      - must NOT be used in final_select(), final pipeline config, or any
        result reported as research output
      - will emit a warning on construction, load, and predict
    """

    PLACEHOLDER_LABEL_MAP: Dict[str, int] = {"CONDITION_A": 0, "CONDITION_B": 1}

    def __init__(
        self,
        label_map: Optional[Dict[str, int]] = None,
        model_id: str = "eeg-svm-v1",
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str = "scale",
        random_state: int = 42,
    ) -> None:
        resolved = label_map if label_map is not None else self.PLACEHOLDER_LABEL_MAP
        super().__init__(resolved, model_id)
        _warn_unfrozen_labels(resolved, "init")
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                C=C, kernel=kernel, gamma=gamma,
                probability=True, random_state=random_state,
            )),
        ])
        self.hyperparams = dict(C=C, kernel=kernel, gamma=gamma, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError(f"EEG SVM expects 2D input, got {X.ndim}D")
        _warn_unfrozen_labels(self.label_map, "fit")
        y_int = np.array([self.label_map[label] for label in y])
        self._pipeline.fit(X, y_int)
        self._trained = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Model not trained.")
        _warn_unfrozen_labels(self.label_map, "predict")
        proba = self._pipeline.predict_proba(X)
        self.validate_output_shape(proba, len(self.label_map))
        return proba

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "model.pkl", "wb") as f:
            pickle.dump(self._pipeline, f)
        self.save_label_map(directory)
        with open(directory / "config.json", "w") as f:
            json.dump({
                "model_type": "eeg-svm",
                "model_id": self.model_id,
                "hyperparams": self.hyperparams,
                "labels_frozen": False,
                "note": (
                    "EEG labels are TBD until pilot completion (PRD §21). "
                    "Do not use in final pipeline."
                ),
            }, f, indent=2)

    def load(self, directory: Path) -> None:
        directory = Path(directory)
        with open(directory / "model.pkl", "rb") as f:
            self._pipeline = pickle.load(f)
        self.load_label_map(directory)
        _warn_unfrozen_labels(self.label_map, "load")
        # Check saved config for labels_frozen flag
        cfg_path = directory / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            if not cfg.get("labels_frozen", True):
                warnings.warn(
                    f"EEGSVMModel.load(): loaded model has labels_frozen=False. "
                    f"This model must not be used in final mode (PRD §21).",
                    UserWarning,
                    stacklevel=2,
                )
        self._trained = True
