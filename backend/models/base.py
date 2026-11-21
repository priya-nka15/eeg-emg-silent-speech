"""
URF/A-SSR Base Model Interface
All models expose: predicted label, probability/confidence, quality state.
PRD §35.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Prediction:
    """
    Single model prediction. PRD §35 — every model exposes
    predicted label, probability/confidence, and quality state.
    """
    label: str                          # e.g. "YES", "NO", "HELP"
    probabilities: Dict[str, float]     # {class_name: prob}
    confidence: float                   # max probability
    quality_ok: bool = True             # False if input quality was poor
    notes: str = ""

    @classmethod
    def no_decision(cls, reason: str = "") -> "Prediction":
        return cls(
            label="NO_DECISION",
            probabilities={},
            confidence=0.0,
            quality_ok=False,
            notes=reason,
        )


class BaseModel(ABC):
    """Abstract base for all URF/A-SSR models."""

    def __init__(
        self,
        label_map: Optional[Dict[str, int]] = None,
        model_id: str = "",
    ) -> None:
        self.label_map = label_map or {}
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        self.model_id = model_id
        self._trained = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability array shape (n_samples, n_classes)."""
        ...

    def predict(self, X: np.ndarray) -> List[Prediction]:
        """Return list of Prediction objects."""
        proba = self.predict_proba(X)
        predictions = []
        for prob_row in proba:
            label_idx = int(np.argmax(prob_row))
            label = self.inv_label_map.get(label_idx, str(label_idx))
            prob_dict = {
                self.inv_label_map.get(i, str(i)): float(p)
                for i, p in enumerate(prob_row)
            }
            predictions.append(Prediction(
                label=label,
                probabilities=prob_dict,
                confidence=float(np.max(prob_row)),
            ))
        return predictions

    @abstractmethod
    def save(self, directory: Path) -> None:
        ...

    @abstractmethod
    def load(self, directory: Path) -> None:
        ...

    def save_label_map(self, directory: Path) -> None:
        with open(directory / "label_map.json", "w") as f:
            json.dump(self.label_map, f, indent=2)

    def load_label_map(self, directory: Path) -> None:
        with open(directory / "label_map.json") as f:
            self.label_map = json.load(f)
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

    @property
    def is_trained(self) -> bool:
        return self._trained

    def validate_output_shape(self, proba: np.ndarray, n_classes: int) -> None:
        if proba.ndim != 2:
            raise ValueError(f"predict_proba must return 2D array, got {proba.ndim}D")
        if proba.shape[1] != n_classes:
            raise ValueError(
                f"Expected {n_classes} classes, got {proba.shape[1]}"
            )
        if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("Probabilities do not sum to 1.")
