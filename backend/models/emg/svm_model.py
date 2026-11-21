"""
URF/A-SSR EMG SVM Model
PRD §26 — Frozen model set: EMG SVM.
Input: engineered feature vector (PRD §27).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from backend.models.base import BaseModel


class EMGSVMModel(BaseModel):
    """
    EMG SVM classifier.
    Input: 1D engineered feature vector per window.
    PRD §26: part of the frozen EMG model set (SVM, GRU, CNN-GRU).
    """

    DEFAULT_LABEL_MAP = {"YES": 0, "NO": 1, "HELP": 2}

    def __init__(
        self,
        label_map: Optional[Dict[str, int]] = None,
        model_id: str = "emg-svm-v1",
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str = "scale",
        random_state: int = 42,
    ) -> None:
        super().__init__(label_map or self.DEFAULT_LABEL_MAP, model_id)
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                probability=True,
                random_state=random_state,
            )),
        ])
        self.hyperparams = dict(C=C, kernel=kernel, gamma=gamma, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the SVM on feature vectors.
        X: shape (n_samples, n_features)
        y: array of class label strings
        """
        if X.ndim != 2:
            raise ValueError(f"EMG SVM expects 2D input (n_samples, n_features), got {X.ndim}D")
        y_int = np.array([self.label_map[label] for label in y])
        self._pipeline.fit(X, y_int)
        self._trained = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability array shape (n_samples, n_classes).
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        if X.ndim != 2:
            raise ValueError(f"Expected 2D input, got {X.ndim}D")
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
                "model_type": "emg-svm",
                "model_id": self.model_id,
                "hyperparams": self.hyperparams,
            }, f, indent=2)

    def load(self, directory: Path) -> None:
        directory = Path(directory)
        with open(directory / "model.pkl", "rb") as f:
            self._pipeline = pickle.load(f)
        self.load_label_map(directory)
        self._trained = True
