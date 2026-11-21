"""
URF/A-SSR EEG 1D-CNN Model
PRD §26 — Frozen model set: EEG 1D-CNN.
Input: windowed 2-channel temporal representation.
EEG labels are TBD until EEG pilot is complete (PRD §21, DL-006).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from backend.models.base import BaseModel
from backend.models.eeg.svm_model import _PLACEHOLDER_LABELS, _warn_unfrozen_labels


class EEGCNN1DModel(BaseModel):
    """
    EEG 1D-CNN classifier.
    Input: 3D array (n_samples, window_length, 2) for Fp1/Fp2.
    PRD §26: part of the frozen EEG model set.

    IMPORTANT: EEG labels are NOT frozen until after the EEG pilot (PRD §21,
    DL-006). Placeholder labels CONDITION_A/CONDITION_B are used only for
    structural/software testing. A model with these labels must not be used
    in final mode or for any research result. Warnings are emitted on init,
    fit, predict, and load when placeholder labels are detected.
    """

    PLACEHOLDER_LABEL_MAP: Dict[str, int] = {"CONDITION_A": 0, "CONDITION_B": 1}

    def __init__(
        self,
        label_map: Optional[Dict[str, int]] = None,
        model_id: str = "eeg-cnn1d-v1",
        cnn_filters: int = 16,
        cnn_kernel_size: int = 7,
        cnn_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        n_epochs: int = 50,
        batch_size: int = 32,
        random_state: int = 42,
    ) -> None:
        resolved = label_map if label_map is not None else self.PLACEHOLDER_LABEL_MAP
        super().__init__(resolved, model_id)
        _warn_unfrozen_labels(resolved, "init")
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_layers = cnn_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self._model = None
        self._n_classes = len(self.label_map)
        self._input_channels: Optional[int] = None

        self.hyperparams = dict(
            cnn_filters=cnn_filters,
            cnn_kernel_size=cnn_kernel_size,
            cnn_layers=cnn_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            batch_size=batch_size,
            random_state=random_state,
        )
        self._torch_available = self._check_torch()

    @staticmethod
    def _check_torch() -> bool:
        try:
            import torch
            return True
        except ImportError:
            warnings.warn("PyTorch not available for EEG 1D-CNN.", ImportWarning)
            return False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        X: shape (n_samples, window_length, 2)  [Fp1, Fp2]
        y: label strings
        """
        if X.ndim != 3 or X.shape[2] != 2:
            raise ValueError(
                f"EEG 1D-CNN expects shape (n_samples, seq_len, 2), got {X.shape}"
            )
        if not self._torch_available:
            raise RuntimeError("PyTorch required for EEG 1D-CNN.")
        _warn_unfrozen_labels(self.label_map, "fit")

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        self._input_channels = X.shape[2]
        y_int = np.array([self.label_map[label] for label in y])

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y_int, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        seq_len = X.shape[1]
        self._model = _CNN1DNet(
            input_channels=self._input_channels,
            seq_len=seq_len,
            cnn_filters=self.cnn_filters,
            cnn_kernel_size=self.cnn_kernel_size,
            cnn_layers=self.cnn_layers,
            n_classes=self._n_classes,
            dropout=self.dropout,
        )

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        self._model.train()
        for _ in range(self.n_epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                out = self._model(X_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()

        self._trained = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._trained or self._model is None:
            raise RuntimeError("Model not trained.")
        _warn_unfrozen_labels(self.label_map, "predict")

        import torch
        import torch.nn.functional as F

        self._model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            logits = self._model(X_t)
            proba = F.softmax(logits, dim=1).numpy()

        self.validate_output_shape(proba, self._n_classes)
        return proba

    def save(self, directory: Path) -> None:
        if not self._torch_available:
            raise RuntimeError("PyTorch required.")
        import torch
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), directory / "model.pt")
        self.save_label_map(directory)
        with open(directory / "config.json", "w") as f:
            json.dump({
                "model_type": "eeg-cnn1d",
                "model_id": self.model_id,
                "input_channels": self._input_channels,
                "hyperparams": self.hyperparams,
                "labels_frozen": False,
            }, f, indent=2)

    def load(self, directory: Path) -> None:
        if not self._torch_available:
            raise RuntimeError("PyTorch required.")
        import torch
        directory = Path(directory)
        with open(directory / "config.json") as f:
            cfg = json.load(f)
        self._input_channels = cfg["input_channels"]
        # seq_len must be tracked in config for reload
        seq_len = cfg.get("seq_len", 256)
        self._model = _CNN1DNet(
            input_channels=self._input_channels,
            seq_len=seq_len,
            cnn_filters=self.cnn_filters,
            cnn_kernel_size=self.cnn_kernel_size,
            cnn_layers=self.cnn_layers,
            n_classes=self._n_classes,
            dropout=self.dropout,
        )
        self._model.load_state_dict(torch.load(directory / "model.pt"))
        self._model.eval()
        self.load_label_map(directory)
        _warn_unfrozen_labels(self.label_map, "load")
        if not cfg.get("labels_frozen", True):
            warnings.warn(
                f"EEGCNN1DModel.load(): loaded model has labels_frozen=False. "
                f"This model must not be used in final mode (PRD §21).",
                UserWarning,
                stacklevel=2,
            )
        self._trained = True


try:
    import torch.nn as nn

    class _CNN1DNet(nn.Module):
        """Small 1D-CNN for 2-channel EEG."""

        def __init__(
            self,
            input_channels: int,
            seq_len: int,
            cnn_filters: int,
            cnn_kernel_size: int,
            cnn_layers: int,
            n_classes: int,
            dropout: float,
        ):
            super().__init__()
            layers = []
            in_ch = input_channels
            for _ in range(cnn_layers):
                layers += [
                    nn.Conv1d(in_ch, cnn_filters, kernel_size=cnn_kernel_size,
                              padding=cnn_kernel_size // 2),
                    nn.BatchNorm1d(cnn_filters),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.MaxPool1d(2),
                ]
                in_ch = cnn_filters

            self.cnn = nn.Sequential(*layers)
            # Compute flattened size
            out_len = seq_len // (2 ** cnn_layers)
            self.fc = nn.Sequential(
                nn.Linear(cnn_filters * max(1, out_len), 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            # x: (batch, seq_len, 2) → permute → (batch, 2, seq_len)
            x = x.permute(0, 2, 1)
            x = self.cnn(x)
            x = x.flatten(1)
            return self.fc(x)

except ImportError:
    class _CNN1DNet:  # type: ignore
        def __init__(self, **kwargs):
            raise ImportError("PyTorch required for 1D-CNN model.")
