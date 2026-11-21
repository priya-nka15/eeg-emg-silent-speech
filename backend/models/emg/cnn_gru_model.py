"""
URF/A-SSR EMG CNN-GRU Model
PRD §26 — Frozen model set: EMG CNN-GRU hybrid.
Input: sequence-shaped (time × channels/features).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from backend.models.base import BaseModel


class EMGCNNGRUModel(BaseModel):
    """
    EMG CNN-GRU hybrid classifier.
    CNN extracts local temporal features; GRU captures sequential dynamics.
    Input: 3D array (n_samples, window_length, n_channels/features).
    PRD §26: part of the frozen EMG model set.
    """

    DEFAULT_LABEL_MAP = {"YES": 0, "NO": 1, "HELP": 2}

    def __init__(
        self,
        label_map: Optional[Dict[str, int]] = None,
        model_id: str = "emg-cnn-gru-v1",
        cnn_filters: int = 32,
        cnn_kernel_size: int = 5,
        gru_hidden_size: int = 64,
        gru_num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        n_epochs: int = 50,
        batch_size: int = 32,
        random_state: int = 42,
    ) -> None:
        super().__init__(label_map or self.DEFAULT_LABEL_MAP, model_id)
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self._model = None
        self._n_classes = len(self.label_map)
        self._input_size: Optional[int] = None

        self.hyperparams = dict(
            cnn_filters=cnn_filters,
            cnn_kernel_size=cnn_kernel_size,
            gru_hidden_size=gru_hidden_size,
            gru_num_layers=gru_num_layers,
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
            warnings.warn(
                "PyTorch not available. EMGCNNGRUModel will not be able to train.",
                ImportWarning,
            )
            return False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        X: shape (n_samples, window_length, n_features)
        y: label strings
        """
        if X.ndim != 3:
            raise ValueError(
                f"CNN-GRU expects 3D input (n_samples, seq_len, n_features), "
                f"got {X.ndim}D"
            )
        if not self._torch_available:
            raise RuntimeError("PyTorch required for CNN-GRU model.")

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._input_size = X.shape[2]
        y_int = np.array([self.label_map[label] for label in y])

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y_int, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model = _CNNGRUNet(
            input_size=self._input_size,
            cnn_filters=self.cnn_filters,
            cnn_kernel_size=self.cnn_kernel_size,
            gru_hidden_size=self.gru_hidden_size,
            gru_num_layers=self.gru_num_layers,
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
        if X.ndim != 3:
            raise ValueError(f"Expected 3D input, got {X.ndim}D")

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
                "model_type": "emg-cnn-gru",
                "model_id": self.model_id,
                "input_size": self._input_size,
                "hyperparams": self.hyperparams,
            }, f, indent=2)

    def load(self, directory: Path) -> None:
        if not self._torch_available:
            raise RuntimeError("PyTorch required.")
        import torch
        directory = Path(directory)
        with open(directory / "config.json") as f:
            cfg = json.load(f)
        self._input_size = cfg["input_size"]
        self._model = _CNNGRUNet(
            input_size=self._input_size,
            cnn_filters=self.cnn_filters,
            cnn_kernel_size=self.cnn_kernel_size,
            gru_hidden_size=self.gru_hidden_size,
            gru_num_layers=self.gru_num_layers,
            n_classes=self._n_classes,
            dropout=self.dropout,
        )
        self._model.load_state_dict(torch.load(directory / "model.pt"))
        self._model.eval()
        self.load_label_map(directory)
        self._trained = True


try:
    import torch.nn as nn

    class _CNNGRUNet(nn.Module):
        """CNN feature extractor → GRU sequential model."""

        def __init__(
            self,
            input_size: int,
            cnn_filters: int,
            cnn_kernel_size: int,
            gru_hidden_size: int,
            gru_num_layers: int,
            n_classes: int,
            dropout: float,
        ):
            super().__init__()
            # CNN operates on (batch, channels, seq_len) — note the permute
            self.cnn = nn.Sequential(
                nn.Conv1d(input_size, cnn_filters, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2),
                nn.BatchNorm1d(cnn_filters),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.gru = nn.GRU(
                input_size=cnn_filters,
                hidden_size=gru_hidden_size,
                num_layers=gru_num_layers,
                batch_first=True,
                dropout=dropout if gru_num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(gru_hidden_size, n_classes)

        def forward(self, x):
            # x: (batch, seq_len, input_size)
            x = x.permute(0, 2, 1)          # → (batch, input_size, seq_len)
            x = self.cnn(x)                  # → (batch, cnn_filters, seq_len)
            x = x.permute(0, 2, 1)          # → (batch, seq_len, cnn_filters)
            out, _ = self.gru(x)
            out = self.dropout(out[:, -1, :])
            return self.fc(out)

except ImportError:
    class _CNNGRUNet:  # type: ignore
        def __init__(self, **kwargs):
            raise ImportError("PyTorch required for CNN-GRU model.")
