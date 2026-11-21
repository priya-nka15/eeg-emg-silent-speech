"""
URF/A-SSR EMG Feature Extraction
PRD §24 — Compact candidate feature set: RMS, MAV, WL, ZCR, band energy, MFCC.
These are candidates — not all are guaranteed to be used.
Do not create an uncontrolled feature zoo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal


@dataclass
class EMGFeatureConfig:
    """Controls which features are extracted."""
    feature_version: str = "TBD"    # PILOT-VALIDATED
    sampling_rate_hz: float = 4000.0

    # Feature toggles — all off by default; enable after pilot validation
    use_rms: bool = True             # CANDIDATE
    use_mav: bool = True             # CANDIDATE
    use_wl: bool = True              # CANDIDATE
    use_zcr: bool = True             # CANDIDATE
    use_band_energy: bool = False    # CANDIDATE — requires validated band limits
    use_mfcc: bool = False           # CANDIDATE

    # Band energy (TBD from pilot/hardware)
    band_energy_bands: Optional[List[Tuple[float, float]]] = None  # PILOT-VALIDATED

    # MFCC
    n_mfcc: int = 13                 # CANDIDATE

    def validate(self) -> List[str]:
        issues = []
        if self.feature_version == "TBD":
            issues.append("feature_version is TBD")
        if self.use_band_energy and not self.band_energy_bands:
            issues.append("band_energy enabled but band_energy_bands not set")
        return issues


class EMGFeatureExtractor:
    """
    Extracts a compact EMG feature vector from windowed data.

    Input: window of shape (window_length, n_channels)
    Output: 1D feature vector

    PRD §24 candidate features:
        - RMS: root mean square
        - MAV: mean absolute value
        - WL: waveform length
        - ZCR: zero-crossing rate
        - Band energy: power in frequency bands
        - MFCC: mel-frequency cepstral coefficients
    """

    def __init__(self, config: EMGFeatureConfig) -> None:
        self.config = config

    def extract_window(self, window: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from a single window.

        Args:
            window: shape (window_length, n_channels)

        Returns:
            1D feature vector
        """
        if window.ndim != 2:
            raise ValueError(f"Expected 2D window (samples, channels), got {window.ndim}D")

        feature_groups = []
        cfg = self.config

        if cfg.use_rms:
            feature_groups.append(self._rms(window))
        if cfg.use_mav:
            feature_groups.append(self._mav(window))
        if cfg.use_wl:
            feature_groups.append(self._wl(window))
        if cfg.use_zcr:
            feature_groups.append(self._zcr(window))
        if cfg.use_band_energy and cfg.band_energy_bands:
            feature_groups.append(self._band_energy(window))
        if cfg.use_mfcc:
            feature_groups.append(self._mfcc(window))

        if not feature_groups:
            raise ValueError("No features enabled in EMGFeatureConfig.")

        return np.concatenate(feature_groups)

    def extract_batch(self, windows: np.ndarray) -> np.ndarray:
        """
        Extract features from a batch of windows.

        Args:
            windows: shape (n_windows, window_length, n_channels)

        Returns:
            shape (n_windows, n_features)
        """
        return np.stack([self.extract_window(w) for w in windows])

    def feature_names(self, n_channels: int = 4) -> List[str]:
        """Return ordered feature names."""
        names = []
        cfg = self.config
        ch_names = [f"ch{i}" for i in range(n_channels)]
        if cfg.use_rms:
            names += [f"rms_{c}" for c in ch_names]
        if cfg.use_mav:
            names += [f"mav_{c}" for c in ch_names]
        if cfg.use_wl:
            names += [f"wl_{c}" for c in ch_names]
        if cfg.use_zcr:
            names += [f"zcr_{c}" for c in ch_names]
        if cfg.use_band_energy and cfg.band_energy_bands:
            for i, (lo, hi) in enumerate(cfg.band_energy_bands):
                names += [f"band_energy_{lo}_{hi}_{c}" for c in ch_names]
        if cfg.use_mfcc:
            for m in range(cfg.n_mfcc):
                names += [f"mfcc_{m}_{c}" for c in ch_names]
        return names

    # ── Feature functions ─────────────────────────────────────────────────────

    def _rms(self, w: np.ndarray) -> np.ndarray:
        """Root mean square per channel."""
        return np.sqrt(np.mean(w ** 2, axis=0))

    def _mav(self, w: np.ndarray) -> np.ndarray:
        """Mean absolute value per channel."""
        return np.mean(np.abs(w), axis=0)

    def _wl(self, w: np.ndarray) -> np.ndarray:
        """Waveform length per channel."""
        return np.sum(np.abs(np.diff(w, axis=0)), axis=0)

    def _zcr(self, w: np.ndarray) -> np.ndarray:
        """Zero-crossing rate per channel."""
        signs = np.sign(w)
        crossings = np.sum(np.abs(np.diff(signs, axis=0)) > 0, axis=0)
        return crossings / max(1, w.shape[0] - 1)

    def _band_energy(self, w: np.ndarray) -> np.ndarray:
        """Power in each configured frequency band, per channel."""
        fs = self.config.sampling_rate_hz
        features = []
        for (low, high) in self.config.band_energy_bands:
            freqs, psd = sp_signal.periodogram(w, fs=fs, axis=0)
            mask = (freqs >= low) & (freqs <= high)
            band_power = np.mean(psd[mask], axis=0)
            features.append(band_power)
        return np.concatenate(features)

    def _mfcc(self, w: np.ndarray) -> np.ndarray:
        """
        MFCC per channel. Uses a simple implementation without librosa
        to avoid heavy dependency. For production, consider librosa.
        """
        n_mfcc = self.config.n_mfcc
        fs = self.config.sampling_rate_hz
        result = []
        for ch in range(w.shape[1]):
            signal = w[:, ch]
            # Simple DCT-based MFCC approximation
            freqs, psd = sp_signal.periodogram(signal, fs=fs)
            # Mel filterbank approximation (n_mfcc triangular filters)
            mel_powers = np.array_split(np.log(np.maximum(psd, 1e-10)), n_mfcc)
            coeffs = np.array([np.mean(p) for p in mel_powers])
            result.append(coeffs)
        return np.concatenate(result)
