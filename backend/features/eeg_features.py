"""
URF/A-SSR EEG Feature Extraction
PRD §23 — Candidate features: theta/alpha/beta power, relative band powers,
Fp1/Fp2 asymmetry (if experimentally justified), temporal features.

These are CANDIDATES — not guaranteed biomarkers of any cognitive/affective state.
Feature selection requires pilot validation.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal


# Candidate frequency bands
THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (8.0, 13.0)
BETA_BAND  = (13.0, 30.0)


@dataclass
class EEGFeatureConfig:
    """Controls which features are extracted."""
    feature_version: str = "TBD"    # PILOT-VALIDATED
    sampling_rate_hz: float = 4000.0

    # Band power toggles — candidates, not guaranteed biomarkers
    use_theta_power: bool = True     # CANDIDATE
    use_alpha_power: bool = True     # CANDIDATE
    use_beta_power: bool = True      # CANDIDATE
    use_relative_bands: bool = True  # CANDIDATE

    # Asymmetry — only if experimentally justified (PRD §23)
    use_fp1_fp2_asymmetry: bool = False  # CANDIDATE — requires justification

    # Temporal features
    use_temporal: bool = False           # CANDIDATE

    # Custom bands
    custom_bands: Optional[List[Tuple[float, float]]] = None  # PILOT-VALIDATED

    def validate(self) -> List[str]:
        issues = []
        if self.feature_version == "TBD":
            issues.append("feature_version is TBD")
        if self.use_fp1_fp2_asymmetry:
            warnings.warn(
                "Fp1/Fp2 asymmetry feature is enabled. "
                "Ensure this is experimentally justified (PRD §23).",
                UserWarning,
            )
        return issues


class EEGFeatureExtractor:
    """
    Extracts EEG feature vector from windowed 2-channel (Fp1, Fp2) data.

    Input: window of shape (window_length, 2)  [Fp1, Fp2]
    Output: 1D feature vector

    IMPORTANT: These features are candidates based on literature.
    They are not guaranteed to be informative for this specific system
    until experimentally validated.
    """

    def __init__(self, config: EEGFeatureConfig) -> None:
        self.config = config

    def extract_window(self, window: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from a single window.

        Args:
            window: shape (window_length, 2)  column 0=Fp1, column 1=Fp2

        Returns:
            1D feature vector
        """
        if window.ndim != 2 or window.shape[1] != 2:
            raise ValueError(
                f"Expected window shape (n_samples, 2), got {window.shape}"
            )

        cfg = self.config
        feature_groups = []

        bands = []
        if cfg.use_theta_power:
            bands.append(("theta", THETA_BAND))
        if cfg.use_alpha_power:
            bands.append(("alpha", ALPHA_BAND))
        if cfg.use_beta_power:
            bands.append(("beta", BETA_BAND))
        if cfg.custom_bands:
            for i, b in enumerate(cfg.custom_bands):
                bands.append((f"custom_{i}", b))

        # Absolute band powers
        abs_powers = {}
        for name, (lo, hi) in bands:
            power = self._band_power(window, lo, hi)  # shape (2,) [Fp1, Fp2]
            abs_powers[name] = power
            feature_groups.append(power)

        # Relative band powers
        if cfg.use_relative_bands and abs_powers:
            total_power = self._band_power(window, 0.5, cfg.sampling_rate_hz / 2 - 1)
            total_power = np.maximum(total_power, 1e-10)
            for name, power in abs_powers.items():
                feature_groups.append(power / total_power)

        # Fp1/Fp2 asymmetry (only if enabled and justified)
        if cfg.use_fp1_fp2_asymmetry:
            for name, (lo, hi) in bands:
                p = self._band_power(window, lo, hi)
                # log asymmetry: log(Fp2) - log(Fp1)
                fp1 = max(p[0], 1e-10)
                fp2 = max(p[1], 1e-10)
                asymmetry = np.log(fp2) - np.log(fp1)
                feature_groups.append(np.array([asymmetry]))

        # Temporal features
        if cfg.use_temporal:
            feature_groups.append(self._temporal(window))

        if not feature_groups:
            raise ValueError("No EEG features enabled.")

        return np.concatenate(feature_groups)

    def extract_batch(self, windows: np.ndarray) -> np.ndarray:
        """
        Extract features from batch of windows.
        Args:
            windows: shape (n_windows, window_length, 2)
        Returns:
            shape (n_windows, n_features)
        """
        return np.stack([self.extract_window(w) for w in windows])

    def _band_power(
        self,
        window: np.ndarray,
        low_hz: float,
        high_hz: float,
    ) -> np.ndarray:
        """Compute power in a frequency band for each channel."""
        fs = self.config.sampling_rate_hz
        freqs, psd = sp_signal.periodogram(window, fs=fs, axis=0)
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        if not np.any(mask):
            return np.zeros(window.shape[1])
        return np.mean(psd[mask], axis=0)

    def _temporal(self, window: np.ndarray) -> np.ndarray:
        """Statistical temporal features per channel."""
        mean = np.mean(window, axis=0)
        std = np.std(window, axis=0)
        skew = self._skewness(window)
        kurt = self._kurtosis(window)
        return np.concatenate([mean, std, skew, kurt])

    def _skewness(self, w: np.ndarray) -> np.ndarray:
        mu = np.mean(w, axis=0)
        sigma = np.std(w, axis=0)
        sigma = np.where(sigma == 0, 1.0, sigma)
        return np.mean(((w - mu) / sigma) ** 3, axis=0)

    def _kurtosis(self, w: np.ndarray) -> np.ndarray:
        mu = np.mean(w, axis=0)
        sigma = np.std(w, axis=0)
        sigma = np.where(sigma == 0, 1.0, sigma)
        return np.mean(((w - mu) / sigma) ** 4, axis=0) - 3.0
