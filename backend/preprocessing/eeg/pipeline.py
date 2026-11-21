"""
URF/A-SSR EEG Preprocessing Pipeline
PRD §23 — EEG preprocessing pipeline.

ICA is NOT mandated for a 2-channel system (PRD §22).
Filter parameters must NOT be invented.
Fp1/Fp2 are susceptible to ocular and facial EMG artifact.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal


@dataclass
class EEGPreprocessingConfig:
    """
    All parameters configurable. TBD fields must be pilot-validated.
    """
    preprocessing_version: str = "TBD"        # PILOT-VALIDATED

    # Hardware-already-applied
    hardware_bandpass_applied: bool = False    # HARDWARE-TBD
    hardware_notch_applied: bool = False       # HARDWARE-TBD

    # Software bandpass
    sw_bandpass_enabled: bool = False          # PILOT-VALIDATED
    sw_bandpass_low_hz: Optional[float] = None   # PILOT-VALIDATED
    sw_bandpass_high_hz: Optional[float] = None  # PILOT-VALIDATED
    sw_bandpass_order: int = 4
    sw_bandpass_filter_type: str = "butter"

    # Software notch
    sw_notch_enabled: bool = False             # PILOT-VALIDATED
    sw_notch_freq_hz: Optional[float] = None   # PILOT-VALIDATED
    sw_notch_q: float = 30.0

    # ICA — OPTIONAL, NOT mandatory for 2-channel EEG (PRD §22)
    ica_enabled: bool = False

    # Normalization
    normalization: Optional[str] = None        # PILOT-VALIDATED

    # Sampling
    sampling_rate_hz: float = 4000.0

    # ADC conversion (TBD from hardware)
    voltage_conversion_factor: Optional[float] = None  # HARDWARE-TBD

    def validate(self) -> List[str]:
        issues = []
        if self.preprocessing_version == "TBD":
            issues.append("preprocessing_version is TBD")
        if self.sw_bandpass_enabled and (
            self.sw_bandpass_low_hz is None or self.sw_bandpass_high_hz is None
        ):
            issues.append("sw_bandpass enabled but frequencies not set")
        if self.sw_notch_enabled and self.sw_notch_freq_hz is None:
            issues.append("sw_notch enabled but frequency not set")
        if self.ica_enabled:
            warnings.warn(
                "ICA is not mandatory for 2-channel EEG (PRD §22). "
                "Ensure this is justified.",
                UserWarning,
            )
        return issues


@dataclass
class EEGPreprocessingResult:
    data: np.ndarray                # shape (n_samples, 2)
    channel_names: List[str]
    preprocessing_version: str
    config: EEGPreprocessingConfig
    source_id: str = ""
    provenance: dict = field(default_factory=dict)

    def is_finite(self) -> bool:
        return bool(np.all(np.isfinite(self.data)))


class EEGPreprocessingPipeline:
    """
    EEG preprocessing pipeline for Fp1/Fp2.

    Pipeline (PRD §23):
        1. Quality checks
        2. Validated filtering
        3. Artifact detection/rejection
        4. Normalization
    """

    EEG_CHANNELS = ["EEG_Fp1", "EEG_Fp2"]

    def __init__(self, config: EEGPreprocessingConfig) -> None:
        self.config = config
        issues = config.validate()
        if issues:
            warnings.warn(
                f"EEGPreprocessingConfig issues: {issues}",
                UserWarning,
                stacklevel=2,
            )

    def process(
        self,
        raw: np.ndarray,
        source_id: str = "",
        normalization_stats: Optional[Dict] = None,
    ) -> EEGPreprocessingResult:
        """
        Process raw EEG array.

        Args:
            raw: shape (n_samples, 2) — [Fp1, Fp2]
            source_id: Recording identifier for provenance.
            normalization_stats: Pre-computed stats from train set.

        Returns:
            EEGPreprocessingResult.
        """
        if raw.shape[1] != 2:
            raise ValueError(
                f"Expected 2 EEG channels (Fp1, Fp2), got shape {raw.shape}"
            )

        provenance = {
            "source_id": source_id,
            "preprocessing_version": self.config.preprocessing_version,
            "input_shape": list(raw.shape),
            "steps_applied": [],
        }
        data = raw.astype(float).copy()
        steps = provenance["steps_applied"]

        # Step 1: Quality checks
        if not np.all(np.isfinite(data)):
            raise ValueError(
                f"EEG data for {source_id!r} contains NaN or Inf. Run QC first."
            )
        steps.append("integrity_check")

        # Step 2: Unit conversion
        if self.config.voltage_conversion_factor is not None:
            data = data * self.config.voltage_conversion_factor
            steps.append(f"unit_conversion_x{self.config.voltage_conversion_factor}")
        else:
            steps.append("unit_conversion_SKIPPED_TBD")

        # Step 3: Software bandpass (if hardware has not applied)
        if self.config.sw_bandpass_enabled and not self.config.hardware_bandpass_applied:
            data = self._apply_bandpass(data)
            steps.append(
                f"sw_bandpass_{self.config.sw_bandpass_low_hz}_"
                f"{self.config.sw_bandpass_high_hz}Hz"
            )

        # Step 4: Software notch
        if self.config.sw_notch_enabled and not self.config.hardware_notch_applied:
            data = self._apply_notch(data)
            steps.append(f"sw_notch_{self.config.sw_notch_freq_hz}Hz")

        # Step 5: ICA (optional, not mandatory)
        if self.config.ica_enabled:
            # Placeholder — full ICA requires MNE or sklearn FastICA
            # For 2-channel EEG, ICA offers limited benefit (PRD §22)
            steps.append("ica_PLACEHOLDER_not_implemented")
            warnings.warn(
                "ICA is enabled but not yet implemented for 2-channel EEG.",
                UserWarning,
            )

        # Step 6: Normalization
        if self.config.normalization == "zscore":
            data, stats = self._zscore(data, normalization_stats)
            steps.append("zscore_normalization")
            provenance["normalization_stats"] = stats
        elif self.config.normalization == "minmax":
            data, stats = self._minmax(data, normalization_stats)
            steps.append("minmax_normalization")
            provenance["normalization_stats"] = stats
        else:
            steps.append("normalization_NONE")

        provenance["output_shape"] = list(data.shape)

        return EEGPreprocessingResult(
            data=data,
            channel_names=self.EEG_CHANNELS,
            preprocessing_version=self.config.preprocessing_version,
            config=self.config,
            source_id=source_id,
            provenance=provenance,
        )

    def _apply_bandpass(self, data: np.ndarray) -> np.ndarray:
        nyq = self.config.sampling_rate_hz / 2.0
        low = self.config.sw_bandpass_low_hz / nyq
        high = self.config.sw_bandpass_high_hz / nyq
        b, a = sp_signal.butter(self.config.sw_bandpass_order, [low, high], btype="band")
        return sp_signal.filtfilt(b, a, data, axis=0)

    def _apply_notch(self, data: np.ndarray) -> np.ndarray:
        b, a = sp_signal.iirnotch(
            self.config.sw_notch_freq_hz,
            self.config.sw_notch_q,
            self.config.sampling_rate_hz,
        )
        return sp_signal.filtfilt(b, a, data, axis=0)

    def _zscore(self, data, stats=None):
        if stats is None:
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            std = np.where(std == 0, 1.0, std)
        else:
            mean = np.array(stats["mean"])
            std = np.array(stats["std"])
        return (data - mean) / std, {"mean": mean.tolist(), "std": std.tolist()}

    def _minmax(self, data, stats=None):
        if stats is None:
            mn, mx = data.min(axis=0), data.max(axis=0)
        else:
            mn, mx = np.array(stats["min"]), np.array(stats["max"])
        denom = np.where((mx - mn) == 0, 1.0, mx - mn)
        return (data - mn) / denom, {"min": mn.tolist(), "max": mx.tolist()}
