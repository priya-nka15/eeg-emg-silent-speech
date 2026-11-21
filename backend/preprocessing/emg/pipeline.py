"""
URF/A-SSR EMG Preprocessing Pipeline
PRD §24 — EMG preprocessing pipeline.

Filter parameters are configurable and must NOT be hard-coded.
All hardware-dependent values are TBD.
Software must not blindly repeat hardware filters.

Output provenance: every call records input version + config version.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal


@dataclass
class EMGPreprocessingConfig:
    """
    All filter parameters configurable. TBD fields must be filled before use.
    """
    preprocessing_version: str = "TBD"        # PILOT-VALIDATED

    # Hardware-already-applied (must be verified before enabling software filters)
    hardware_bandpass_applied: bool = False    # HARDWARE-TBD
    hardware_notch_applied: bool = False       # HARDWARE-TBD

    # Software bandpass (only if hardware has NOT applied it)
    sw_bandpass_enabled: bool = False          # PILOT-VALIDATED
    sw_bandpass_low_hz: Optional[float] = None   # PILOT-VALIDATED
    sw_bandpass_high_hz: Optional[float] = None  # PILOT-VALIDATED
    sw_bandpass_order: int = 4                 # PILOT-VALIDATED
    sw_bandpass_filter_type: str = "butter"    # PILOT-VALIDATED

    # Software notch
    sw_notch_enabled: bool = False             # PILOT-VALIDATED
    sw_notch_freq_hz: Optional[float] = None   # PILOT-VALIDATED
    sw_notch_q: float = 30.0                   # PILOT-VALIDATED

    # Normalization
    normalization: Optional[str] = None        # PILOT-VALIDATED: "zscore" | "minmax" | None

    # Sample rate (must match hardware config)
    sampling_rate_hz: float = 4000.0

    # ADC voltage conversion (TBD from hardware)
    voltage_conversion_factor: Optional[float] = None  # HARDWARE-TBD

    def validate(self) -> List[str]:
        """Return list of configuration issues."""
        issues = []
        if self.preprocessing_version == "TBD":
            issues.append("preprocessing_version is TBD — not suitable for final mode")
        if self.sw_bandpass_enabled and (
            self.sw_bandpass_low_hz is None or self.sw_bandpass_high_hz is None
        ):
            issues.append("sw_bandpass enabled but frequencies not set")
        if self.sw_notch_enabled and self.sw_notch_freq_hz is None:
            issues.append("sw_notch enabled but frequency not set")
        if self.voltage_conversion_factor is None:
            issues.append("voltage_conversion_factor is TBD — ADC units not converted")
        return issues


@dataclass
class EMGPreprocessingResult:
    data: np.ndarray                # shape (n_samples, n_channels)
    channel_names: List[str]
    preprocessing_version: str
    config: EMGPreprocessingConfig
    source_id: str = ""             # session or recording identifier
    provenance: dict = field(default_factory=dict)

    def is_finite(self) -> bool:
        return bool(np.all(np.isfinite(self.data)))


class EMGPreprocessingPipeline:
    """
    EMG preprocessing pipeline.
    Steps (PRD §24):
        1. Integrity check
        2. Unit conversion (if required)
        3. Hardware-processing verification
        4. Software QC
        5. Software filtering (if not done by hardware)
        6. Normalization
    """

    EMG_CHANNELS = [
        "EMG_Submental",
        "EMG_Orbicularis",
        "EMG_Zygomaticus",
        "EMG_Buccinator",
    ]

    def __init__(self, config: EMGPreprocessingConfig) -> None:
        self.config = config
        issues = config.validate()
        if issues:
            warnings.warn(
                f"EMGPreprocessingConfig issues: {issues}",
                UserWarning,
                stacklevel=2,
            )

    def process(
        self,
        raw: np.ndarray,
        source_id: str = "",
        normalization_stats: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> EMGPreprocessingResult:
        """
        Process raw EMG array.

        Args:
            raw: shape (n_samples, 4) — EMG channels in order defined by EMG_CHANNELS
            source_id: Recording identifier for provenance.
            normalization_stats: Pre-computed {"mean": ..., "std": ...} from train set.
                If None, stats computed from this data (use for train only).

        Returns:
            EMGPreprocessingResult with processed array and provenance.
        """
        provenance = {
            "source_id": source_id,
            "preprocessing_version": self.config.preprocessing_version,
            "input_shape": list(raw.shape),
            "steps_applied": [],
        }

        data = raw.astype(float).copy()
        steps = provenance["steps_applied"]

        # Step 1: Integrity
        if not np.all(np.isfinite(data)):
            raise ValueError(
                f"Raw EMG data for {source_id!r} contains NaN or Inf. "
                "Check QC before preprocessing."
            )
        steps.append("integrity_check")

        # Step 2: Unit conversion
        if self.config.voltage_conversion_factor is not None:
            data = data * self.config.voltage_conversion_factor
            steps.append(f"unit_conversion_x{self.config.voltage_conversion_factor}")
        else:
            steps.append("unit_conversion_SKIPPED_TBD")

        # Step 3: Hardware verification
        if self.config.hardware_bandpass_applied:
            steps.append("hardware_bandpass_verified_applied")
        if self.config.hardware_notch_applied:
            steps.append("hardware_notch_verified_applied")

        # Step 4: Software bandpass (only if hardware has not applied)
        if self.config.sw_bandpass_enabled and not self.config.hardware_bandpass_applied:
            data = self._apply_bandpass(data)
            steps.append(
                f"sw_bandpass_{self.config.sw_bandpass_low_hz}_"
                f"{self.config.sw_bandpass_high_hz}Hz"
            )
        elif self.config.sw_bandpass_enabled:
            warnings.warn(
                "sw_bandpass_enabled=True but hardware already applied bandpass. "
                "Skipping to avoid double-filtering.",
                UserWarning,
            )

        # Step 5: Software notch
        if self.config.sw_notch_enabled and not self.config.hardware_notch_applied:
            data = self._apply_notch(data)
            steps.append(f"sw_notch_{self.config.sw_notch_freq_hz}Hz")

        # Step 6: Normalization
        if self.config.normalization == "zscore":
            data, stats = self._zscore(data, normalization_stats)
            steps.append(f"zscore_normalization")
            provenance["normalization_stats"] = stats
        elif self.config.normalization == "minmax":
            data, stats = self._minmax(data, normalization_stats)
            steps.append("minmax_normalization")
            provenance["normalization_stats"] = stats
        else:
            steps.append("normalization_NONE")

        provenance["output_shape"] = list(data.shape)

        return EMGPreprocessingResult(
            data=data,
            channel_names=self.EMG_CHANNELS,
            preprocessing_version=self.config.preprocessing_version,
            config=self.config,
            source_id=source_id,
            provenance=provenance,
        )

    def _apply_bandpass(self, data: np.ndarray) -> np.ndarray:
        nyq = self.config.sampling_rate_hz / 2.0
        low = self.config.sw_bandpass_low_hz / nyq
        high = self.config.sw_bandpass_high_hz / nyq
        b, a = sp_signal.butter(
            self.config.sw_bandpass_order,
            [low, high],
            btype="band",
        )
        return sp_signal.filtfilt(b, a, data, axis=0)

    def _apply_notch(self, data: np.ndarray) -> np.ndarray:
        b, a = sp_signal.iirnotch(
            self.config.sw_notch_freq_hz,
            self.config.sw_notch_q,
            self.config.sampling_rate_hz,
        )
        return sp_signal.filtfilt(b, a, data, axis=0)

    def _zscore(
        self,
        data: np.ndarray,
        stats: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        if stats is None:
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            std = np.where(std == 0, 1.0, std)
        else:
            mean = np.array(stats["mean"])
            std = np.array(stats["std"])
        normalized = (data - mean) / std
        return normalized, {"mean": mean.tolist(), "std": std.tolist()}

    def _minmax(
        self,
        data: np.ndarray,
        stats: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        if stats is None:
            mn = data.min(axis=0)
            mx = data.max(axis=0)
        else:
            mn = np.array(stats["min"])
            mx = np.array(stats["max"])
        denom = np.where((mx - mn) == 0, 1.0, mx - mn)
        normalized = (data - mn) / denom
        return normalized, {"min": mn.tolist(), "max": mx.tolist()}
