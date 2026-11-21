"""
URF/A-SSR Signal Quality Control
PRD §49 QC tests — missing samples, flatline, saturation, invalid values.
Thresholds that require pilot/hardware validation are configurable.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from backend.acquisition.frame_parser import CHANNEL_NAMES, Sample


@dataclass
class ChannelQCResult:
    channel: str
    n_samples: int
    n_nan: int = 0
    n_inf: int = 0
    n_out_of_range: int = 0
    n_flatline_windows: int = 0
    n_saturation_runs: int = 0
    n_lead_off_events: int = 0
    rms: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    passed: bool = True
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SessionQCResult:
    session_id: str
    total_samples: int
    dropped_samples: int
    measured_sample_rate_hz: Optional[float]
    per_channel: Dict[str, ChannelQCResult] = field(default_factory=dict)
    passed: bool = True
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["per_channel"] = {k: v.to_dict() for k, v in self.per_channel.items()}
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class SignalQC:
    """
    Runs automated quality control on a list of samples.

    Thresholds that are HARDWARE-TBD or PILOT-VALIDATED are passed as
    constructor arguments. When a threshold is not yet known, the
    corresponding check is either skipped or flagged with a warning.

    Do not treat default threshold values as validated — they are initial
    placeholders that must be replaced after pilot analysis and hardware
    confirmation.
    """

    def __init__(
        self,
        # HARDWARE-TBD threshold — None until confirmed from hardware spec
        adc_max: Optional[int] = None,          # 24-bit positive saturation value
        # PILOT-VALIDATED thresholds — defaults are UNVALIDATED placeholders
        flatline_variance_threshold: float = 1.0,
        flatline_window_size: int = 100,
        saturation_consecutive: int = 3,
        high_amplitude_factor: float = 10.0,    # relative to session RMS
        target_sample_rate_hz: float = 4000.0,
        rate_tolerance_pct: float = 1.0,
    ) -> None:
        import warnings

        self.adc_max = adc_max
        self.flatline_variance_threshold = flatline_variance_threshold
        self.flatline_window_size = flatline_window_size
        self.saturation_consecutive = saturation_consecutive
        self.high_amplitude_factor = high_amplitude_factor
        self.target_sample_rate_hz = target_sample_rate_hz
        self.rate_tolerance_pct = rate_tolerance_pct

        # Warn when critical thresholds are in their unvalidated default state
        if adc_max is None:
            warnings.warn(
                "SignalQC: adc_max is None (HARDWARE-TBD). "
                "Saturation detection is DISABLED until the hardware ADC "
                "saturation level is confirmed. Do not interpret a QC 'passed' "
                "result as confirming absence of saturation.",
                UserWarning,
                stacklevel=2,
            )
        warnings.warn(
            "SignalQC: flatline_variance_threshold and other thresholds are "
            "UNVALIDATED DEFAULTS (PILOT-VALIDATED status). Results should not "
            "be treated as scientifically validated until pilot analysis confirms "
            "appropriate thresholds for this hardware configuration.",
            UserWarning,
            stacklevel=2,
        )

    def run(self, samples: List[Sample], session_id: str = "") -> SessionQCResult:
        """Run all QC checks on a list of samples."""
        n = len(samples)
        if n == 0:
            return SessionQCResult(
                session_id=session_id,
                total_samples=0,
                dropped_samples=0,
                measured_sample_rate_hz=None,
                passed=False,
                notes=["No samples provided"],
            )

        dropped = self._count_dropped(samples)
        measured_hz = self._measure_rate(samples)

        per_channel = {}
        overall_pass = True
        notes = []

        for ch_idx, ch in enumerate(CHANNEL_NAMES):
            values = np.array([getattr(s, ch) for s in samples], dtype=float)
            ch_result = self._check_channel(ch, values, samples, ch_idx)
            per_channel[ch] = ch_result
            if not ch_result.passed:
                overall_pass = False

        # Rate check
        if measured_hz is not None:
            dev_pct = abs(measured_hz - self.target_sample_rate_hz) / self.target_sample_rate_hz * 100
            if dev_pct > self.rate_tolerance_pct:
                overall_pass = False
                notes.append(
                    f"Sample rate {measured_hz:.1f} Hz deviates "
                    f"{dev_pct:.2f}% from target {self.target_sample_rate_hz} Hz"
                )

        if dropped > 0:
            notes.append(f"{dropped} dropped samples detected")

        return SessionQCResult(
            session_id=session_id,
            total_samples=n,
            dropped_samples=dropped,
            measured_sample_rate_hz=measured_hz,
            per_channel=per_channel,
            passed=overall_pass,
            notes=notes,
        )

    # ── Private checks ────────────────────────────────────────────────────────

    def _check_channel(
        self,
        ch: str,
        values: np.ndarray,
        samples: List[Sample],
        ch_idx: int,
    ) -> ChannelQCResult:
        n = len(values)
        result = ChannelQCResult(channel=ch, n_samples=n)

        # NaN / Inf
        result.n_nan = int(np.sum(np.isnan(values)))
        result.n_inf = int(np.sum(np.isinf(values)))
        if result.n_nan > 0 or result.n_inf > 0:
            result.passed = False
            result.notes.append(f"NaN: {result.n_nan}, Inf: {result.n_inf}")

        # 24-bit signed range
        valid_range = (-2**23, 2**23 - 1)
        out_of_range = int(np.sum(
            (values < valid_range[0]) | (values > valid_range[1])
        ))
        result.n_out_of_range = out_of_range
        if out_of_range > 0:
            result.passed = False
            result.notes.append(f"{out_of_range} values outside 24-bit signed range")

        clean = values[np.isfinite(values)]
        if len(clean) == 0:
            result.passed = False
            return result

        result.mean = float(np.mean(clean))
        result.std = float(np.std(clean))
        result.rms = float(np.sqrt(np.mean(clean ** 2)))

        # Flatline
        n_windows = len(clean) // self.flatline_window_size
        flatline_count = 0
        for w in range(n_windows):
            seg = clean[w * self.flatline_window_size:(w + 1) * self.flatline_window_size]
            if np.var(seg) < self.flatline_variance_threshold:
                flatline_count += 1
        result.n_flatline_windows = flatline_count
        if flatline_count > 0:
            result.notes.append(
                f"{flatline_count} flatline windows (var < {self.flatline_variance_threshold})"
            )

        # Saturation (ADC max — TBD from hardware)
        if self.adc_max is not None:
            sat_count = 0
            run = 0
            for v in clean:
                if abs(v) >= self.adc_max:
                    run += 1
                    if run >= self.saturation_consecutive:
                        sat_count += 1
                        run = 0
                else:
                    run = 0
            result.n_saturation_runs = sat_count
            if sat_count > 0:
                result.passed = False
                result.notes.append(f"{sat_count} saturation runs detected")
        else:
            result.notes.append("Saturation threshold TBD — not checked")

        # Lead-off events
        lead_off_count = sum(1 for s in samples if s.is_lead_off(ch_idx))
        result.n_lead_off_events = lead_off_count
        if lead_off_count > 0:
            result.notes.append(f"{lead_off_count} lead-off events")

        return result

    def _count_dropped(self, samples: List[Sample]) -> int:
        dropped = 0
        for i in range(1, len(samples)):
            expected = (samples[i - 1].sample_id + 1) & 0xFFFFFFFF
            if samples[i].sample_id != expected:
                dropped += (samples[i].sample_id - samples[i - 1].sample_id - 1) & 0xFFFFFFFF
        return dropped

    def _measure_rate(self, samples: List[Sample]) -> Optional[float]:
        if len(samples) < 2:
            return None
        duration_us = samples[-1].timestamp_us - samples[0].timestamp_us
        if duration_us <= 0:
            return None
        return (len(samples) - 1) / (duration_us / 1e6)
