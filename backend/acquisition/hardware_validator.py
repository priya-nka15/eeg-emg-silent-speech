"""
URF/A-SSR Hardware Validator
PRD §9 — Hardware validation gates H1–H8.
Requires physical hardware. Gates that cannot be run without hardware
return a NOT_STARTED result with an explanation.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np

from backend.acquisition.frame_parser import Sample, CHANNEL_NAMES


class GateResult(str, Enum):
    PASS        = "PASS"
    FAIL        = "FAIL"
    NOT_STARTED = "NOT_STARTED"
    BLOCKED     = "BLOCKED"  # blocked on hardware availability


@dataclass
class ValidationResult:
    gate: str
    result: GateResult
    description: str
    details: dict = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class HardwareValidator:
    """
    Implements hardware validation gates H1–H8 (PRD §9).

    Human data collection must not begin until all gates pass.
    """

    # ── H1: Channel mapping ──────────────────────────────────────────────────

    def validate_channel_mapping(
        self,
        probe_results: Optional[dict] = None,
    ) -> ValidationResult:
        """
        H1 — Verify each software channel maps to the correct anatomical location.

        probe_results: dict of {channel_name: verified_location} from hardware test.
        If None, returns NOT_STARTED (requires physical hardware).
        """
        if probe_results is None:
            return ValidationResult(
                gate="H1_channel_mapping",
                result=GateResult.NOT_STARTED,
                description="Requires physical hardware and electrode placement verification.",
                details={
                    "required_mappings": {
                        "EEG_Fp1": "Fp1",
                        "EEG_Fp2": "Fp2",
                        "EMG_Submental": "Submental/anterior-digastric",
                        "EMG_Orbicularis": "Orbicularis oris",
                        "EMG_Zygomaticus": "Zygomaticus major",
                        "EMG_Buccinator": "Buccinator",
                    }
                },
            )

        expected = {
            "EEG_Fp1": "Fp1",
            "EEG_Fp2": "Fp2",
            "EMG_Submental": "Submental/anterior-digastric",
            "EMG_Orbicularis": "Orbicularis oris",
            "EMG_Zygomaticus": "Zygomaticus major",
            "EMG_Buccinator": "Buccinator",
        }
        failures = {}
        for ch, expected_loc in expected.items():
            actual = probe_results.get(ch)
            if actual != expected_loc:
                failures[ch] = {"expected": expected_loc, "actual": actual}

        result = GateResult.PASS if not failures else GateResult.FAIL
        return ValidationResult(
            gate="H1_channel_mapping",
            result=result,
            description="Channel mapping verification",
            details={"failures": failures, "verified": probe_results},
        )

    # ── H2: Sampling rate ────────────────────────────────────────────────────

    def validate_sampling_rate(
        self,
        samples: Optional[List[Sample]] = None,
        target_hz: float = 4000.0,
        tolerance_pct: float = 1.0,
    ) -> ValidationResult:
        """
        H2 — Measure actual samples/second/channel from counter/timestamps.
        Never assume the configured rate; measure from data.
        """
        if samples is None or len(samples) < 2:
            return ValidationResult(
                gate="H2_sampling_rate",
                result=GateResult.NOT_STARTED,
                description="Requires live sample stream from hardware.",
            )

        n = len(samples)
        duration_us = samples[-1].timestamp_us - samples[0].timestamp_us
        if duration_us <= 0:
            return ValidationResult(
                gate="H2_sampling_rate",
                result=GateResult.FAIL,
                description="Invalid timestamps (duration <= 0)",
            )

        measured_hz = (n - 1) / (duration_us / 1e6)
        deviation_pct = abs(measured_hz - target_hz) / target_hz * 100.0
        passed = deviation_pct <= tolerance_pct

        # Counter-based estimate
        id_delta = samples[-1].sample_id - samples[0].sample_id
        counter_hz = id_delta / (duration_us / 1e6) if duration_us > 0 else None

        return ValidationResult(
            gate="H2_sampling_rate",
            result=GateResult.PASS if passed else GateResult.FAIL,
            description="Sampling rate measurement from timestamps and counter",
            details={
                "target_hz": target_hz,
                "measured_hz": round(measured_hz, 2),
                "counter_hz": round(counter_hz, 2) if counter_hz else None,
                "deviation_pct": round(deviation_pct, 3),
                "tolerance_pct": tolerance_pct,
                "n_samples": n,
                "duration_us": duration_us,
            },
        )

    # ── H3: Synchronization ──────────────────────────────────────────────────

    def validate_synchronization(
        self,
        samples: Optional[List[Sample]] = None,
    ) -> ValidationResult:
        """
        H3 — Structural synchronisation check.

        What this gate can verify in software (without physical hardware):
          - Every Sample record carries a single sample_id and timestamp that
            covers all 6 channels simultaneously, consistent with a
            simultaneous-sampling AFE.
          - sample_id values are monotonically increasing with no unexpected
            repeats (which would indicate duplicate or out-of-order frames).
          - Timestamps are strictly non-decreasing.

        What this gate CANNOT verify without physical hardware:
          - That the AFE actually sampled all channels at the same physical
            instant (simultaneous-acquisition property of ADS1298-class AFE).
          - Per-channel timing skew at the analogue front-end level.

        Physical simultaneous-acquisition validation requires hardware
        measurement (e.g., injecting a known signal and checking phase).
        This gate will report INFORMATIONAL, not PASS, because software
        alone cannot confirm hardware simultaneity.
        """
        if samples is None or len(samples) < 10:
            return ValidationResult(
                gate="H3_synchronization",
                result=GateResult.NOT_STARTED,
                description=(
                    "Requires a live sample stream from hardware. "
                    "Physical simultaneous-acquisition cannot be verified in software."
                ),
            )

        issues = []

        # Check 1: monotonically increasing sample_ids (no duplicates/reorder)
        prev_id = None
        n_non_monotone = 0
        for i, s in enumerate(samples):
            if prev_id is not None and s.sample_id <= prev_id:
                n_non_monotone += 1
                if len(issues) < 5:  # report first 5 only
                    issues.append(
                        f"sample_id not monotonically increasing at position {i}: "
                        f"{prev_id} → {s.sample_id}"
                    )
            prev_id = s.sample_id

        # Check 2: timestamps non-decreasing
        n_ts_decreasing = 0
        for i in range(1, len(samples)):
            if samples[i].timestamp_us < samples[i - 1].timestamp_us:
                n_ts_decreasing += 1
                if len(issues) < 5:
                    issues.append(
                        f"timestamp decreased at position {i}: "
                        f"{samples[i-1].timestamp_us} → {samples[i].timestamp_us}"
                    )

        structural_ok = (n_non_monotone == 0 and n_ts_decreasing == 0)

        return ValidationResult(
            gate="H3_synchronization",
            result=GateResult.NOT_STARTED,  # software cannot confirm hardware simultaneity
            description=(
                "Structural synchronisation check (software only). "
                "Physical simultaneous-acquisition requires hardware measurement — "
                "this gate returns NOT_STARTED regardless of structural results."
            ),
            details={
                "structural_check_passed": structural_ok,
                "n_samples_checked": len(samples),
                "non_monotone_sample_ids": n_non_monotone,
                "decreasing_timestamps": n_ts_decreasing,
                "structural_issues": issues,
                "note": (
                    "PASS requires physical hardware evidence of simultaneous "
                    "sampling (e.g. injected test signal phase verification). "
                    "Software can only verify structural properties of the "
                    "data stream, not analogue simultaneity."
                ),
            },
        )

    # ── H4: Dropped samples ──────────────────────────────────────────────────

    def validate_dropped_samples(
        self,
        samples: Optional[List[Sample]] = None,
        max_allowed: int = 0,
    ) -> ValidationResult:
        """
        H4 — Compare expected vs actual sample counts; detect counter gaps.
        """
        if samples is None or len(samples) < 2:
            return ValidationResult(
                gate="H4_dropped_samples",
                result=GateResult.NOT_STARTED,
                description="Requires live sample stream from hardware.",
            )

        gaps = []
        for i in range(1, len(samples)):
            expected_id = (samples[i - 1].sample_id + 1) & 0xFFFFFFFF
            actual_id = samples[i].sample_id
            if actual_id != expected_id:
                gap = (actual_id - samples[i - 1].sample_id) & 0xFFFFFFFF
                gaps.append({
                    "position": i,
                    "prev_sample_id": samples[i - 1].sample_id,
                    "next_sample_id": actual_id,
                    "dropped": gap - 1,
                })

        total_dropped = sum(g["dropped"] for g in gaps)
        passed = total_dropped <= max_allowed

        return ValidationResult(
            gate="H4_dropped_samples",
            result=GateResult.PASS if passed else GateResult.FAIL,
            description="Sample counter continuity check",
            details={
                "total_samples": len(samples),
                "total_dropped": total_dropped,
                "gaps": gaps,
                "max_allowed": max_allowed,
            },
        )

    # ── H5: Saturation ───────────────────────────────────────────────────────

    def validate_saturation(
        self,
        samples: Optional[List[Sample]] = None,
        adc_max: Optional[int] = None,
        consecutive_threshold: int = 3,
    ) -> ValidationResult:
        """
        H5 — Detect repeated ADC limit hits.
        adc_max: the 24-bit positive saturation value (TBD from hardware).
        """
        if samples is None:
            return ValidationResult(
                gate="H5_saturation",
                result=GateResult.NOT_STARTED,
                description="Requires live sample stream from hardware.",
            )

        if adc_max is None:
            return ValidationResult(
                gate="H5_saturation",
                result=GateResult.BLOCKED,
                description="adc_max is TBD — requires hardware specification.",
            )

        saturation_events: dict[str, int] = {ch: 0 for ch in CHANNEL_NAMES}
        ch_attrs = CHANNEL_NAMES

        for ch in ch_attrs:
            consecutive = 0
            prev = None
            for s in samples:
                val = getattr(s, ch)
                if abs(val) >= adc_max:
                    consecutive += 1
                    if consecutive >= consecutive_threshold:
                        saturation_events[ch] += 1
                else:
                    consecutive = 0
                prev = val

        total_events = sum(saturation_events.values())
        return ValidationResult(
            gate="H5_saturation",
            result=GateResult.PASS if total_events == 0 else GateResult.FAIL,
            description="Saturation detection",
            details={
                "adc_max": adc_max,
                "consecutive_threshold": consecutive_threshold,
                "saturation_events_per_channel": saturation_events,
                "total_saturation_events": total_events,
            },
        )

    # ── H6: Flatline ─────────────────────────────────────────────────────────

    def validate_flatline(
        self,
        samples: Optional[List[Sample]] = None,
        variance_threshold: float = 1.0,
        window_size: int = 100,
    ) -> ValidationResult:
        """
        H6 — Detect suspiciously low-variance channels.
        """
        if samples is None or len(samples) < window_size:
            return ValidationResult(
                gate="H6_flatline",
                result=GateResult.NOT_STARTED,
                description="Requires live sample stream from hardware.",
            )

        flatline_events: dict[str, int] = {}
        for ch in CHANNEL_NAMES:
            values = np.array([getattr(s, ch) for s in samples], dtype=float)
            n_windows = len(values) // window_size
            events = 0
            for w in range(n_windows):
                segment = values[w * window_size:(w + 1) * window_size]
                if np.var(segment) < variance_threshold:
                    events += 1
            flatline_events[ch] = events

        total = sum(flatline_events.values())
        return ValidationResult(
            gate="H6_flatline",
            result=GateResult.PASS if total == 0 else GateResult.FAIL,
            description="Flatline (low-variance) detection",
            details={
                "variance_threshold": variance_threshold,
                "window_size": window_size,
                "flatline_events_per_channel": flatline_events,
            },
        )

    # ── H7: Noise ────────────────────────────────────────────────────────────

    def validate_noise(
        self,
        electronics_only_samples: Optional[List[Sample]] = None,
        contact_samples: Optional[List[Sample]] = None,
        controlled_signal_samples: Optional[List[Sample]] = None,
    ) -> ValidationResult:
        """
        H7 — Noise characterization under three conditions.
        """
        if all(s is None for s in [
            electronics_only_samples, contact_samples, controlled_signal_samples
        ]):
            return ValidationResult(
                gate="H7_noise",
                result=GateResult.NOT_STARTED,
                description=(
                    "Requires hardware test under 3 conditions: "
                    "electronics-only, contact/electrode, controlled signal."
                ),
            )

        results = {}
        for label, sample_list in [
            ("electronics_only", electronics_only_samples),
            ("contact_electrode", contact_samples),
            ("controlled_signal", controlled_signal_samples),
        ]:
            if sample_list is None:
                results[label] = "NOT_TESTED"
                continue
            per_ch = {}
            for ch in CHANNEL_NAMES:
                values = np.array([getattr(s, ch) for s in sample_list], dtype=float)
                per_ch[ch] = {
                    "rms": float(np.sqrt(np.mean(values ** 2))),
                    "std": float(np.std(values)),
                }
            results[label] = per_ch

        return ValidationResult(
            gate="H7_noise",
            result=GateResult.PASS,
            description="Noise characterization (informational — no pass/fail threshold)",
            details=results,
        )

    # ── H8: Sustained streaming ──────────────────────────────────────────────

    def validate_sustained_streaming(
        self,
        samples: Optional[List[Sample]] = None,
        target_duration_s: float = 300.0,
        max_drop_rate_ppm: float = 100.0,
        expected_sampling_rate_hz: Optional[float] = None,
    ) -> ValidationResult:
        """
        H8 — Stress test for buffer overflow, transport loss, firmware stability.

        Args:
            samples: Samples collected during the streaming test.
            target_duration_s: Minimum required test duration in seconds.
            max_drop_rate_ppm: Maximum acceptable drop rate in parts per million.
            expected_sampling_rate_hz: The expected/configured hardware sample rate.
                MUST be supplied and must come from actual hardware specification,
                NOT assumed from the 4 kSPS target. If not supplied, the expected
                sample count cannot be computed and the gate returns NOT_STARTED
                rather than using a hard-coded assumption.
        """
        if samples is None or len(samples) < 100:
            return ValidationResult(
                gate="H8_sustained_streaming",
                result=GateResult.NOT_STARTED,
                description=(
                    f"Requires sustained streaming test of ≥{target_duration_s}s. "
                    "Reveals buffer overflow, transport loss, memory leaks, "
                    "firmware instability."
                ),
            )

        if expected_sampling_rate_hz is None:
            return ValidationResult(
                gate="H8_sustained_streaming",
                result=GateResult.NOT_STARTED,
                description=(
                    "expected_sampling_rate_hz not supplied. "
                    "Cannot compute expected sample count without hardware-verified "
                    "sample rate. Supply the rate measured by H2 (not the 4 kSPS "
                    "target) to run this gate (PRD rule 1 — never invent hardware specs)."
                ),
            )

        duration_s = (samples[-1].timestamp_us - samples[0].timestamp_us) / 1e6
        if duration_s < target_duration_s * 0.9:
            return ValidationResult(
                gate="H8_sustained_streaming",
                result=GateResult.FAIL,
                description=f"Test duration {duration_s:.1f}s < required {target_duration_s}s",
                details={
                    "duration_s": round(duration_s, 1),
                    "target_duration_s": target_duration_s,
                },
            )

        # Count gaps using sample_id counter
        dropped = 0
        for i in range(1, len(samples)):
            expected = (samples[i - 1].sample_id + 1) & 0xFFFFFFFF
            if samples[i].sample_id != expected:
                gap = (samples[i].sample_id - samples[i - 1].sample_id) & 0xFFFFFFFF
                dropped += gap - 1

        # Expected sample count from the hardware-verified rate, NOT from an assumption
        total_expected = int(duration_s * expected_sampling_rate_hz)
        drop_rate_ppm = (dropped / total_expected * 1e6) if total_expected > 0 else 0.0
        passed = drop_rate_ppm <= max_drop_rate_ppm

        return ValidationResult(
            gate="H8_sustained_streaming",
            result=GateResult.PASS if passed else GateResult.FAIL,
            description="Sustained streaming stability",
            details={
                "duration_s": round(duration_s, 1),
                "target_duration_s": target_duration_s,
                "n_samples": len(samples),
                "expected_sampling_rate_hz": expected_sampling_rate_hz,
                "total_expected_samples": total_expected,
                "dropped_samples": dropped,
                "drop_rate_ppm": round(drop_rate_ppm, 2),
                "max_allowed_ppm": max_drop_rate_ppm,
            },
        )

    # ── Run all gates ─────────────────────────────────────────────────────────

    def run_all(self, **kwargs) -> List[ValidationResult]:
        """Run all validation gates and return results."""
        results = [
            self.validate_channel_mapping(kwargs.get("probe_results")),
            self.validate_sampling_rate(
                kwargs.get("samples"),
                target_hz=kwargs.get("target_hz", kwargs.get("expected_sampling_rate_hz") or 4000.0),
            ),
            self.validate_synchronization(kwargs.get("samples")),
            self.validate_dropped_samples(kwargs.get("samples")),
            self.validate_saturation(kwargs.get("samples"), kwargs.get("adc_max")),
            self.validate_flatline(kwargs.get("samples")),
            self.validate_noise(
                kwargs.get("electronics_only_samples"),
                kwargs.get("contact_samples"),
                kwargs.get("controlled_signal_samples"),
            ),
            self.validate_sustained_streaming(
                kwargs.get("sustained_samples"),
                expected_sampling_rate_hz=kwargs.get("expected_sampling_rate_hz"),
            ),
        ]
        return results

    def save_report(self, results: List[ValidationResult], output_path: Path) -> None:
        """Save validation report to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "report_version": "1",
            "human_data_collection_approved": all(
                r.result == GateResult.PASS for r in results
            ),
            "gates": [r.to_dict() for r in results],
        }
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    @staticmethod
    def check_human_use_ready(hardware_config_path: Path) -> bool:
        """
        Check whether human_use_ready is true in hardware config.
        Live mode must call this before starting.
        """
        import yaml
        with open(hardware_config_path) as f:
            cfg = yaml.safe_load(f)
        return bool(cfg.get("human_use_ready", False))
