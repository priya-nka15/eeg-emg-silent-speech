"""
URF/A-SSR Trial Validator
PRD §15 — Trial validity checking. Never silently delete invalid trials.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from backend.protocol.session_manager import TrialRecord, TrialValidity, INVALID_REASONS
from backend.acquisition.frame_parser import Sample
from backend.qc.signal_qc import SignalQC


class TrialValidator:
    """
    Checks trials for validity based on signal quality.
    Invalid trials are flagged, NOT deleted.
    """

    def __init__(self, signal_qc: SignalQC) -> None:
        self.qc = signal_qc

    def validate_trial(
        self,
        trial: TrialRecord,
        samples: List[Sample],
    ) -> TrialRecord:
        """
        Run QC on the samples within a trial window.
        Updates trial.validity and trial.invalid_reason if issues are found.
        The original signal is NEVER deleted.
        """
        if not samples:
            trial.mark_invalid("missing_samples")
            return trial

        qc_result = self.qc.run(samples, session_id=trial.session_id)

        if qc_result.dropped_samples > 0:
            trial.mark_invalid("missing_samples")
            return trial

        for ch_name, ch_result in qc_result.per_channel.items():
            if ch_result.n_nan > 0 or ch_result.n_inf > 0:
                trial.mark_invalid("transmission_failure")
                return trial

            if ch_result.n_saturation_runs > 0:
                trial.mark_invalid("saturation")
                return trial

            # Severe flatline: >50% of windows flatlined is considered electrode issue
            total_windows = max(1, len(samples) // self.qc.flatline_window_size)
            if ch_result.n_flatline_windows > total_windows * 0.5:
                trial.mark_invalid("electrode_detachment")
                return trial

        return trial

    def validate_all_trials(
        self,
        trials: List[TrialRecord],
        samples_by_trial: dict,
    ) -> List[TrialRecord]:
        """
        Validate all trials. Returns updated list.
        Operator-marked invalids are preserved.
        """
        for trial in trials:
            if trial.validity == TrialValidity.INVALID:
                # Already marked by operator — preserve
                continue
            trial_samples = samples_by_trial.get(trial.trial_id, [])
            self.validate_trial(trial, trial_samples)
        return trials

    def generate_validity_report(self, trials: List[TrialRecord]) -> dict:
        """Generate a trial validity summary."""
        total = len(trials)
        valid = sum(1 for t in trials if t.validity == TrialValidity.VALID)
        invalid = total - valid

        by_reason: dict = {}
        for t in trials:
            if t.validity == TrialValidity.INVALID and t.invalid_reason:
                by_reason[t.invalid_reason] = by_reason.get(t.invalid_reason, 0) + 1

        by_class_valid: dict = {}
        for t in trials:
            if t.validity == TrialValidity.VALID:
                by_class_valid[t.intent] = by_class_valid.get(t.intent, 0) + 1

        return {
            "total": total,
            "valid": valid,
            "invalid": invalid,
            "invalid_by_reason": by_reason,
            "valid_by_class": by_class_valid,
        }

    def save_validity_report(self, trials: List[TrialRecord], output_path: Path) -> None:
        report = self.generate_validity_report(trials)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
