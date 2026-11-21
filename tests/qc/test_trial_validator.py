"""
Tests for TrialValidator — trial-level QC, never-delete invariant (PRD §15, §49).
"""
import warnings
import pytest

from backend.acquisition.frame_parser import Sample
from backend.protocol.session_manager import TrialRecord, TrialValidity
from backend.qc.signal_qc import SignalQC
from backend.qc.trial_validator import TrialValidator


def make_sample(sample_id: int, ch_val: int = 100) -> Sample:
    return Sample(
        sample_id=sample_id,
        timestamp_us=sample_id * 250,
        EEG_Fp1=ch_val, EEG_Fp2=ch_val,
        EMG_Submental=ch_val, EMG_Orbicularis=ch_val,
        EMG_Zygomaticus=ch_val, EMG_Buccinator=ch_val,
    )


def make_trial(trial_id="T1", intent="YES") -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id,
        subject_id="S001",
        session_id="S001_Sess01",
        intent=intent,
        start_sample=0,
        end_sample=100,
        validity=TrialValidity.VALID,
    )


@pytest.fixture
def validator():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qc = SignalQC()
    return TrialValidator(qc)


class TestTrialValidatorNeverDelete:
    def test_valid_trial_stays_valid(self, validator):
        import numpy as np
        trial = make_trial()
        samples = [make_sample(i, int(100 * np.sin(2 * np.pi * i / 40)))
                   for i in range(100)]
        result = validator.validate_trial(trial, samples)
        # Trial must remain in whatever state it was — never silently removed
        assert result is trial

    def test_empty_samples_marks_missing(self, validator):
        trial = make_trial()
        result = validator.validate_trial(trial, [])
        assert result.validity == TrialValidity.INVALID
        assert result.invalid_reason == "missing_samples"
        # Original TrialRecord object is returned, not deleted
        assert result is trial

    def test_dropped_samples_marks_invalid(self, validator):
        trial = make_trial()
        samples = [make_sample(i) for i in range(50)]
        # Insert gap at position 25
        samples[25] = Sample(
            sample_id=samples[24].sample_id + 10,
            timestamp_us=samples[25].timestamp_us,
            EEG_Fp1=100, EEG_Fp2=100,
            EMG_Submental=100, EMG_Orbicularis=100,
            EMG_Zygomaticus=100, EMG_Buccinator=100,
        )
        result = validator.validate_trial(trial, samples)
        assert result.validity == TrialValidity.INVALID
        assert result.invalid_reason == "missing_samples"

    def test_saturation_marks_invalid(self, validator):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qc_with_adc = SignalQC(adc_max=200, saturation_consecutive=3)
        v = TrialValidator(qc_with_adc)
        trial = make_trial()
        samples = []
        for i in range(50):
            val = 300 if 10 <= i < 15 else 50
            samples.append(Sample(i, i * 250, val, 50, 50, 50, 50, 50))
        result = v.validate_trial(trial, samples)
        assert result.validity == TrialValidity.INVALID
        assert result.invalid_reason == "saturation"

    def test_operator_marked_invalid_preserved(self, validator):
        """Operator-pre-marked invalids must not be overwritten."""
        trial = make_trial()
        trial.mark_invalid("vocalization")  # operator marked
        samples = [make_sample(i) for i in range(50)]
        result = validator.validate_all_trials([trial], {"T1": samples})
        assert result[0].validity == TrialValidity.INVALID
        assert result[0].invalid_reason == "vocalization"

    def test_validity_report_counts_correctly(self, validator):
        trials = [make_trial(f"T{i}", "YES") for i in range(5)]
        trials[0].mark_invalid("movement")
        trials[1].mark_invalid("saturation")
        report = validator.generate_validity_report(trials)
        assert report["total"] == 5
        assert report["valid"] == 3
        assert report["invalid"] == 2
        assert report["invalid_by_reason"]["movement"] == 1
        assert report["invalid_by_reason"]["saturation"] == 1

    def test_save_validity_report(self, validator, tmp_path):
        trials = [make_trial("T1", "YES")]
        out_path = tmp_path / "qc" / "trial_validity.json"
        validator.save_validity_report(trials, out_path)
        assert out_path.exists()
        import json
        with open(out_path) as f:
            data = json.load(f)
        assert "total" in data
