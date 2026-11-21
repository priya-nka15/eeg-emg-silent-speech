"""
Tests for dataset schema validation (PRD §49 Dataset tests).
"""
import pytest
from backend.datasets.schema import (
    validate_subject, validate_session, validate_trial, validate_sample
)


class TestSubjectValidation:

    def test_valid_subject(self):
        d = {"subject_id": "S001", "consent_status": "CONSENTED"}
        assert validate_subject(d) == []

    def test_missing_subject_id(self):
        d = {"consent_status": "CONSENTED"}
        errors = validate_subject(d)
        assert any("subject_id" in e for e in errors)

    def test_invalid_consent_status(self):
        d = {"subject_id": "S001", "consent_status": "MAYBE"}
        errors = validate_subject(d)
        assert any("consent_status" in e for e in errors)

    def test_no_spaces_in_subject_id(self):
        d = {"subject_id": "John Smith", "consent_status": "CONSENTED"}
        errors = validate_subject(d)
        assert any("space" in e.lower() for e in errors)


class TestTrialValidation:

    def test_valid_trial(self):
        d = {
            "trial_id": "S001_Sess01_T001",
            "subject_id": "S001",
            "session_id": "S001_Sess01",
            "intent": "YES",
            "start_sample": 0,
            "end_sample": 400,
            "validity": "VALID",
        }
        assert validate_trial(d) == []

    def test_invalid_intent(self):
        d = {
            "trial_id": "T1",
            "subject_id": "S001",
            "session_id": "Sess01",
            "intent": "MAYBE",
            "start_sample": 0,
            "end_sample": 100,
            "validity": "VALID",
        }
        errors = validate_trial(d)
        assert any("intent" in e for e in errors)

    def test_invalid_without_reason_fails(self):
        d = {
            "trial_id": "T1",
            "subject_id": "S001",
            "session_id": "Sess01",
            "intent": "YES",
            "start_sample": 0,
            "end_sample": 100,
            "validity": "INVALID",
        }
        errors = validate_trial(d)
        assert any("invalid_reason" in e for e in errors)

    def test_invalid_with_valid_reason(self):
        d = {
            "trial_id": "T1",
            "subject_id": "S001",
            "session_id": "Sess01",
            "intent": "HELP",
            "start_sample": 0,
            "end_sample": 100,
            "validity": "INVALID",
            "invalid_reason": "saturation",
        }
        assert validate_trial(d) == []

    def test_invalid_with_bad_reason(self):
        d = {
            "trial_id": "T1",
            "subject_id": "S001",
            "session_id": "Sess01",
            "intent": "NO",
            "start_sample": 0,
            "end_sample": 100,
            "validity": "INVALID",
            "invalid_reason": "bad_vibes",
        }
        errors = validate_trial(d)
        assert any("invalid_reason" in e for e in errors)

    def test_baseline_and_eeg_task_intents_valid(self):
        for intent in ("BASELINE", "EEG_TASK"):
            d = {
                "trial_id": "T1",
                "subject_id": "S001",
                "session_id": "Sess01",
                "intent": intent,
                "start_sample": 0,
                "end_sample": 100,
                "validity": "VALID",
            }
            assert validate_trial(d) == []


class TestSampleValidation:

    def test_valid_sample(self):
        d = {
            "sample_id": 0,
            "timestamp_us": 0,
            "EEG_Fp1": 100,
            "EEG_Fp2": -100,
            "EMG_Submental": 200,
            "EMG_Orbicularis": -200,
            "EMG_Zygomaticus": 300,
            "EMG_Buccinator": -300,
        }
        assert validate_sample(d) == []

    def test_out_of_range_24bit(self):
        d = {
            "sample_id": 0,
            "timestamp_us": 0,
            "EEG_Fp1": 2**24,  # too big
            "EEG_Fp2": 0,
            "EMG_Submental": 0,
            "EMG_Orbicularis": 0,
            "EMG_Zygomaticus": 0,
            "EMG_Buccinator": 0,
        }
        errors = validate_sample(d)
        assert any("EEG_Fp1" in e for e in errors)

    def test_missing_channel(self):
        d = {
            "sample_id": 0,
            "timestamp_us": 0,
            "EEG_Fp1": 0,
            # EEG_Fp2 missing
            "EMG_Submental": 0,
            "EMG_Orbicularis": 0,
            "EMG_Zygomaticus": 0,
            "EMG_Buccinator": 0,
        }
        errors = validate_sample(d)
        assert any("EEG_Fp2" in e for e in errors)
