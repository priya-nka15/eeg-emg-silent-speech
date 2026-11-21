"""
Tests for confidence gating / rejection (PRD §35, §49).
"""
import warnings
import pytest

from backend.fusion.confidence import ConfidenceConfig, ConfidenceGate
from backend.models.base import Prediction
from backend.protocol.output_states import IntentOutput


def make_prediction(label, confidence):
    return Prediction(
        label=label,
        probabilities={label: confidence},
        confidence=confidence,
    )


class TestConfidenceGate:

    def test_high_confidence_passes(self):
        cfg = ConfidenceConfig(emg_min_confidence=0.6, eeg_min_confidence=0.6)
        gate = ConfidenceGate(cfg)
        pred = make_prediction("YES", confidence=0.9)
        result = gate.check_emg(pred, signal_quality_ok=True)
        assert result.label == "YES"

    def test_low_confidence_rejected(self):
        cfg = ConfidenceConfig(emg_min_confidence=0.7, eeg_min_confidence=0.7)
        gate = ConfidenceGate(cfg)
        pred = make_prediction("YES", confidence=0.5)
        result = gate.check_emg(pred, signal_quality_ok=True)
        assert result.label == IntentOutput.NO_DECISION

    def test_bad_signal_quality_rejected(self):
        cfg = ConfidenceConfig(emg_min_confidence=0.0, eeg_min_confidence=0.0)
        gate = ConfidenceGate(cfg)
        pred = make_prediction("YES", confidence=0.99)
        result = gate.check_emg(pred, signal_quality_ok=False)
        assert result.label == IntentOutput.NO_DECISION

    def test_tbd_threshold_does_not_reject_on_confidence(self):
        # When threshold is None (TBD), confidence gating is inactive
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = ConfidenceConfig(emg_min_confidence=None, eeg_min_confidence=None)
            gate = ConfidenceGate(cfg)
        pred = make_prediction("HELP", confidence=0.01)
        result = gate.check_emg(pred, signal_quality_ok=True)
        assert result.label == "HELP"  # Not rejected — threshold inactive

    def test_eeg_confidence_gate(self):
        cfg = ConfidenceConfig(emg_min_confidence=0.5, eeg_min_confidence=0.8)
        gate = ConfidenceGate(cfg)
        pred = make_prediction("CONDITION_A", confidence=0.7)
        result = gate.check_eeg(pred, signal_quality_ok=True)
        assert result.label == IntentOutput.NO_DECISION

    def test_no_decision_has_zero_confidence(self):
        nd = Prediction.no_decision("test reason")
        assert nd.label == IntentOutput.NO_DECISION
        assert nd.confidence == 0.0
        assert not nd.quality_ok
