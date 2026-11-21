"""
Tests for decision-level fusion (PRD §49 Fusion).
"""
import pytest
from backend.fusion.decision_fusion import DecisionFusion, FusionResult
from backend.models.base import Prediction
from backend.protocol.output_states import IntentOutput


def make_prediction(label, confidence=0.9):
    probs = {"YES": 0.0, "NO": 0.0, "HELP": 0.0}
    if label in probs:
        probs[label] = confidence
    return Prediction(
        label=label,
        probabilities=probs,
        confidence=confidence,
    )


def make_no_decision(reason="test"):
    return Prediction.no_decision(reason)


class TestDecisionFusion:

    def test_both_available_uses_emg_intent(self):
        fusion = DecisionFusion()
        emg = make_prediction("YES")
        eeg = make_prediction("CONDITION_A")
        result = fusion.fuse(emg, eeg)
        assert result.intent == "YES"
        assert result.is_decided

    def test_emg_none_gives_no_decision(self):
        fusion = DecisionFusion()
        result = fusion.fuse(None, make_prediction("CONDITION_A"))
        assert result.intent == IntentOutput.NO_DECISION
        assert "EMG unavailable" in result.reason

    def test_emg_no_decision_propagates(self):
        fusion = DecisionFusion()
        emg = make_no_decision("low confidence")
        result = fusion.fuse(emg, make_prediction("CONDITION_A"))
        assert result.intent == IntentOutput.NO_DECISION
        assert "EMG NO_DECISION" in result.reason

    def test_eeg_unavailable_emg_still_passes(self):
        fusion = DecisionFusion(require_eeg=False)
        emg = make_prediction("NO")
        result = fusion.fuse(emg, None)
        assert result.intent == "NO"
        assert "EEG unavailable" in result.reason

    def test_require_eeg_blocks_when_eeg_none(self):
        fusion = DecisionFusion(require_eeg=True)
        emg = make_prediction("HELP")
        result = fusion.fuse(emg, None)
        assert result.intent == IntentOutput.NO_DECISION

    def test_eeg_no_decision_does_not_block_intent(self):
        fusion = DecisionFusion()
        emg = make_prediction("HELP")
        eeg = make_no_decision("EEG uncertain")
        result = fusion.fuse(emg, eeg)
        # EEG NO_DECISION should not block EMG intent
        assert result.intent == "HELP"

    def test_both_none_gives_no_decision(self):
        fusion = DecisionFusion()
        result = fusion.fuse(None, None)
        assert result.intent == IntentOutput.NO_DECISION

    def test_result_contains_emg_prediction(self):
        fusion = DecisionFusion()
        emg = make_prediction("YES")
        result = fusion.fuse(emg, None)
        assert result.emg_prediction is not None
        assert result.emg_prediction.label == "YES"

    def test_fusion_method_is_decision_level(self):
        fusion = DecisionFusion()
        result = fusion.fuse(make_prediction("NO"), None)
        assert result.fusion_method == "decision_level"
