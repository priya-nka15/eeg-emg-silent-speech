"""
Tests for temporal consistency filter (PRD §36, §49).
"""
import warnings
import pytest

from backend.fusion.temporal_consistency import (
    TemporalConsistencyFilter, TemporalConsistencyConfig, ConsistencyStrategy
)
from backend.models.base import Prediction
from backend.protocol.output_states import IntentOutput


def make_prediction(label, confidence=0.9):
    return Prediction(label=label, probabilities={}, confidence=confidence)


def make_no_decision():
    return Prediction.no_decision()


class TestTemporalConsistency:

    def test_no_decision_on_insufficient_windows(self):
        cfg = TemporalConsistencyConfig(required_windows=3)
        tcf = TemporalConsistencyFilter(cfg)
        # First two predictions: still accumulating
        r1 = tcf.update(make_prediction("YES"))
        r2 = tcf.update(make_prediction("YES"))
        assert r1.label == IntentOutput.NO_DECISION
        assert r2.label == IntentOutput.NO_DECISION

    def test_consistent_after_n_windows(self):
        cfg = TemporalConsistencyConfig(required_windows=3)
        tcf = TemporalConsistencyFilter(cfg)
        for _ in range(2):
            tcf.update(make_prediction("YES"))
        result = tcf.update(make_prediction("YES"))
        assert result.label == "YES"

    def test_inconsistent_windows_give_no_decision(self):
        cfg = TemporalConsistencyConfig(
            required_windows=3,
            strategy=ConsistencyStrategy.CONSECUTIVE,
        )
        tcf = TemporalConsistencyFilter(cfg)
        tcf.update(make_prediction("YES"))
        tcf.update(make_prediction("NO"))
        result = tcf.update(make_prediction("YES"))
        assert result.label == IntentOutput.NO_DECISION

    def test_majority_vote_passes_majority(self):
        cfg = TemporalConsistencyConfig(
            required_windows=3,
            strategy=ConsistencyStrategy.MAJORITY,
        )
        tcf = TemporalConsistencyFilter(cfg)
        tcf.update(make_prediction("YES"))
        tcf.update(make_prediction("YES"))
        result = tcf.update(make_prediction("NO"))
        assert result.label == "YES"

    def test_no_decision_input_resets_buffer(self):
        cfg = TemporalConsistencyConfig(required_windows=3)
        tcf = TemporalConsistencyFilter(cfg)
        tcf.update(make_prediction("YES"))
        tcf.update(make_prediction("YES"))
        tcf.update(make_no_decision())  # resets buffer
        result = tcf.update(make_prediction("YES"))  # only 1 after reset
        assert result.label == IntentOutput.NO_DECISION

    def test_tbd_config_passes_through(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = TemporalConsistencyConfig(required_windows=None)
            tcf = TemporalConsistencyFilter(cfg)
        pred = make_prediction("HELP")
        result = tcf.update(pred)
        assert result.label == "HELP"

    def test_reset_clears_buffer(self):
        cfg = TemporalConsistencyConfig(required_windows=3)
        tcf = TemporalConsistencyFilter(cfg)
        for _ in range(3):
            tcf.update(make_prediction("YES"))
        tcf.reset()
        result = tcf.update(make_prediction("YES"))
        assert result.label == IntentOutput.NO_DECISION  # buffer cleared
