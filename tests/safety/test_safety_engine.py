"""
Tests for the deterministic safety engine (PRD §37, §49 Safety).
Every rule tested. No unintended trigger. NO_DECISION path explicit.
"""
import pytest

from backend.safety.safety_engine import SafetyEngine, SafetyConfig
from backend.protocol.output_states import IntentOutput


def make_config(enabled=True, trigger_state="STRESS", min_intent=0.8, min_state=0.7, temporal=True):
    return SafetyConfig(
        enabled=enabled,
        trigger_intent="HELP",
        trigger_state=trigger_state,
        min_intent_confidence=min_intent,
        min_state_confidence=min_state,
        temporal_requirement=temporal,
    )


def run_engine(cfg, **kwargs):
    engine = SafetyEngine(cfg)
    defaults = dict(
        intent="HELP",
        state="STRESS",
        intent_confidence=0.9,
        state_confidence=0.8,
        signal_quality="OK",
        temporal_consistency_met=True,
    )
    defaults.update(kwargs)
    return engine.evaluate(**defaults)


class TestSafetyEngine:

    def test_all_conditions_met_gives_action(self):
        cfg = make_config()
        decision = run_engine(cfg)
        assert decision.output == "ACTION"

    def test_engine_disabled_gives_no_decision(self):
        cfg = SafetyConfig(enabled=False)
        engine = SafetyEngine(cfg)
        decision = engine.evaluate(
            intent="HELP", state="STRESS",
            intent_confidence=0.99, state_confidence=0.99,
            signal_quality="OK", temporal_consistency_met=True,
        )
        assert decision.output == IntentOutput.NO_DECISION
        assert decision.rule_applied == "engine_disabled"

    def test_wrong_intent_gives_no_decision(self):
        cfg = make_config()
        decision = run_engine(cfg, intent="YES")
        assert decision.output == IntentOutput.NO_DECISION
        assert decision.rule_applied == "intent_mismatch"

    def test_wrong_state_gives_no_decision(self):
        cfg = make_config()
        decision = run_engine(cfg, state="CALM")
        assert decision.output == IntentOutput.NO_DECISION
        assert decision.rule_applied == "state_mismatch"

    def test_tbd_trigger_state_gives_no_decision(self):
        cfg = make_config(trigger_state=None)
        # Can't instantiate with enabled=True and trigger_state=None — should raise
        with pytest.raises(ValueError, match="TBD"):
            SafetyEngine(cfg)

    def test_low_intent_confidence_gives_no_decision(self):
        cfg = make_config(min_intent=0.9)
        decision = run_engine(cfg, intent_confidence=0.5)
        assert decision.output == IntentOutput.NO_DECISION
        assert decision.rule_applied == "intent_confidence_below_threshold"

    def test_low_state_confidence_gives_no_decision(self):
        cfg = make_config(min_state=0.9)
        decision = run_engine(cfg, state_confidence=0.5)
        assert decision.output == IntentOutput.NO_DECISION
        assert decision.rule_applied == "state_confidence_below_threshold"

    def test_temporal_not_met_gives_no_decision(self):
        cfg = make_config()
        decision = run_engine(cfg, temporal_consistency_met=False)
        assert decision.output == IntentOutput.NO_DECISION
        assert decision.rule_applied == "temporal_consistency_not_met"

    def test_bad_signal_quality_gives_no_decision(self):
        cfg = make_config()
        decision = run_engine(cfg, signal_quality="INVALID")
        assert decision.output == IntentOutput.NO_DECISION
        assert decision.rule_applied == "signal_quality_not_ok"

    def test_degraded_signal_gives_no_decision(self):
        cfg = make_config()
        decision = run_engine(cfg, signal_quality="DEGRADED")
        assert decision.output == IntentOutput.NO_DECISION

    def test_action_logged(self):
        cfg = make_config()
        engine = SafetyEngine(cfg)
        engine.evaluate(
            intent="HELP", state="STRESS",
            intent_confidence=0.9, state_confidence=0.8,
            signal_quality="OK", temporal_consistency_met=True,
        )
        log = engine.audit_log()
        assert len(log) == 1
        assert log[0]["output"] == "ACTION"

    def test_no_decision_also_logged(self):
        cfg = make_config()
        engine = SafetyEngine(cfg)
        engine.evaluate(
            intent="YES", state="STRESS",
            intent_confidence=0.9, state_confidence=0.8,
            signal_quality="OK", temporal_consistency_met=True,
        )
        log = engine.audit_log()
        assert len(log) == 1
        assert log[0]["output"] == IntentOutput.NO_DECISION

    def test_multiple_calls_accumulate_log(self):
        cfg = make_config()
        engine = SafetyEngine(cfg)
        for _ in range(5):
            engine.evaluate(
                intent="HELP", state="STRESS",
                intent_confidence=0.9, state_confidence=0.8,
                signal_quality="OK", temporal_consistency_met=True,
            )
        assert len(engine.audit_log()) == 5

    def test_clear_audit_log(self):
        cfg = make_config()
        engine = SafetyEngine(cfg)
        run_engine(cfg)
        engine.clear_audit_log()
        assert len(engine.audit_log()) == 0

    def test_each_single_failing_condition_blocks_action(self):
        """Independently fail each condition to confirm no partial triggers."""
        failing_cases = [
            {"intent": "NO"},
            {"state": "CALM"},
            {"intent_confidence": 0.1},
            {"state_confidence": 0.1},
            {"temporal_consistency_met": False},
            {"signal_quality": "INVALID"},
        ]
        cfg = make_config()
        for case in failing_cases:
            decision = run_engine(cfg, **case)
            assert decision.output == IntentOutput.NO_DECISION, (
                f"Expected NO_DECISION with {case}, got ACTION"
            )
