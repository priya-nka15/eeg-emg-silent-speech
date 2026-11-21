"""
URF/A-SSR Safety Engine
PRD §37, §42 — Deterministic software safety engine, separate from ML.
Every decision is logged and auditable.
Rules are unit-tested. No unintended triggers. NO_DECISION path tested.

Safety engine is DISABLED by default.
trigger_state is TBD until EEG labels are frozen.
All thresholds are TBD until validated.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from backend.fusion.decision_fusion import FusionResult
from backend.protocol.output_states import IntentOutput

logger = logging.getLogger(__name__)


@dataclass
class SafetyConfig:
    """
    PRD §37 safety configuration.
    All TBD fields must be validated before enabling.
    """
    enabled: bool = False                       # MUST remain False until validated
    trigger_intent: str = "HELP"               # FROZEN (PRD §37)
    trigger_state: Optional[str] = None        # TBD — EEG label not frozen
    min_intent_confidence: Optional[float] = None  # PILOT-VALIDATED
    min_state_confidence: Optional[float] = None   # PILOT-VALIDATED
    temporal_requirement: Optional[bool] = None    # True = consistency met

    def validate(self) -> List[str]:
        issues = []
        if self.enabled:
            if self.trigger_state is None:
                issues.append("trigger_state is TBD — cannot enable safety engine")
            if self.min_intent_confidence is None:
                issues.append("min_intent_confidence is TBD")
            if self.min_state_confidence is None:
                issues.append("min_state_confidence is TBD")
            if self.temporal_requirement is None:
                issues.append("temporal_requirement is TBD")
        return issues

    @classmethod
    def from_yaml(cls, path: Path) -> "SafetyConfig":
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        cfg = raw.get("safety", {})
        return cls(
            enabled=cfg.get("enabled", False),
            trigger_intent=cfg.get("trigger_intent", "HELP"),
            trigger_state=None if cfg.get("trigger_state") == "TBD" else cfg.get("trigger_state"),
            min_intent_confidence=None if cfg.get("min_intent_confidence") == "TBD" else cfg.get("min_intent_confidence"),
            min_state_confidence=None if cfg.get("min_state_confidence") == "TBD" else cfg.get("min_state_confidence"),
            temporal_requirement=None if cfg.get("temporal_requirement") == "TBD" else cfg.get("temporal_requirement"),
        )


@dataclass
class SafetyDecision:
    output: str                    # "ACTION" | "NO_DECISION"
    rule_applied: str
    inputs: dict = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp or datetime.utcnow().isoformat(),
            "output": self.output,
            "rule_applied": self.rule_applied,
            "inputs": self.inputs,
        }


class SafetyEngine:
    """
    Deterministic safety engine.

    PRD §37 conceptual rule:
        IF intent == HELP
        AND state == trigger_state
        AND intent_confidence >= min_intent_confidence
        AND state_confidence >= min_state_confidence
        AND temporal_consistency_met == True
        THEN output = ACTION
        ELSE output = NO_DECISION

    Safety engine is DISABLED by default.
    Do not enable until all TBD values are resolved and validated.
    """

    def __init__(self, config: SafetyConfig, audit_log_path: Optional[Path] = None) -> None:
        self.config = config
        self.audit_log_path = audit_log_path
        self._audit_log: List[dict] = []

        if config.enabled:
            issues = config.validate()
            if issues:
                raise ValueError(
                    f"SafetyEngine cannot be enabled with TBD values: {issues}"
                )

    def evaluate(
        self,
        intent: str,
        state: Optional[str],
        intent_confidence: float,
        state_confidence: float,
        signal_quality: str,
        temporal_consistency_met: bool,
    ) -> SafetyDecision:
        """
        Apply the deterministic safety rule.

        Args:
            intent: EMG intent output (e.g. "HELP")
            state: EEG state output (TBD label or None)
            intent_confidence: EMG model confidence [0, 1]
            state_confidence: EEG model confidence [0, 1]
            signal_quality: "OK" | "DEGRADED" | "INVALID"
            temporal_consistency_met: True if temporal filter passed

        Returns:
            SafetyDecision with output "ACTION" or "NO_DECISION"
        """
        inputs = {
            "intent": intent,
            "state": state,
            "intent_confidence": intent_confidence,
            "state_confidence": state_confidence,
            "signal_quality": signal_quality,
            "temporal_consistency_met": temporal_consistency_met,
            "engine_enabled": self.config.enabled,
        }

        # Rule 0: Engine disabled
        if not self.config.enabled:
            decision = SafetyDecision(
                output=IntentOutput.NO_DECISION,
                rule_applied="engine_disabled",
                inputs=inputs,
                timestamp=datetime.utcnow().isoformat(),
            )
            self._log(decision)
            return decision

        # Rule 1: Intent must match trigger
        if intent != self.config.trigger_intent:
            return self._no_decision("intent_mismatch", inputs)

        # Rule 2: State must match trigger_state (TBD check)
        if self.config.trigger_state is None:
            return self._no_decision("trigger_state_tbd", inputs)
        if state != self.config.trigger_state:
            return self._no_decision("state_mismatch", inputs)

        # Rule 3: Intent confidence
        if (
            self.config.min_intent_confidence is not None
            and intent_confidence < self.config.min_intent_confidence
        ):
            return self._no_decision("intent_confidence_below_threshold", inputs)

        # Rule 4: State confidence
        if (
            self.config.min_state_confidence is not None
            and state_confidence < self.config.min_state_confidence
        ):
            return self._no_decision("state_confidence_below_threshold", inputs)

        # Rule 5: Temporal consistency
        if not temporal_consistency_met:
            return self._no_decision("temporal_consistency_not_met", inputs)

        # Rule 6: Signal quality
        if signal_quality not in ("OK",):
            return self._no_decision("signal_quality_not_ok", inputs)

        # All rules passed → ACTION
        decision = SafetyDecision(
            output="ACTION",
            rule_applied="all_rules_passed",
            inputs=inputs,
            timestamp=datetime.utcnow().isoformat(),
        )
        self._log(decision)
        logger.info("SafetyEngine: ACTION triggered. %s", inputs)
        return decision

    def _no_decision(self, rule: str, inputs: dict) -> SafetyDecision:
        decision = SafetyDecision(
            output=IntentOutput.NO_DECISION,
            rule_applied=rule,
            inputs=inputs,
            timestamp=datetime.utcnow().isoformat(),
        )
        self._log(decision)
        return decision

    def _log(self, decision: SafetyDecision) -> None:
        entry = decision.to_dict()
        self._audit_log.append(entry)
        if self.audit_log_path:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def audit_log(self) -> List[dict]:
        return list(self._audit_log)

    def clear_audit_log(self) -> None:
        self._audit_log.clear()
