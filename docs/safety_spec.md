# URF/A-SSR Safety Specification

**Version:** 1.0  
**Date:** 2026-10-08

---

## Scope

This document covers:
1. Human-use safety (pre-study requirements)
2. Software safety engine (runtime)

---

## 1. Human-Use Safety

### Requirements

- The custom PCB and electrode setup must undergo appropriate electrical and experimental safety review before any human participant data collection.
- The hardware team / institution is responsible for this review.
- `human_use_ready` flag in `configs/hardware.yaml` must be `false` until the review is completed and documented.
- Software must check this flag before allowing Live mode.
- This system is a **research prototype**. It must not be described as clinically validated or as a certified medical device.

### Safety Review Documentation

When the review is completed, record in `docs/hardware_validation_report_v1.md`:
- Review type and standard applied
- Reviewing body / institution
- Date
- Outcome
- Any conditions / restrictions

Then set `human_use_ready: true` in `configs/hardware.yaml`.

---

## 2. Software Safety Engine

The safety engine is a deterministic software component, entirely separate from ML models.

### Purpose

Provide a rule-based final gate before issuing safety-critical actions (e.g. triggering a HELP response).

### Design Constraints

- **Deterministic:** Same inputs → same output, every time.
- **Auditable:** Every decision is logged with full input state.
- **Testable:** Every rule has explicit unit tests. No unintended trigger. NO_DECISION path explicitly tested.
- **No ML in the decision path:** The safety engine reads ML outputs but applies only rule-based logic.
- **Disabled by default:** `enabled: false`. Must be explicitly enabled with validated thresholds.

### Current Configuration

```yaml
safety:
  enabled: false           # must remain false until thresholds validated
  trigger_intent: HELP
  trigger_state: TBD       # EEG label not frozen
  min_intent_confidence: TBD
  min_state_confidence: TBD
  temporal_requirement: TBD
```

### Trigger Logic

```
IF enabled == true
AND intent == trigger_intent (HELP)
AND state == trigger_state (TBD)
AND intent_confidence >= min_intent_confidence (TBD)
AND state_confidence >= min_state_confidence (TBD)
AND temporal_consistency_met == true
THEN output = ACTION
ELSE output = NO_DECISION
```

### NO_DECISION Conditions

The safety engine outputs NO_DECISION when:
- `enabled == false`
- `intent != HELP`
- `state != trigger_state` (or trigger_state is TBD)
- `intent_confidence < min_intent_confidence`
- `state_confidence < min_state_confidence`
- `temporal_consistency_met == false`
- Any required field is missing or invalid

### Audit Log

Every evaluation by the safety engine produces an audit entry:
```json
{
  "timestamp": "ISO 8601",
  "input": {
    "intent": "...",
    "state": "...",
    "intent_confidence": 0.0,
    "state_confidence": 0.0,
    "signal_quality": "...",
    "temporal_consistency": false
  },
  "output": "NO_DECISION | ACTION",
  "rule_applied": "...",
  "enabled": false
}
```

### Unit Test Requirements

Every safety engine rule must have:
- A test that triggers the rule correctly
- A test that confirms it does NOT trigger when any single condition fails
- A test for the NO_DECISION path
- A test with TBD/None thresholds (must always produce NO_DECISION)
