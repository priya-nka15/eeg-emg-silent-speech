# URF/A-SSR Fusion Specification

**Version:** 1.0  
**Date:** 2026-10-08

---

## First Implementation: Decision-Level Fusion

The first multimodal implementation uses decision-level fusion.  
Raw EEG+EMG concatenation into a large multimodal neural network is prohibited for the first implementation.

### Inputs to Fusion Layer

```
EMG intent probabilities    [P(YES), P(NO), P(HELP)]
EEG state probabilities     [TBD — labels not frozen]
signal quality              per channel
confidence                  per modality
```

### Output

```
YES | NO | HELP | NO_DECISION
```

---

## Confidence / Rejection

Every model exposes:
- predicted label
- probability / confidence representation
- quality state

The system returns `NO_DECISION` when:
- Confidence is too low (thresholds TBD from validation)
- Signal quality is inadequate
- A channel is disconnected
- Samples are missing
- Predictions are inconsistent across temporal window
- A required modality is unavailable

**Thresholds must be validated, not invented.**

---

## Missing Modality Handling

The fusion layer must handle:
- EEG unavailable (channel fault, poor contact) → fall back to EMG only
- EMG unavailable → fall back to EEG only (intent uncertain)
- Both unavailable → NO_DECISION
- Partial channel failure → degrade gracefully

---

## Temporal Consistency

Do not trigger actions from one noisy window.

Configurable temporal consistency layer supports:
- Consecutive compatible predictions
- Probability smoothing
- Majority vote
- Hysteresis

The exact number of windows and thresholds must be selected from validation data, not invented.

---

## Safety Engine

The safety engine is deterministic software, separate from ML.

### Inputs
```
intent
state
intent_confidence
state_confidence
signal_quality
temporal_consistency
```

### Outputs
```
ACTION | NO_DECISION
```

### Conceptual Rule
```
IF intent == HELP
AND relevant safety-critical state detected     [trigger_state: TBD]
AND confidence requirements pass                [thresholds: TBD]
AND temporal consistency passes                 [requirement: TBD]
THEN trigger predefined help/safety action
```

**trigger_state is TBD** — final EEG label is not frozen.  
Safety engine is **disabled by default** (`enabled: false`).

### Safety Engine Configuration (`configs/safety.yaml`)

```yaml
safety:
  enabled: false
  trigger_intent: HELP
  trigger_state: TBD
  min_intent_confidence: TBD
  min_state_confidence: TBD
  temporal_requirement: TBD
```

### Safety Engine Requirements
- Deterministic (same inputs → same output, always)
- Auditable (every decision logged)
- Unit-tested (every rule, no unintended trigger, NO_DECISION path)
- No ML inference in the safety decision path
