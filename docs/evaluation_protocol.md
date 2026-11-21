# URF/A-SSR Evaluation Protocol

**Version:** 1.0  
**Date:** 2026-10-08

---

## Principles

1. Model selection uses validation data only.
2. Test set is locked until all decisions are frozen.
3. Every evaluation is reproducible from config + dataset version + git commit + seed.
4. Subject/session-aware splitting is mandatory — no random window splitting.
5. Do not invent or borrow accuracy numbers from other papers.

---

## Split Strategies

| Strategy | Description | When to Use |
|---|---|---|
| `session_held_out` | Hold out a specific session (e.g. Session 2 of every subject) | Session generalization testing |
| `subject_held_out` | Hold out a specific subject entirely | Cross-subject generalization |
| `LOSO` | Leave-One-Subject-Out | Primary evaluation (where practical) |

All splits recorded in `split_manifest.json` with seed.

---

## EMG Evaluation Checklist

- [ ] Macro-F1 on validation set (model selection)
- [ ] Macro-F1 on test set (final — one-time only)
- [ ] Accuracy
- [ ] Per-class precision: YES, NO, HELP
- [ ] Per-class recall: YES, NO, HELP
- [ ] Confusion matrix
- [ ] **HELP recall** (safety-critical — must be explicitly reported)
- [ ] False HELP activations (false positive rate for HELP)
- [ ] Session generalization (train Sess1, test Sess2)
- [ ] Subject generalization (LOSO)
- [ ] Model inference latency
- [ ] Model size (parameters)

---

## EEG Evaluation Checklist

- [ ] Macro-F1 on validation set
- [ ] Macro-F1 on test set (one-time)
- [ ] Accuracy
- [ ] Per-class precision and recall
- [ ] Confusion matrix
- [ ] Subject/session generalization
- [ ] Rejection/NO_DECISION rate
- [ ] Latency
- [ ] Failure case analysis
- [ ] Comparison with chance level
- [ ] EEG contamination assessment (pilot required)

**If EEG generalization is weak, report it honestly.**  
The EEG branch may conclude with "feasibility not demonstrated" — that is a valid scientific result.

---

## Experiment Manifest

Every experiment records:

```json
{
  "experiment_id": "string",
  "git_commit": "string",
  "dataset_version": "string",
  "hardware_version": "string",
  "firmware_version": "string",
  "preprocessing_version": "string",
  "windowing_version": "string",
  "feature_version": "string",
  "model": "string",
  "seed": 42,
  "split_manifest": "path to split_manifest.json",
  "metrics": {}
}
```

---

## Ablation Studies (Secondary Phase)

### EMG Channel Ablation
- All 4 channels
- Leave-one-channel-out (4 experiments)
- Individual channel analysis where useful

### EEG Channel Ablation
- Fp1 + Fp2
- Fp1 only
- Fp2 only

Ablation is a secondary research phase, not a prerequisite for the first working pipeline.

---

## Robustness Tests (Gate 12)

| Condition | Expected Behavior | Actual Behavior | Classification |
|---|---|---|---|
| New session (Session 2) | Performance maintained | TBD | TBD |
| New subject | Generalization assessed | TBD | TBD |
| Electrode repositioning | Performance stability | TBD | TBD |
| Moderate movement | Signal quality maintained | TBD | TBD |
| Noisy channel | Graceful degradation | TBD | TBD |
| Rest (no command) | NO_DECISION | TBD | TBD |
| Repeated commands | Temporal stability | TBD | TBD |
| Partial channel failure | Fallback behavior | TBD | TBD |
| Temporary transport degradation | NO_DECISION | TBD | TBD |
