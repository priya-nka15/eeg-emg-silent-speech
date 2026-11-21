# URF/A-SSR Model Development Guide

**Version:** 1.0  
**Date:** 2026-10-08

---

## Frozen Model Set

### EMG Models

| Model | Architecture | Input | Status |
|---|---|---|---|
| EMG-SVM | Support Vector Machine | Engineered feature vector | FROZEN |
| EMG-GRU | Gated Recurrent Unit | Sequence (time × channels/features) | FROZEN |
| EMG-CNN-GRU | CNN + GRU hybrid | Sequence (time × channels/features) | FROZEN |

### EEG Models

| Model | Architecture | Input | Status |
|---|---|---|---|
| EEG-SVM | Support Vector Machine | Engineered feature vector | FROZEN |
| EEG-1D-CNN | 1D Convolutional Neural Network | Windowed 2-channel temporal | FROZEN |

**The following are EXCLUDED from the first implementation:**  
Transformer, Conformer, LSTM, TCN, XGBoost, Random Forest, LDA.  
Adding any of these requires a separate documented research decision.

---

## Training Pipeline

```
1. Load preprocessed, windowed data
2. Apply subject/session-aware split (from split_manifest.json)
3. [Optional] Augment TRAIN set only
4. Train model on TRAIN set
5. Evaluate on VALIDATION set
6. Tune hyperparameters on VALIDATION set only
7. Record experiment_manifest.json
8. (After freeze) Evaluate ONCE on TEST set
```

---

## Data Splitting

**Rule:** Never randomly split windows from the same recording into train/test.

Supported strategies:
- `session_held_out` — hold out full sessions
- `subject_held_out` — hold out full subjects
- `LOSO` — Leave-One-Subject-Out (preferred where practical)

Split manifest (`split_manifest.json`) records:
- dataset version
- train/validation/test subjects and sessions
- seed
- split strategy

---

## Augmentation

If augmentation is used:
1. Split first
2. Augment TRAIN only
3. Never augment validation or test sets

---

## Metrics

### EMG (Primary)
- **Macro-F1** (primary selection metric)
- Accuracy
- Precision per class (YES, NO, HELP)
- Recall per class
- Confusion matrix
- HELP recall (safety-critical)
- HELP false negatives
- False HELP activations
- Latency (where relevant)
- Model size (where relevant)

### EEG (Primary)
- **Macro-F1**
- Accuracy
- Per-class precision and recall
- Confusion matrix
- Subject/session generalization
- Rejection rate
- Latency
- Failure cases

---

## Model Selection

Selection uses VALIDATION data only.

Priority order:
1. Macro-F1
2. Safety-relevant recall (HELP recall for EMG)
3. Session/subject robustness
4. Latency
5. Model complexity

Exact tie-break rules must be frozen before final model selection (Gate 8/9).

After selection: ONE EMG MODEL + ONE EEG MODEL are frozen for the final system.

---

## Test Set Policy

The test set is locked until ALL of the following are frozen:
- Preprocessing configuration
- Feature representation
- Model architecture
- Hyperparameters
- Thresholds
- Fusion configuration

Test set performance is a final evaluation, not a tuning signal.

---

## Model Artifacts

Every trained model must include:
```
model artifact         (e.g. .pkl, .pt, .h5)
config.yaml            (exact hyperparameters)
label_map.json         (class → index mapping)
preprocessing_version.txt
dataset_version.txt
experiment_manifest.json
metrics.json
```

A model cannot be loaded in final mode without version compatibility check.

---

## Model Registry

Statuses: EXPERIMENTAL | VALIDATED | SELECTED | RETIRED  
Only `SELECTED` models can be used by the real-time final mode.

---

## Important Notes

- Do not use test-set performance to select a model.
- Do not copy another paper's accuracy as a URF result.
- If EEG generalization is weak, report it honestly.
- EEG labels are not frozen until after the pilot.
- HELP recall is safety-critical and must be explicitly reported.
