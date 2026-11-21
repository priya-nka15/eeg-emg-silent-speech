# URF/A-SSR Reproducibility Guide

**Version:** 1.0  
**Date:** 2026-10-08

---

## Principle

Every experiment must be reproducible from its configuration, dataset version, code commit, and seed.

---

## Reproducibility Requirements

### For Every Training Experiment

Record in `experiment_manifest.json`:
- `experiment_id` — unique identifier
- `git_commit` — exact commit hash
- `dataset_version` — version of the dataset used
- `hardware_version` — hardware configuration version
- `firmware_version` — firmware revision
- `preprocessing_version` — preprocessing config version
- `windowing_version` — windowing config version
- `feature_version` — feature extraction version
- `model` — model architecture name
- `seed` — all random seeds (Python, numpy, PyTorch/sklearn)
- `split_manifest` — path to split manifest used
- `metrics` — all computed metrics

### For Every Raw Session

The raw recording plus its metadata must be sufficient to reproduce all downstream outputs.

### For Replay

Running replay on a raw session with the same config and commit must produce identical output to the original live run.

---

## Seed Management

Use explicit seeds everywhere:
- Python `random.seed()`
- `numpy.random.seed()`
- `torch.manual_seed()` (if using PyTorch)
- sklearn estimators: `random_state=seed`
- Data splitter: `seed` parameter

---

## Version Tracking

| Artifact | Version field | How versioned |
|---|---|---|
| Hardware config | `hardware_version` | Manual bump in `configs/hardware.yaml` |
| Firmware | `firmware_revision` | Hardware team provides |
| Preprocessing | `preprocessing_version` | Manual bump in `configs/preprocessing.yaml` |
| Windowing | `windowing_version` | Manual bump in `configs/windowing.yaml` |
| Features | `feature_version` | Manual bump in `configs/features.yaml` |
| Dataset | `dataset_version` | Recorded in dataset manifest |
| Model | `model_id` | Registered in model registry |
| Code | `git_commit` | Automatic from `git rev-parse HEAD` |

---

## Preventing Leakage

- Never randomly split closely related windows from the same recording into train/test.
- Normalisation statistics (mean, std) are computed on TRAIN only and applied to validation/test.
- No information from validation or test sets influences preprocessing, feature selection, or hyperparameter tuning.
- The test set is locked until all decisions are frozen (see `docs/model_development.md`).

---

## Replay Verification

After training a model, replay validation must confirm:
1. Load raw recording.
2. Apply preprocessing with recorded `preprocessing_version`.
3. Apply windowing with recorded `windowing_version`.
4. Apply feature extraction with recorded `feature_version`.
5. Run inference with the registered model.
6. Compare output to original live run output.
7. Any discrepancy must be investigated and resolved before the model can be used.
