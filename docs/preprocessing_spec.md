# URF/A-SSR Preprocessing Specification

**Version:** 1.0  
**Date:** 2026-10-08  
**Status:** DRAFT — filter parameters are TBD pending hardware team

---

## Principle

Software preprocessing must not blindly repeat hardware filters. Every step must:
- know what the hardware has already applied (TBD until hardware team confirms);
- have a documented reason for each additional software step;
- be deterministic and reproducible;
- preserve provenance (input version → output version).

---

## EMG Preprocessing Pipeline

```
RAW CSV
 ↓
integrity check
  - check sample count vs expected
  - check for NaN / Inf
  - check sample_id gaps
 ↓
unit conversion (if required)
  - ADC → voltage: factor TBD from hardware
 ↓
hardware-processing verification
  - confirm what hardware has already applied (TBD)
  - do not re-apply hardware filters
 ↓
software QC
  - flatline detection
  - saturation detection
  - high-amplitude transient detection
 ↓
software bandpass filtering (if hardware did not apply)
  - parameters: TBD — pilot-validated, not invented
 ↓
software notch filtering (if hardware did not apply)
  - parameters: TBD — pilot-validated, not invented
 ↓
normalization
  - strategy: TBD — candidate: per-session z-score using baseline
  - must not leak test statistics into train
 ↓
windowing
  - window_length: TBD (pilot-validated)
  - stride: TBD (pilot-validated)
 ↓
feature / input generation
  - see feature_spec section below
```

### EMG Feature Candidates

These are candidates, not guaranteed biomarkers:

| Feature | Description | Status |
|---|---|---|
| RMS | Root mean square amplitude | CANDIDATE |
| MAV | Mean absolute value | CANDIDATE |
| WL | Waveform length | CANDIDATE |
| ZCR | Zero-crossing rate | CANDIDATE |
| Band energy | Power in specific frequency bands | CANDIDATE |
| MFCC | Mel-frequency cepstral coefficients | CANDIDATE |

Feature selection must be validated, not invented.

---

## EEG Preprocessing Pipeline

```
Fp1 / Fp2 raw signals
 ↓
quality checks
  - channel variance check
  - saturation check
  - flatline check
 ↓
validated filtering
  - parameters: TBD — must not be invented
  - hardware pre-filtering: TBD
 ↓
artifact detection / rejection
  - ocular artifact detection (frontal EEG, high susceptibility)
  - movement/EMG artifact detection
  - ICA: OPTIONAL — not mandatory for 2-channel system (PRD §22)
 ↓
normalization
  - strategy: TBD — pilot-validated
 ↓
features / model input
```

### EEG Feature Candidates

These are candidates; confirmation requires pilot validation:

| Feature | Description | Status |
|---|---|---|
| Theta power | ~4–8 Hz | CANDIDATE |
| Alpha power | ~8–13 Hz | CANDIDATE |
| Beta power | ~13–30 Hz | CANDIDATE |
| Relative band powers | Normalized to total | CANDIDATE |
| Fp1/Fp2 asymmetry | Lateralization index | CANDIDATE — requires experimental justification |
| Time-frequency features | TF analysis | CANDIDATE — only if required |
| Temporal features | Statistical moments | CANDIDATE |

**Important:** These are candidate features, not guaranteed biomarkers of any cognitive/affective state.

---

## Normalization

| Parameter | Value | Status |
|---|---|---|
| Strategy | TBD | PILOT-VALIDATED |
| Baseline window | Per-session baseline recording | FROZEN |
| Normalization fitted on | Train set only | FROZEN |
| Applied to test set | Using train statistics only | FROZEN |

---

## Provenance Requirements

Every preprocessing output must record:
- source raw recording path
- preprocessing config version
- software version / git commit
- parameters applied

This allows full reproducibility from raw data.

---

## LMS/NLMS (if hardware uses it)

If hardware applies LMS/NLMS adaptive filtering, document:

| Parameter | Value | Status |
|---|---|---|
| Primary signal | TBD | HARDWARE-TBD |
| Reference signal | TBD | HARDWARE-TBD |
| Artifact estimated | TBD | HARDWARE-TBD |
| Filter length | TBD | HARDWARE-TBD |
| Step size | TBD | HARDWARE-TBD |
| Update mode | TBD | HARDWARE-TBD |
| Output definition | TBD | HARDWARE-TBD |

**Important:** A common EEG reference electrode is not automatically a motion-artifact reference.
If reference is unknown, mark LMS as hardware-unverified.

---

## Version Control

Preprocessing configurations are versioned. Every processed file records the config version.
Changing a preprocessing parameter requires a new version, not an overwrite.
