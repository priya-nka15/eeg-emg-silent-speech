# URF/A-SSR Quality Control Protocol

**Version:** 1.0  
**Date:** 2026-10-08

---

## Signal-Level QC Checks

| Check | Condition | Action | Status |
|---|---|---|---|
| Missing samples | sample_id gap detected | Mark affected trials INVALID (missing_samples) | IMPLEMENTED |
| Flatline | Channel variance below threshold for N consecutive samples | Flag channel; mark trial INVALID if severe | IMPLEMENTED |
| Saturation | ADC value at or near ±max for N consecutive samples | Flag; mark trial INVALID (saturation) | IMPLEMENTED |
| Invalid values | NaN, Inf, or out-of-range | Flag; mark trial INVALID | IMPLEMENTED |
| Lead-off | status bitfield lead-off bit set | Flag channel; consider INVALID (poor_contact) | IMPLEMENTED |
| High impedance | Impedance above threshold (TBD) | Warn operator; record | HARDWARE-TBD |

---

## Trial-Level QC

- Every trial has a `validity` field: VALID or INVALID.
- Invalid trials record `invalid_reason`.
- Invalid trials are PRESERVED in the dataset with original signal.
- Invalid trials are EXCLUDED from training/evaluation sets.
- Invalid trial counts are reported in the QC report.

---

## Session-Level QC

Per session, compute and report:
- Total trials attempted
- Valid trial count per class
- Invalid trial count per class with reason breakdown
- Dropped sample count
- Per-channel: flatline incidents, saturation incidents, lead-off events
- Measured sample rate vs target

---

## Thresholds

| Parameter | Value | Status |
|---|---|---|
| Flatline variance threshold | TBD | PILOT-VALIDATED |
| Flatline duration threshold | TBD | PILOT-VALIDATED |
| Saturation level (ADC) | TBD | HARDWARE-TBD |
| Saturation duration threshold | TBD | PILOT-VALIDATED |
| High-amplitude transient threshold | TBD | PILOT-VALIDATED |

---

## Automated QC

`backend/qc/signal_qc.py` implements automated checks.  
`backend/qc/trial_validator.py` implements trial-level checks.  
`scripts/validate_recording.py` runs all checks on a recording.

---

## Manual QC

Operator review recommended for:
- Trials near validity threshold
- Sessions with unusually high invalid rate
- First session of each participant (familiarisation effects)
- Any session with operator-recorded anomalies

---

## QC Report

Stored at `Session_NN/qc/qc_report.json`:

```json
{
  "session_id": "...",
  "generated": "ISO 8601 datetime",
  "qc_version": "...",
  "total_samples": 0,
  "dropped_samples": 0,
  "measured_sample_rate_hz": null,
  "per_channel": {
    "EEG_Fp1": {"flatline_events": 0, "saturation_events": 0, "lead_off_events": 0},
    "EEG_Fp2": {"flatline_events": 0, "saturation_events": 0, "lead_off_events": 0},
    "EMG_Submental": {"flatline_events": 0, "saturation_events": 0, "lead_off_events": 0},
    "EMG_Orbicularis": {"flatline_events": 0, "saturation_events": 0, "lead_off_events": 0},
    "EMG_Zygomaticus": {"flatline_events": 0, "saturation_events": 0, "lead_off_events": 0},
    "EMG_Buccinator": {"flatline_events": 0, "saturation_events": 0, "lead_off_events": 0}
  },
  "trials": {
    "total": 0,
    "valid": 0,
    "invalid": 0,
    "invalid_by_reason": {}
  },
  "notes": ""
}
```
