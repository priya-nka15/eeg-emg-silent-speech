# URF/A-SSR Dataset Schema

**Version:** 1.0  
**Date:** 2026-10-08

---

## Data Models

### Subject

```json
{
  "subject_id": "string — pseudonymous ID (e.g. S001), never real name",
  "consent_status": "string — CONSENTED | WITHDRAWN | PENDING",
  "allowed_demographic_metadata": {
    "age_group": "optional string — coarse bracket only",
    "handedness": "optional string"
  }
}
```

**Privacy rules:**
- Use pseudonymous IDs only.
- Do not place participant names in filenames or metadata.
- Do not collect unnecessary PII.

---

### Session

```json
{
  "session_id": "string — e.g. S001_Sess01",
  "subject_id": "string — links to Subject",
  "date": "string — ISO 8601 date",
  "time": "string — ISO 8601 time (local)",
  "hardware_version": "string — TBD until PCB confirmed",
  "firmware_version": "string — TBD until firmware confirmed",
  "electrode_configuration": "string — config version",
  "gain_eeg": "number | TBD",
  "gain_emg": "number | TBD",
  "sampling_rate_hz": 4000,
  "measured_sampling_rate_hz": "number | null — from counter",
  "preprocessing_version": "string",
  "protocol_version": "string",
  "notes": "string | null"
}
```

---

### Trial

```json
{
  "trial_id": "string — e.g. S001_Sess01_T042",
  "subject_id": "string",
  "session_id": "string",
  "intent": "string — YES | NO | HELP | BASELINE | EEG_TASK",
  "state": "string — EEG condition label (TBD until pilot)",
  "start_sample": "integer — inclusive sample_id",
  "end_sample": "integer — inclusive sample_id",
  "start_timestamp": "number — µs",
  "end_timestamp": "number — µs",
  "validity": "string — VALID | INVALID",
  "invalid_reason": "string | null — see invalid reason codes",
  "randomization_position": "integer — position in randomized sequence",
  "notes": "string | null"
}
```

**Invalid reason codes:**
- `vocalization` — participant vocalized instead of silent articulation
- `wrong_command` — performed different command than cued
- `movement` — excessive movement artifact
- `swallowing` — swallowing artifact
- `coughing` — coughing artifact
- `electrode_detachment` — electrode came off
- `poor_contact` — high impedance / poor contact
- `saturation` — signal saturated ADC
- `missing_samples` — sample counter gaps detected
- `transmission_failure` — transport error
- `timing_error` — event marker timing problem
- `interruption` — trial interrupted externally
- `protocol_violation` — other protocol violation

**Rule:** Do not silently delete invalid trials. Preserve original signal.

---

### Sample

```json
{
  "sample_id": "integer — monotonically increasing uint32",
  "timestamp": "integer — microseconds since session start",
  "EEG_Fp1": "integer — raw 24-bit ADC value",
  "EEG_Fp2": "integer — raw 24-bit ADC value",
  "EMG_Submental": "integer — raw 24-bit ADC value",
  "EMG_Orbicularis": "integer — raw 24-bit ADC value",
  "EMG_Zygomaticus": "integer — raw 24-bit ADC value",
  "EMG_Buccinator": "integer — raw 24-bit ADC value",
  "event_marker": "integer — 0-255 event code",
  "status": "integer — status bitfield"
}
```

---

## Raw Data Storage Structure

```
URF_DATA/
  Subject_01/          # pseudonymous ID
    Session_01/
      raw/             # IMMUTABLE — never overwrite
        recording.csv  # sample-level data
        recording.bin  # optional binary format
      metadata/
        session.json
        trials.json
        subject.json
        randomization_sequence.json
      events/
        events.json    # event log with sample_id references
      qc/
        qc_report.json
        trial_validity.json
      processed/
        <preprocessing_version>/
          emg_processed.npy
          eeg_processed.npy
          processing_manifest.json
      features/
        <feature_version>/
          emg_features.npy
          eeg_features.npy
          feature_manifest.json
    Session_02/
      ...
  Subject_02/
    ...
```

---

## Dataset Manifest

Stored at `data/manifests/dataset_manifest.json`:

```json
{
  "manifest_version": "string",
  "dataset_id": "string",
  "created": "ISO 8601 datetime",
  "subjects": ["S001", "S002", "..."],
  "sessions_per_subject": {},
  "total_valid_trials": {},
  "valid_trials_per_class": {"YES": 0, "NO": 0, "HELP": 0},
  "invalid_trials": 0,
  "preprocessing_version": "string",
  "protocol_version": "string",
  "notes": "string"
}
```

---

## Split Manifest

Stored at `data/manifests/split_manifest.json`:

```json
{
  "manifest_version": "string",
  "dataset_version": "string",
  "split_strategy": "subject_held_out | session_held_out | LOSO",
  "seed": 42,
  "train": {"subjects": [], "sessions": []},
  "validation": {"subjects": [], "sessions": []},
  "test": {"subjects": [], "sessions": []},
  "test_locked": true,
  "notes": "string"
}
```

**Rule:** Never randomly split closely related windows from the same recording into train/test.
