# URF/A-SSR Implementation Status

**Last updated:** 2026-10-08  
**Repository state:** Greenfield — initial implementation

---

## Gate Progress

| Gate | Description | Status | Notes |
|---|---|---|---|
| Gate 1 | Repository / specification | IN-PROGRESS | Docs, configs, traceability created |
| Gate 2 | Hardware interface | IN-PROGRESS | Interface implemented; hardware values TBD |
| Gate 3 | Hardware validation | NOT-STARTED | Requires physical hardware |
| Gate 4 | Recorder | PARTIALLY-IMPLEMENTED | Session/trial/event/metadata/QC complete; replay runs pipeline; live acquisition is a stub pending hardware |
| Gate 5 | Pilot | NOT-STARTED | Requires hardware + participants |
| Gate 6 | Protocol freeze | NOT-STARTED | Blocked on pilot |
| Gate 7 | Main dataset | NOT-STARTED | Blocked on pilot + protocol freeze |
| Gate 8 | EMG ML | IMPLEMENTED | Models coded; training requires dataset |
| Gate 9 | EEG ML | IMPLEMENTED | Models coded; EEG labels not frozen |
| Gate 10 | Fusion | IMPLEMENTED | Decision-level fusion, confidence, temporal, safety |
| Gate 11 | Real time | IMPLEMENTED | Pipeline coded; live mode requires hardware |
| Gate 12 | Robustness / final evaluation | NOT-STARTED | Requires complete dataset + trained models |

---

## Phase Status

### Phase A — Repository + Specification
- [x] Repository directory structure created
- [x] `docs/decision_log.md`
- [x] `docs/requirements_traceability.md`
- [x] `docs/implementation_status.md` (this file)
- [x] `docs/architecture.md`
- [x] `docs/hardware_interface_spec_v1.md`
- [x] `docs/electrode_map.md`
- [x] `docs/preprocessing_spec.md`
- [x] `docs/dataset_schema.md`
- [x] `docs/data_collection_protocol.md`
- [x] `docs/qc_protocol.md`
- [x] `docs/model_development.md`
- [x] `docs/evaluation_protocol.md`
- [x] `docs/fusion_spec.md`
- [x] `docs/safety_spec.md`
- [x] `docs/realtime_spec.md`
- [x] `docs/limitations.md`
- [x] `docs/reproducibility.md`
- [x] `docs/validation_report.md` (template)

### Phase B — Acquisition
- [x] Transport frame parser (`backend/acquisition/frame_parser.py`)
- [x] Event system (`backend/protocol/events.py`)
- [x] Raw storage (`backend/datasets/raw_store.py`)
- [x] Session/trial recorder (`backend/protocol/session_manager.py`)
- [x] Mock data source (`backend/realtime/mock_source.py`)

### Phase C — Hardware Validation
- [x] Hardware validator tools (`backend/acquisition/hardware_validator.py`)
- [x] Validation script (`scripts/validate_hardware.py`)
- [ ] Validation report — **BLOCKED: requires physical hardware**

### Phase D — Preprocessing
- [x] QC module (`backend/qc/`)
- [x] EEG preprocessing pipeline (`backend/preprocessing/eeg/pipeline.py`)
- [x] EMG preprocessing pipeline (`backend/preprocessing/emg/pipeline.py`)
- [x] Windowing (`backend/windowing/windower.py`)
- [x] Feature extraction EMG (`backend/features/emg_features.py`)
- [x] Feature extraction EEG (`backend/features/eeg_features.py`)

### Phase E — Protocol Freeze
- [ ] **BLOCKED: requires pilot data**

### Phase F — Main Dataset
- [x] Dataset ingestion and validation (`backend/datasets/`)
- [x] Split manifests (`backend/datasets/splitter.py`)
- [ ] Actual data collection — **BLOCKED: requires hardware + protocol freeze**

### Phase G — EMG ML
- [x] SVM (`backend/models/emg/svm_model.py`)
- [x] GRU (`backend/models/emg/gru_model.py`)
- [x] CNN-GRU (`backend/models/emg/cnn_gru_model.py`)
- [x] Evaluation (`backend/evaluation/`)
- [x] Model registry (`backend/models/model_registry.py`)
- [ ] Training/comparison — **BLOCKED: requires dataset**

### Phase H — EEG ML
- [x] SVM (`backend/models/eeg/svm_model.py`)
- [x] 1D-CNN (`backend/models/eeg/cnn1d_model.py`)
- [ ] EEG labels — **BLOCKED: requires pilot (DL-006)**
- [ ] Training/comparison — **BLOCKED: requires dataset + labels**

### Phase I — Fusion
- [x] Decision-level fusion (`backend/fusion/decision_fusion.py`)
- [x] Confidence / rejection (`backend/fusion/confidence.py`)
- [x] Temporal consistency (`backend/fusion/temporal_consistency.py`)
- [x] Safety engine (`backend/safety/safety_engine.py`)

### Phase J — Real Time
- [x] Real-time pipeline (`backend/realtime/pipeline.py`)
- [x] Mock mode (all outputs labelled MOCK_DATA, `is_mock=True`)
- [x] Replay mode (`backend/realtime/replay.py`) — calls process_window() per sample
- [x] Latency monitor (`backend/realtime/latency_monitor.py`)
- [x] LIVE safety gate — fail-closed; absolute path resolution; refuses on missing hardware.yaml, human_use_ready=false, missing/TBD final_pipeline.yaml
- [ ] Live hardware streaming — **BLOCKED: requires hardware + human_use_ready=true**
- [ ] Full UI — **BLOCKED: requires hardware**

### Phase K — Robustness
- [ ] **NOT-STARTED: requires trained models + hardware**

### Phase L — Research Package
- [ ] **NOT-STARTED**

---

## Known Blockers

| Blocker | Type | Unblocked By |
|---|---|---|
| AFE exact part number | HARDWARE-TBD | Hardware team |
| MCU exact variant + firmware | HARDWARE-TBD | Hardware team |
| Measured sample rate | HARDWARE-TBD | Gate H2 |
| IN+/IN− wiring per channel | HARDWARE-TBD | PCB schematic |
| All hardware filter parameters | HARDWARE-TBD | Hardware team |
| LMS reference signal | HARDWARE-TBD | Hardware team |
| Gain (EEG, EMG) | HARDWARE-TBD | Hardware team |
| PCB revision | HARDWARE-TBD | Hardware team |
| human_use_ready | SAFETY-REVIEW | Institutional review |
| EEG task/labels | PILOT-VALIDATED | EEG pilot (Gate 5) |
| Trial timing (exact durations) | PILOT-VALIDATED | EMG pilot (Gate 5) |
| Window size/stride | PILOT-VALIDATED | Pilot analysis (Gate 5→6) |
| Confidence thresholds | PILOT-VALIDATED | Validation phase (Gate 8/9) |
| Temporal consistency windows | PILOT-VALIDATED | Validation phase (Gate 10) |
| Safety trigger_state | PILOT-VALIDATED | EEG label freeze + pilot |

---

## TBD Registry

All TBD values are recorded here for tracking.

| Config File | Field | Status | Notes |
|---|---|---|---|
| `configs/hardware.yaml` | `afe_part_number` | HARDWARE-TBD | Exact ADS1298-class variant |
| `configs/hardware.yaml` | `mcu_module` | HARDWARE-TBD | Exact ESP32-S3 module |
| `configs/hardware.yaml` | `firmware_revision` | HARDWARE-TBD | — |
| `configs/hardware.yaml` | `pcb_revision` | HARDWARE-TBD | — |
| `configs/hardware.yaml` | `gain_eeg` | HARDWARE-TBD | — |
| `configs/hardware.yaml` | `gain_emg` | HARDWARE-TBD | — |
| `configs/hardware.yaml` | `transport_protocol` | HARDWARE-TBD | — |
| `configs/hardware.yaml` | `frame_format_version` | HARDWARE-TBD | — |
| `configs/hardware_preprocessing.yaml` | `bandpass.*` | HARDWARE-TBD | All fields |
| `configs/hardware_preprocessing.yaml` | `notch.*` | HARDWARE-TBD | All fields |
| `configs/hardware_preprocessing.yaml` | `lms.*` | HARDWARE-TBD | All fields |
| `configs/hardware_preprocessing.yaml` | `wavelet.*` | HARDWARE-TBD | All fields |
| `configs/electrodes.yaml` | `ch*/in_plus`, `ch*/in_minus` | HARDWARE-TBD | All 6 channels |
| `configs/windowing.yaml` | `window_length_samples` | PILOT-VALIDATED | — |
| `configs/windowing.yaml` | `stride_samples` | PILOT-VALIDATED | — |
| `configs/windowing.yaml` | `alignment` | PILOT-VALIDATED | — |
| `configs/windowing.yaml` | `label_policy` | PILOT-VALIDATED | — |
| `configs/eeg_experiment.yaml` | `final_conditions` | PILOT-VALIDATED | Not frozen before pilot |
| `configs/safety.yaml` | `trigger_state` | PILOT-VALIDATED | EEG label not frozen |
| `configs/safety.yaml` | `min_intent_confidence` | PILOT-VALIDATED | — |
| `configs/safety.yaml` | `min_state_confidence` | PILOT-VALIDATED | — |
| `configs/safety.yaml` | `temporal_requirement` | PILOT-VALIDATED | — |
| `configs/final_pipeline.yaml` | `hardware_version` | HARDWARE-TBD | — |
| `configs/final_pipeline.yaml` | `preprocessing_version` | PILOT-VALIDATED | — |
| `configs/final_pipeline.yaml` | `emg.model_id` | PILOT-VALIDATED | After Gate 8 |
| `configs/final_pipeline.yaml` | `eeg.model_id` | PILOT-VALIDATED | After Gate 9 |
