# URF/A-SSR Requirements Traceability Matrix

Every requirement from the PRD maps to an implementation reference, test, and current status.

**Status values:** IMPLEMENTED | IN-PROGRESS | BLOCKED-HARDWARE | BLOCKED-PILOT | PLANNED | NOT-STARTED

---

| Req ID | Description | PRD Source | Status | Implementation | Test | Evidence / Notes |
|---|---|---|---|---|---|---|
| REQ-001 | Six simultaneous biopotential channels: CH1=EEG Fp1, CH2=EEG Fp2, CH3=EMG Submental, CH4=EMG Orbicularis, CH5=EMG Zygomaticus, CH6=EMG Buccinator | §5.1 | IMPLEMENTED | `configs/hardware.yaml`, `backend/acquisition/frame_parser.py` | `tests/acquisition/test_frame_parser.py` | FROZEN channel config |
| REQ-002 | Differential electrode topology: 6 diff channels, 12 signal electrodes, 1 EEG ref, 1 EMG ref = 14 total electrodes | §5.2 | IMPLEMENTED | `docs/electrode_map.md`, `configs/electrodes.yaml` | — | IN+/IN− wiring TBD pending schematic |
| REQ-003 | AFE: ADS1298-class 8-channel biopotential AFE | §6.1 | BLOCKED-HARDWARE | `configs/hardware.yaml` (afe_part_number: TBD) | `tests/acquisition/test_frame_parser.py` | Exact part TBD |
| REQ-004 | MCU: ESP32-S3 | §6.2 | BLOCKED-HARDWARE | `configs/hardware.yaml` (mcu_module: TBD) | — | Exact variant TBD |
| REQ-005 | Sampling rate target: 4,000 SPS/channel | §6.3 | IMPLEMENTED | `configs/hardware.yaml` | `tests/acquisition/test_frame_parser.py` | Measured rate requires hardware (H2) |
| REQ-006 | Hardware-software contract document | §7 | IMPLEMENTED | `docs/hardware_interface_spec_v1.md`, `configs/hardware.yaml` | — | TBD fields for all hardware-dependent values |
| REQ-007 | Hardware preprocessing config (bandpass, notch, LMS, wavelet) with TBD fields | §8 | IMPLEMENTED | `configs/hardware_preprocessing.yaml` | — | All filter values TBD pending hardware team |
| REQ-008 | Hardware validation gates H1–H8 | §9 | IMPLEMENTED | `backend/acquisition/hardware_validator.py`, `scripts/validate_hardware.py` | `tests/acquisition/test_hardware_validator.py` | H3 returns NOT_STARTED — physical simultaneity requires hardware |
| REQ-009 | `human_use_ready` flag = false until safety review | §10 | IMPLEMENTED | `configs/hardware.yaml` | `tests/acquisition/test_hardware_validator.py` | OPEN — requires institutional review |
| REQ-010 | Repository architecture per §11 | §11 | IMPLEMENTED | All directories created | — | — |
| REQ-011 | Subject data model (subject_id, consent_status, demographics) | §12 | IMPLEMENTED | `backend/datasets/schema.py` | `tests/dataset/test_schema.py` | Pseudonymous IDs only |
| REQ-012 | Session data model | §12 | IMPLEMENTED | `backend/datasets/schema.py` | `tests/dataset/test_schema.py` | — |
| REQ-013 | Trial data model with validity and invalid_reason | §12, §15 | IMPLEMENTED | `backend/datasets/schema.py` | `tests/dataset/test_schema.py` | — |
| REQ-014 | Sample data model (sample_id, timestamp, 6 channels, event_marker, status) | §12 | IMPLEMENTED | `backend/datasets/schema.py` | `tests/dataset/test_schema.py` | — |
| REQ-015 | Raw data immutability policy | §13 | IMPLEMENTED | `backend/datasets/raw_store.py`, README | `tests/dataset/test_raw_store.py` | Never overwrite raw/ |
| REQ-016 | Structured raw data storage: URF_DATA/Subject_NN/Session_NN/raw|metadata|events|qc|processed|features | §13 | IMPLEMENTED | `backend/datasets/raw_store.py` | `tests/dataset/test_raw_store.py` | — |
| REQ-017 | Event system with all required event types | §14 | IMPLEMENTED | `backend/protocol/events.py` | `tests/acquisition/test_events.py` | — |
| REQ-018 | Events synchronized to signal samples | §14 | IMPLEMENTED | `backend/acquisition/frame_parser.py` | `tests/acquisition/test_frame_parser.py` | — |
| REQ-019 | Trial validity tracking (VALID/INVALID with reason) | §15 | IMPLEMENTED | `backend/datasets/schema.py`, `backend/qc/trial_validator.py` | `tests/qc/test_trial_validator.py` | Never silently delete |
| REQ-020 | Pilot: 2–3 participants, ~20–30 trials/class/session | §16 | PLANNED | `backend/protocol/session_manager.py` | — | Requires hardware and participants |
| REQ-021 | Main dataset target: 12 participants × 2 sessions × 3 commands × ≥100 valid trials | §17 | PLANNED | `backend/datasets/` | `tests/dataset/` | Requires completed pilot |
| REQ-022 | EMG trial protocol: REST→CUE→SILENT ARTICULATION→REST/RECOVERY | §18 | IMPLEMENTED | `backend/protocol/session_manager.py` | `tests/acquisition/test_events.py` | Exact durations TBD from pilot |
| REQ-023 | Reproducible randomized/block-randomized trial sequence, saved in metadata | §19 | IMPLEMENTED | `backend/protocol/randomization.py` | `tests/dataset/test_randomization.py` | — |
| REQ-024 | Baseline recording every session | §20 | IMPLEMENTED | `backend/protocol/session_manager.py` | — | REST ≠ final intent class |
| REQ-025 | EEG conditions not frozen before pilot | §21 | IMPLEMENTED | `configs/eeg_experiment.yaml` (candidates only) | — | Label freeze deferred to Gate 6 |
| REQ-026 | EEG contamination pilot comparing rest/cognitive/articulation/combined | §22 | PLANNED | — | — | Requires pilot |
| REQ-027 | EEG preprocessing pipeline | §23 | IMPLEMENTED | `backend/preprocessing/eeg/pipeline.py` | `tests/preprocessing/test_eeg_pipeline.py` | Filter params TBD |
| REQ-028 | EMG preprocessing pipeline | §24 | IMPLEMENTED | `backend/preprocessing/emg/pipeline.py` | `tests/preprocessing/test_emg_pipeline.py` | Filter params TBD |
| REQ-029 | Configurable windowing, pilot-validated | §25 | IMPLEMENTED | `backend/windowing/windower.py`, `configs/windowing.yaml` | `tests/preprocessing/test_windowing.py` | Size/stride TBD from pilot |
| REQ-030 | EMG model set: SVM, GRU, CNN-GRU (frozen) | §26 | IMPLEMENTED | `backend/models/emg/` | `tests/models/test_emg_models.py` | No additions without documented justification |
| REQ-031 | EEG model set: SVM, 1D-CNN (frozen) | §26 | IMPLEMENTED | `backend/models/eeg/` | `tests/models/test_eeg_models.py` | — |
| REQ-032 | Model inputs per §27 | §27 | IMPLEMENTED | `backend/models/emg/`, `backend/models/eeg/` | `tests/models/` | — |
| REQ-033 | Subject/session-aware data splitting, no random window splitting | §28 | IMPLEMENTED | `backend/datasets/splitter.py` | `tests/dataset/test_splitter.py` | split_manifest.json generated |
| REQ-034 | Augmentation: split first, augment TRAIN only, never augment test | §29 | IMPLEMENTED | `backend/datasets/augmentation.py` | `tests/dataset/test_augmentation.py` | — |
| REQ-035 | EMG metrics: Macro-F1 primary, per-class precision/recall, confusion matrix, HELP recall | §30 | IMPLEMENTED | `backend/evaluation/metrics.py` | `tests/models/test_metrics.py` | — |
| REQ-036 | EEG metrics: Macro-F1, accuracy, per-class, confusion, generalization, rejection rate | §31 | IMPLEMENTED | `backend/evaluation/metrics.py` | `tests/models/test_metrics.py` | — |
| REQ-037 | Model selection on validation data only; documented selection criteria | §32 | IMPLEMENTED | `backend/evaluation/model_selector.py` | `tests/models/test_model_selector.py` | — |
| REQ-038 | Test set locked until all preprocessing/model/threshold decisions frozen | §33 | IMPLEMENTED | `backend/datasets/splitter.py` (test_locked flag) | `tests/dataset/test_splitter.py` | — |
| REQ-039 | Decision-level fusion; missing modality support | §34 | IMPLEMENTED | `backend/fusion/decision_fusion.py` | `tests/fusion/test_decision_fusion.py` | — |
| REQ-040 | Confidence/rejection; NO_DECISION output | §35 | IMPLEMENTED | `backend/fusion/confidence.py` | `tests/fusion/test_confidence.py` | Thresholds TBD from validation |
| REQ-041 | Temporal consistency layer (configurable) | §36 | IMPLEMENTED | `backend/fusion/temporal_consistency.py` | `tests/fusion/test_temporal_consistency.py` | Window count TBD from validation |
| REQ-042 | Deterministic safety engine, separate from ML | §37 | IMPLEMENTED | `backend/safety/safety_engine.py` | `tests/safety/test_safety_engine.py` | trigger_state TBD; enabled=false |
| REQ-043 | Output states: YES/NO/HELP/NO_DECISION; internal states | §38 | IMPLEMENTED | `backend/protocol/output_states.py` | `tests/fusion/test_decision_fusion.py` | — |
| REQ-044 | Real-time pipeline architecture | §39 | IMPLEMENTED | `backend/realtime/pipeline.py` | — | Inference on PC first |
| REQ-045 | Mock mode (no hardware, labelled MOCK_DATA) | §40 | IMPLEMENTED | `backend/realtime/mock_source.py` | `tests/acquisition/` | Mock data must never enter research results |
| REQ-046 | Replay mode (load raw session, exact production pipeline) | §40 | IMPLEMENTED | `backend/realtime/replay.py`, `backend/realtime/pipeline.py` | `tests/replay/test_replay.py`, `tests/realtime/test_pipeline.py` | process_window() called per sample; no fabricated predictions |
| REQ-047 | Live mode with full UI status display | §40 | STUBBED | `backend/realtime/pipeline.py` | `tests/realtime/test_pipeline.py` | Live acquisition not yet written — requires hardware |
| REQ-048 | Real-time measurements (acquisition, processing, end-to-end latency) | §41 | IMPLEMENTED | `backend/realtime/latency_monitor.py` | — | Do not invent latency target |
| REQ-049 | Channel ablation (post-core) | §42 | PLANNED | `backend/evaluation/ablation.py` | — | Secondary research phase |
| REQ-050 | Robustness testing (9 conditions) | §43 | PLANNED | — | — | Post-core |
| REQ-051 | Experiment tracking with experiment_manifest.json | §44 | IMPLEMENTED | `backend/evaluation/experiment_tracker.py` | — | git_commit, dataset_version required |
| REQ-052 | Model artifacts with config, label_map, provenance | §45 | IMPLEMENTED | `backend/models/model_registry.py` | `tests/models/test_model_registry.py` | Version compatibility check |
| REQ-053 | Model registry with EXPERIMENTAL/VALIDATED/SELECTED/RETIRED statuses | §46 | IMPLEMENTED | `backend/models/model_registry.py` | `tests/models/test_model_registry.py` | Only SELECTED used in final mode |
| REQ-054 | `configs/final_pipeline.yaml` with TBD fields; refuses to start if TBD present | §47 | IMPLEMENTED | `configs/final_pipeline.yaml`, `backend/realtime/pipeline.py` | `tests/acquisition/test_pipeline_config.py` | — |
| REQ-055 | CLI commands per §48 | §48 | PARTIALLY-IMPLEMENTED | `backend/__main__.py` | — | `train` and `realtime` are stubs; `replay` runs pipeline |
| REQ-056 | Automated tests for all subsystems | §49 | IMPLEMENTED | `tests/` | — | — |
| REQ-057 | Privacy: pseudonymous IDs, no PII in filenames, no external upload | §50 | IMPLEMENTED | `backend/datasets/schema.py`, `backend/datasets/raw_store.py` | — | — |
| REQ-058 | All documentation deliverables per §51 | §51 | IMPLEMENTED | `docs/` | — | — |
| REQ-059 | Decision log per §52 | §52 | IMPLEMENTED | `docs/decision_log.md` | — | — |
| REQ-060 | Requirements traceability per §53 | §53 | IMPLEMENTED | this file | — | — |
| REQ-061 | Acceptance gates 1–12 | §54 | IN-PROGRESS | — | — | Gate 1 complete; Gates 2+ require hardware/data |
| REQ-062 | Reproducibility: config + dataset version + git commit + seed | §44, §57 | IMPLEMENTED | `backend/evaluation/experiment_tracker.py` | — | — |
| REQ-063 | ICA not mandated for 2-channel EEG | §22 | IMPLEMENTED | `backend/preprocessing/eeg/pipeline.py` (ICA optional flag) | — | PRD explicitly states ICA is not automatic for 2-ch |
| REQ-064 | Software must not claim medical-device certification | §10, §13 | IMPLEMENTED | README, `docs/limitations.md` | — | — |
| REQ-065 | All window boundary policies documented | §25 | IMPLEMENTED | `configs/windowing.yaml`, `backend/windowing/windower.py` | `tests/preprocessing/test_windowing.py` | — |

---

## Hardware-Blocked Items (require physical hardware)

| Req ID | Blocking Item | Gate |
|---|---|---|
| REQ-003 | AFE exact part number | H1 / Gate 2 |
| REQ-004 | MCU exact module/variant | Gate 2 |
| REQ-005 | Measured sample rate | H2 / Gate 3 |
| REQ-007 | All filter parameters | Gate 6 |
| REQ-009 | human_use_ready | Gate 3 / institutional |
| REQ-010 (wiring) | IN+/IN− per channel | Gate 2 |

## Pilot-Blocked Items (require pilot data)

| Req ID | Blocking Item | Gate |
|---|---|---|
| REQ-022 | Exact trial timing | Gate 5 → 6 |
| REQ-025 | EEG task/labels | Gate 5 → 6 |
| REQ-029 | Window size/stride | Gate 5 → 6 |
| REQ-040 | Confidence thresholds | Gate 8/9 |
| REQ-041 | Temporal consistency windows | Gate 10 |
| REQ-042 | Safety trigger_state | Gate 9 → 10 |
