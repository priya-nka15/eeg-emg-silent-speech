# URF/A-SSR Decision Log

Every major decision is recorded here with full traceability.
Hardware-dependent values that are currently unknown are marked **TBD** and must not be invented.

---

## Format

| Field | Description |
|---|---|
| Date | ISO 8601 date |
| Decision ID | Unique identifier |
| Decision | What was decided |
| Reason | Why this decision was made |
| Evidence | Supporting source(s) |
| Alternatives considered | What else was evaluated |
| Status | OPEN / CLOSED / SUPERSEDED |
| Impact | What this affects |

---

## DL-001

- **Date:** 2026-10-08
- **Decision ID:** DL-001
- **Decision:** Greenfield repository creation. No prior code existed to preserve.
- **Reason:** Workspace was empty on first inspection.
- **Evidence:** Directory listing of `/home/ubuntupj/Desktop/URF/assr1` returned empty.
- **Alternatives considered:** N/A
- **Status:** CLOSED
- **Impact:** All code is new. PRD sections 4.14 (preserve working components) is satisfied by the absence of prior components.

---

## DL-002

- **Date:** 2026-10-08
- **Decision ID:** DL-002
- **Decision:** AFE target recorded as ADS1298-class. Exact part number is TBD pending hardware team confirmation.
- **Reason:** PRD section 6.1 specifies ADS1298-class as the current direction but explicitly states the exact purchased part number is hardware-dependent.
- **Evidence:** PRD §6.1; TI ADS1298 datasheet (8-ch, 24-bit, simultaneous-sampling biopotential AFE, 250 SPS–32 kSPS).
- **Alternatives considered:** N/A — decision deferred to hardware team.
- **Status:** OPEN — requires hardware team to supply exact part number.
- **Impact:** `configs/hardware.yaml` field `afe_part_number` is TBD. Software uses the interface contract, not the part number.

---

## DL-003

- **Date:** 2026-10-08
- **Decision ID:** DL-003
- **Decision:** MCU target recorded as ESP32-S3. Exact module/variant and firmware revision are TBD.
- **Reason:** PRD §6.2 specifies ESP32-S3 as the current target but requires exact variant from hardware team.
- **Evidence:** PRD §6.2.
- **Alternatives considered:** N/A — decision deferred to hardware team.
- **Status:** OPEN — requires hardware team to supply exact module and firmware revision.
- **Impact:** `configs/hardware.yaml` fields `mcu_module` and `firmware_revision` are TBD.

---

## DL-004

- **Date:** 2026-10-08
- **Decision ID:** DL-004
- **Decision:** Sampling rate target set to 4,000 SPS/channel in configuration. Actual measured rate must be verified by hardware counter/timestamp, not assumed.
- **Reason:** PRD §6.3 sets 4 kSPS as the current project target and explicitly requires verification from counters/timestamps.
- **Evidence:** PRD §6.3.
- **Alternatives considered:** N/A — target from PRD.
- **Status:** CLOSED for configuration target; OPEN for measured verification (requires hardware, Gate H2).
- **Impact:** `configs/hardware.yaml` `sampling_rate_hz: 4000` (target). `hardware_validation_report_v1.md` must record the measured rate.

---

## DL-005

- **Date:** 2026-10-08
- **Decision ID:** DL-005
- **Decision:** All hardware filter parameters (bandpass, notch, LMS/NLMS, wavelet) recorded as TBD.
- **Reason:** PRD §8 explicitly states exact values are not currently frozen and must not be invented.
- **Evidence:** PRD §8.
- **Alternatives considered:** N/A — invention of parameters is prohibited by PRD rule 3.
- **Status:** OPEN — requires hardware team confirmation.
- **Impact:** `configs/hardware_preprocessing.yaml` all filter fields are TBD.

---

## DL-006

- **Date:** 2026-10-08
- **Decision ID:** DL-006
- **Decision:** EEG task/labels are not frozen. Candidate conditions listed in PRD §21 are recorded as candidates. Final labels require pilot validation.
- **Reason:** PRD §21 explicitly prohibits permanently freezing EEG labels before the pilot. PRD rule 5 prohibits premature freezing.
- **Evidence:** PRD §21, §58 RQ3.
- **Alternatives considered:** N/A — prohibition is non-negotiable.
- **Status:** OPEN — to be resolved after EEG pilot (Gate 5, Gate 6).
- **Impact:** EEG model label map uses placeholder labels. `configs/final_pipeline.yaml` eeg model_id is TBD.

---

## DL-007

- **Date:** 2026-10-08
- **Decision ID:** DL-007
- **Decision:** Decision-level fusion selected as the first multimodal implementation.
- **Reason:** PRD §34 explicitly requires decision-level fusion as the first implementation. Raw concatenation into a large multimodal neural network is explicitly prohibited.
- **Evidence:** PRD §34.
- **Alternatives considered:** Feature-level fusion, raw concatenation — both prohibited for first implementation.
- **Status:** CLOSED
- **Impact:** `backend/fusion/` implements decision-level fusion only.

---

## DL-008

- **Date:** 2026-10-08
- **Decision ID:** DL-008
- **Decision:** Model set frozen: EMG → SVM, GRU, CNN-GRU; EEG → SVM, 1D-CNN. No additional models without documented research justification.
- **Reason:** PRD §26 explicitly freezes the model set. Adding Transformer, LSTM, TCN, XGBoost, Random Forest, or LDA requires a separate documented research decision.
- **Evidence:** PRD §26.
- **Alternatives considered:** Transformer, LSTM, TCN, XGBoost, Random Forest, LDA — all excluded per PRD.
- **Status:** CLOSED
- **Impact:** `backend/models/emg/` and `backend/models/eeg/` implement only the frozen model set.

---

## DL-009

- **Date:** 2026-10-08
- **Decision ID:** DL-009
- **Decision:** IN+/IN− electrode wiring for all 6 differential channels recorded as TBD until the actual PCB schematic is supplied.
- **Reason:** PRD §5.2 and rule 2 prohibit inventing electrode wiring or IN+/IN− positions.
- **Evidence:** PRD §5.2, rule 2.
- **Alternatives considered:** N/A — invention is prohibited.
- **Status:** OPEN — requires actual PCB schematic from hardware team.
- **Impact:** `docs/electrode_map.md` and `configs/electrodes.yaml` contain TBD wiring fields.

---

## DL-010

- **Date:** 2026-10-08
- **Decision ID:** DL-010
- **Decision:** LMS/NLMS reference signal recorded as TBD. LMS is marked hardware-unverified until the reference signal source is confirmed.
- **Reason:** PRD §8 LMS requirement explicitly states the reference signal must be documented. A common EEG reference is not automatically a motion-artifact reference. PRD rule 4 prohibits inventing the LMS reference.
- **Evidence:** PRD §8, rule 4.
- **Alternatives considered:** N/A — invention is prohibited.
- **Status:** OPEN — requires hardware team to confirm reference signal source.
- **Impact:** `configs/hardware_preprocessing.yaml` LMS fields are TBD.

---

## DL-011

- **Date:** 2026-10-08
- **Decision ID:** DL-011
- **Decision:** `human_use_ready` flag initialised to `false`. Remains false until hardware team/institution completes electrical and experimental safety review.
- **Reason:** PRD §10 requires this flag and prohibits setting it without qualified safety documentation.
- **Evidence:** PRD §10.
- **Alternatives considered:** N/A.
- **Status:** OPEN — requires institutional safety review.
- **Impact:** `configs/hardware.yaml` `human_use_ready: false`. Live mode must check this flag.

---

## DL-012

- **Date:** 2026-10-08
- **Decision ID:** DL-012
- **Decision:** Window size, stride, alignment and label policy are TBD pending pilot analysis.
- **Reason:** PRD §25 explicitly prohibits hard-coding final window size before pilot analysis.
- **Evidence:** PRD §25.
- **Alternatives considered:** N/A — pilot-driven.
- **Status:** OPEN — to be resolved after pilot (Gate 5 → Gate 6).
- **Impact:** `configs/windowing.yaml` fields are TBD.

---

## DL-013

- **Date:** 2026-10-08
- **Decision ID:** DL-013
- **Decision:** Safety engine trigger state (`trigger_state`) is TBD because final EEG label is not frozen.
- **Reason:** PRD §37 explicitly prohibits hard-coding DISTRESS as the final state identifier.
- **Evidence:** PRD §37.
- **Alternatives considered:** N/A.
- **Status:** OPEN — depends on EEG pilot and label freeze (Gate 6, Gate 9).
- **Impact:** `configs/safety.yaml` `trigger_state: TBD`. Safety engine enabled flag is `false`.

---

*Add new entries below this line as decisions are made.*

---

## DL-014

- **Date:** 2026-10-08
- **Decision ID:** DL-014
- **Decision:** LIVE safety gate hardened: hardware.yaml resolved from repository root (not CWD); fail-closed when file missing; `final_pipeline_config_path=None` now raises; all four bypass cases covered by regression tests.
- **Reason:** Audit finding C-1 — LIVE mode started without hardware check when process launched from non-repo directory.
- **Evidence:** Audit confirmed: `RealtimePipeline(mode=LIVE)` from `/tmp` with no hardware.yaml returned success.
- **Alternatives considered:** CWD-relative path — rejected because it is silently unsafe.
- **Status:** CLOSED
- **Impact:** `backend/realtime/pipeline.py` `_check_live_prerequisites`; `tests/realtime/test_pipeline.py`.

---

## DL-015

- **Date:** 2026-10-08
- **Decision ID:** DL-015
- **Decision:** `cmd_replay` now calls `process_window()` for every sample. Imports `RealtimePipeline` and invokes the production pipeline. No predictions are fabricated — NO_DECISION is returned when no model is loaded, which is correct.
- **Reason:** Audit finding C-2 — old `cmd_replay` only counted samples.
- **Evidence:** Audit confirmed `process_window` was never called in the original implementation.
- **Alternatives considered:** N/A — PRD §40 requires replay to run the exact production pipeline.
- **Status:** CLOSED
- **Impact:** `backend/__main__.py`; `tests/realtime/test_pipeline.py`.

---

## DL-016

- **Date:** 2026-10-08
- **Decision ID:** DL-016
- **Decision:** H8 `validate_sustained_streaming` now requires `expected_sampling_rate_hz` to be supplied explicitly. Returns NOT_STARTED if not provided. Hard-coded `4000` removed.
- **Reason:** Audit finding C-3 — hard-coded 4000 violated PRD rule 1 (never invent hardware specs).
- **Evidence:** Audit confirmed `total_expected = int(duration_s * 4000)` on line 410 of hardware_validator.py.
- **Alternatives considered:** Default to target rate — rejected because it silently assumes the target equals the measured rate.
- **Status:** CLOSED
- **Impact:** `backend/acquisition/hardware_validator.py`; `tests/acquisition/test_hardware_validator.py`.

---

## DL-017

- **Date:** 2026-10-08
- **Decision ID:** DL-017
- **Decision:** H3 `validate_synchronization` now performs a structural check (monotone sample_ids, non-decreasing timestamps) but always returns NOT_STARTED. Physical simultaneous-acquisition verification requires hardware measurement.
- **Reason:** Audit finding C-5 — old H3 used `hasattr()` on dataclass fields which always returns True, so the gate could never fail.
- **Evidence:** Audit confirmed all `hasattr(s, ch)` calls always return True for a dataclass.
- **Alternatives considered:** Return PASS on structural check passing — rejected because structural correctness ≠ physical simultaneity.
- **Status:** CLOSED
- **Impact:** `backend/acquisition/hardware_validator.py`; `tests/acquisition/test_hardware_validator.py`.

---

## DL-018

- **Date:** 2026-10-08
- **Decision ID:** DL-018
- **Decision:** `temporal_consistency_met` passed to SafetyEngine now captures the actual return value of `TemporalConsistencyFilter.update()`, not the fusion output.
- **Reason:** Audit finding C-4 — old code inferred temporal consistency from fusion result, bypassing the temporal filter entirely.
- **Evidence:** Audit confirmed `temporal_consistency_met = (fusion_result.intent != NO_DECISION)` on pipeline.py line 69.
- **Alternatives considered:** N/A — PRD §37 explicitly lists temporal_consistency as a safety engine input.
- **Status:** CLOSED
- **Impact:** `backend/realtime/pipeline.py`; `tests/realtime/test_pipeline.py`.

---

## DL-019

- **Date:** 2026-10-08
- **Decision ID:** DL-019
- **Decision:** `SessionManager._session_start_us` initialised to -1 (sentinel). `_now_us()` raises `RuntimeError` if called before `start_session()`.
- **Reason:** Audit finding H-7 — before `start_session()`, `_now_us()` returned a large epoch-relative value (~13 billion µs), producing incorrect event timestamps.
- **Evidence:** Audit confirmed `_session_start_us=0` caused `_now_us()` to return 13_325_886_434 µs before session started.
- **Alternatives considered:** N/A — a sentinel that raises is unambiguous.
- **Status:** CLOSED
- **Impact:** `backend/protocol/session_manager.py`; `tests/acquisition/test_session_manager.py`.

---

## DL-020

- **Date:** 2026-10-08
- **Decision ID:** DL-020
- **Decision:** `ModelSelector` redesigned with two explicit methods: `final_select()` (raises `SelectionError` if criteria not frozen or `min_safety_recall` is None) and `preliminary_select()` (warns only). `min_safety_recall` default changed from `0.0` to `None`.
- **Reason:** Audit findings H-10 and M-6 — unfrozen criteria only warned without blocking; `0.0` default silently disabled the HELP recall safety filter.
- **Evidence:** Confirmed `criteria_frozen=False` + `min_safety_recall=0.0` allowed selection to proceed without any block.
- **Alternatives considered:** Raising on any call with `criteria_frozen=False` — rejected as too restrictive for exploratory analysis.
- **Status:** CLOSED
- **Impact:** `backend/evaluation/model_selector.py`; `tests/models/test_model_selector.py`.

---

## DL-021

- **Date:** 2026-10-08
- **Decision ID:** DL-021
- **Decision:** EEG model classes (`EEGSVMModel`, `EEGCNN1DModel`) now emit `UserWarning` on init, fit, predict, and load when the label map contains placeholder labels (CONDITION_A, CONDITION_B). The warning fires regardless of whether the label map was passed explicitly or defaulted.
- **Reason:** Audit finding M-7 — old warning only fired when `label_map=None`, not when the placeholder dict was passed explicitly.
- **Evidence:** Confirmed that `EEGSVMModel(label_map={"CONDITION_A": 0, "CONDITION_B": 1})` produced no warning.
- **Alternatives considered:** N/A.
- **Status:** CLOSED
- **Impact:** `backend/models/eeg/svm_model.py`, `backend/models/eeg/cnn1d_model.py`; `tests/models/test_eeg_models.py`.

---

## DL-022

- **Date:** 2026-10-08
- **Decision ID:** DL-022
- **Decision:** `SignalQC.__init__` now emits `UserWarning` when `adc_max=None` (saturation detection disabled) and a second warning that all thresholds are unvalidated defaults.
- **Reason:** Audit finding H-6 — `SignalQC()` with defaults provided no indication that saturation detection was inactive or that thresholds were unvalidated placeholders.
- **Alternatives considered:** N/A.
- **Status:** CLOSED
- **Impact:** `backend/qc/signal_qc.py`; existing tests updated to catch warnings.

---

## DL-023

- **Date:** 2026-10-08
- **Decision ID:** DL-023
- **Decision:** `validate-dataset` CLI command now validates `subject.json`, `session.json`, and `trials.json` in every `metadata/` directory. Missing metadata or empty directories now fail validation. `cmd_realtime` no longer prints "Prerequisites passed. Starting live pipeline..." without starting anything.
- **Reason:** Audit findings H-5 (validate-dataset), M-9 (misleading print).
- **Alternatives considered:** N/A.
- **Status:** CLOSED
- **Impact:** `backend/__main__.py`; `tests/dataset/test_pipeline_config.py`.

---

## DL-024

- **Date:** 2026-10-08
- **Decision ID:** DL-024
- **Decision:** `scripts/validate_hardware.py` created as a proper CLI runner for H1–H8 gates. Reports PASS/FAIL/NOT_STARTED/BLOCKED/INFORMATIONAL. Never manufactures hardware results. H8 requires `--expected-rate` explicitly; refuses to default to 4000.
- **Reason:** Traceability matrix referenced this script as IMPLEMENTED but the file did not exist.
- **Alternatives considered:** N/A — file is required by REQ-008 traceability entry.
- **Status:** CLOSED
- **Impact:** `scripts/validate_hardware.py`.
