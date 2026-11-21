# URF/A-SSR — Multimodal EEG+EMG Silent Communication Research Prototype

**Version:** 0.1.0  
**Status:** Phase A/B complete — hardware-dependent fields pending

> **This is a research prototype.** It is not a medical device and has not been clinically validated. Do not use or describe it as a clinical tool without appropriate regulatory approval.

---

## Overview

URF/A-SSR acquires six simultaneous biopotential channels and implements a multimodal silent-communication system combining EMG-based intent recognition (YES/NO/HELP) with EEG-based cognitive/affective state estimation.

| Channel | Modality | Target |
|---|---|---|
| CH1 | EEG | Fp1 |
| CH2 | EEG | Fp2 |
| CH3 | EMG | Submental/anterior-digastric |
| CH4 | EMG | Orbicularis oris |
| CH5 | EMG | Zygomaticus major |
| CH6 | EMG | Buccinator |

Hardware: ADS1298-class AFE + ESP32-S3 MCU @ 4 kSPS target.

---

## Repository Structure

```
urf-a-ssr/
├── firmware/             # ESP32-S3 firmware (hardware team)
├── backend/              # All Python source code
│   ├── acquisition/      # Frame parser, hardware validator
│   ├── protocol/         # Events, session manager, output states
│   ├── qc/               # Signal QC, trial validator
│   ├── preprocessing/    # EEG + EMG preprocessing pipelines
│   ├── windowing/        # Configurable windowing
│   ├── features/         # Feature extraction (EMG + EEG)
│   ├── datasets/         # Schema, raw store, splitter, augmentation
│   ├── models/           # EMG: SVM/GRU/CNN-GRU; EEG: SVM/1D-CNN
│   ├── evaluation/       # Metrics, model selector, experiment tracker
│   ├── fusion/           # Decision fusion, confidence, temporal consistency
│   ├── safety/           # Deterministic safety engine
│   └── realtime/         # Pipeline, mock/replay/live modes, latency monitor
├── configs/              # YAML configurations (many fields TBD)
├── data/                 # Raw/processed data (not committed)
├── experiments/          # Experiment manifests and results
├── tests/                # Automated test suite (155 tests)
├── scripts/              # CLI entry points
├── notebooks/            # Analysis notebooks
└── docs/                 # All documentation
```

---

## Quick Start

```bash
# Install
pip install -e ".[test]"

# Run all tests
python -m pytest tests/ -v

# CLI help
python -m backend --help

# Validate a recording
python -m backend validate-recording /path/to/recording.csv

# Replay a session
python -m backend replay /path/to/recording.csv
```

---

## Key Rules (PRD §4)

1. **Never invent hardware specifications** — all TBD fields are placeholders
2. **Never invent electrode wiring** — IN+/IN− requires PCB schematic
3. **Never invent filter parameters** — hardware team provides these
4. **Never invent LMS reference signals** — hardware team confirms
5. **EEG labels not frozen** — pilot determines conditions
6. **Raw data is immutable** — never overwrite `raw/`
7. **No leakage-prone window splitting** — subject/session-aware splits only
8. **Never use test set for model selection** — validation only
9. **Never claim other papers' accuracy** — measure from real data
10. **Never force a decision when quality is inadequate** — return NO_DECISION

---

## Current Status

| Phase | Status | Blocker |
|---|---|---|
| A: Repository + Specification | **COMPLETE** | — |
| B: Acquisition | **COMPLETE** | — |
| C: Hardware Validation | BLOCKED | Physical hardware |
| D: Preprocessing | **COMPLETE** | — |
| E: Protocol Freeze | BLOCKED | Pilot required |
| F: Main Dataset | BLOCKED | Pilot + protocol freeze |
| G: EMG ML | **IMPLEMENTED** | Training requires dataset |
| H: EEG ML | **IMPLEMENTED** | Labels + dataset required |
| I: Fusion | **COMPLETE** | — |
| J: Real Time | **IMPLEMENTED** | Live mode requires hardware |
| K: Robustness | NOT STARTED | Trained models + hardware |
| L: Research Package | NOT STARTED | All prior phases |

### TBD Registry (Hardware-Dependent)

- `afe_part_number`, `mcu_module`, `firmware_revision`, `pcb_revision`
- `gain_eeg`, `gain_emg`
- All electrode IN+/IN− wiring (6 channels)
- All filter parameters (bandpass, notch, LMS, wavelet)
- Transport protocol and frame format version
- ADC voltage conversion factor
- `human_use_ready` — requires institutional safety review

### TBD Registry (Pilot-Validated)

- EEG task conditions and labels
- Exact trial timing (REST/CUE/ARTICULATION durations)
- Window size, stride, label policy
- Confidence rejection thresholds
- Temporal consistency window count
- Safety engine trigger_state and thresholds

---

## Documentation

| Document | Location |
|---|---|
| System architecture | `docs/architecture.md` |
| Hardware interface spec | `docs/hardware_interface_spec_v1.md` |
| Electrode map | `docs/electrode_map.md` |
| Preprocessing spec | `docs/preprocessing_spec.md` |
| Dataset schema | `docs/dataset_schema.md` |
| Data collection protocol | `docs/data_collection_protocol.md` |
| QC protocol | `docs/qc_protocol.md` |
| Model development | `docs/model_development.md` |
| Evaluation protocol | `docs/evaluation_protocol.md` |
| Fusion specification | `docs/fusion_spec.md` |
| Safety specification | `docs/safety_spec.md` |
| Real-time specification | `docs/realtime_spec.md` |
| Hardware validation report | `docs/validation_report.md` |
| Decision log | `docs/decision_log.md` |
| Requirements traceability | `docs/requirements_traceability.md` |
| Implementation status | `docs/implementation_status.md` |
| Limitations | `docs/limitations.md` |
| Reproducibility | `docs/reproducibility.md` |

---

## Acceptance Gates

Gates 1–2 are in progress. Gates 3–12 require hardware and/or data. See `docs/implementation_status.md`.

---

## Research Questions

- **RQ1:** Can facial/submental EMG reliably distinguish YES/NO/HELP during silent articulation?
- **RQ2:** Which of SVM/GRU/CNN-GRU provides the best performance-robustness-cost trade-off?
- **RQ3:** Can 2-channel frontal EEG discriminate experimentally defined cognitive/affective conditions?
- **RQ4:** Can independent EMG intent + EEG state predictions produce reliable safety-aware communication?
