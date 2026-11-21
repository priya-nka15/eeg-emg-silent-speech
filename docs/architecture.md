# URF/A-SSR System Architecture

**Version:** 1.0  
**Date:** 2026-10-08

---

## Overview

URF/A-SSR is a research prototype for multimodal silent communication using simultaneous EEG and EMG acquisition. It is not a medical device and has not been clinically validated.

---

## Signal Channels

| Channel | Modality | Target Location | Status |
|---|---|---|---|
| CH1 | EEG | Fp1 | FROZEN |
| CH2 | EEG | Fp2 | FROZEN |
| CH3 | EMG | Submental / anterior-digastric | FROZEN |
| CH4 | EMG | Orbicularis oris | FROZEN |
| CH5 | EMG | Zygomaticus major | FROZEN |
| CH6 | EMG | Buccinator | FROZEN |

---

## Hardware Stack

```
Custom PCB
  ADS1298-class 8-channel 24-bit biopotential AFE  [part: TBD]
  ESP32-S3 MCU  [module: TBD, firmware: TBD]
  14 electrodes (12 signal + 1 EEG ref + 1 EMG ref)
```

Target sample rate: **4,000 SPS/channel** (measured rate requires hardware validation).

---

## Full System Architecture

```
                    HUMAN SUBJECT
                         │
                 EEG + EMG simultaneously
                         │
                         ▼
              CUSTOM 6-CHANNEL HARDWARE
              ADS1298-class + ESP32-S3
                         │
                    4 kSPS target
                         │
                         ▼
                SYNCHRONIZED STREAM
                         │
                         ▼
                    RAW DATA STORE
                  (immutable, never overwritten)
                         │
                         ▼
                         QC
                  (missing, flatline, saturation)
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
            EEG                   EMG
          Fp1/Fp2              4 channels
              │                     │
              ▼                     ▼
       preprocessing         preprocessing
    (filtering, artifact)  (filtering, normalization)
              │                     │
              ▼                     ▼
          windowing             windowing
       [size: TBD]           [size: TBD]
              │                     │
              ▼                     ▼
       features/input        features/input
      (band powers, etc.)  (RMS, MAV, WL, ZCR, etc.)
              │                     │
          ┌───┴───┐           ┌────┼────┐
          ▼       ▼           ▼    ▼    ▼
         SVM    1D-CNN       SVM  GRU CNN-GRU
          │       │           │    │    │
          └───┬───┘           └────┴────┘
              │                     │
              ▼                     ▼
          EEG state             EMG intent
        [labels: TBD]        YES / NO / HELP
              │                     │
              └──────────┬──────────┘
                         ▼
                DECISION-LEVEL FUSION
                         │
                         ▼
                  CONFIDENCE / QC
                 (NO_DECISION if insufficient)
                         │
                         ▼
                TEMPORAL CONSISTENCY
                   [windows: TBD]
                         │
                         ▼
                   SAFETY ENGINE
                  (deterministic, not ML)
                  [trigger_state: TBD]
                  [enabled: false]
                         │
                    ┌────┴────┐
                    ▼         ▼
                  ACTION   NO_DECISION
```

---

## Repository Structure

```
urf-a-ssr/
├── firmware/                   # ESP32-S3 firmware (hardware team)
├── backend/
│   ├── acquisition/            # Frame parser, hardware validator
│   ├── protocol/               # Events, session manager, output states
│   ├── qc/                     # Signal quality, trial validator
│   ├── preprocessing/
│   │   ├── eeg/                # EEG preprocessing pipeline
│   │   └── emg/                # EMG preprocessing pipeline
│   ├── calibration/            # Participant-specific calibration
│   ├── windowing/              # Windowing with configurable params
│   ├── features/               # EMG + EEG feature extraction
│   ├── datasets/               # Schema, raw store, splitter, augmentation
│   ├── models/
│   │   ├── emg/                # SVM, GRU, CNN-GRU
│   │   └── eeg/                # SVM, 1D-CNN
│   ├── evaluation/             # Metrics, model selector, experiment tracker, ablation
│   ├── fusion/                 # Decision fusion, confidence, temporal consistency
│   ├── safety/                 # Deterministic safety engine
│   └── realtime/               # Pipeline, mock source, replay, latency monitor
├── configs/                    # All YAML configuration files
├── data/
│   ├── raw/                    # Immutable raw recordings
│   ├── processed/              # Preprocessed data
│   ├── windows/                # Windowed segments
│   └── manifests/              # Dataset and split manifests
├── experiments/                # Experiment results and manifests
├── tests/                      # Automated test suite
├── scripts/                    # CLI entry points
├── notebooks/                  # Analysis notebooks
└── docs/                       # All documentation
```

---

## Operating Modes

| Mode | Description | Hardware Required |
|---|---|---|
| Mock | Software dev/testing; all data labelled MOCK_DATA | No |
| Replay | Load raw session, run production pipeline | No |
| Live | Real hardware streaming | Yes + human_use_ready=true |

---

## Output States

**User-facing intent:** YES | NO | HELP | NO_DECISION  
**Internal:** SIGNAL_INVALID | CHANNEL_FAULT | STATE_UNCERTAIN | MODEL_UNCERTAIN

---

## Research Questions

- **RQ1:** Can facial/submental EMG reliably distinguish YES/NO/HELP during silent articulation?
- **RQ2:** Which of SVM/GRU/CNN-GRU gives best performance-robustness-cost trade-off?
- **RQ3:** Can 2-channel frontal EEG discriminate experimentally defined cognitive/affective conditions?
- **RQ4:** Can independent EMG intent + EEG state predictions produce reliable safety-aware communication?

---

## Important Constraints

- This is a **research prototype**, not a medical device.
- `human_use_ready` must be `true` before live use with participants.
- EEG labels are **not frozen** until after the pilot.
- All hardware-dependent parameters remain TBD until hardware team confirms.
- Raw data is **immutable**.
- Test set is **locked** until all preprocessing/model/threshold decisions are frozen.
