# Implementation Plan & Roadmap
## Multimodal Assistive Silent Speech & Affective State Recognition (URF/A-SSR)

**Document Version:** 1.0  
**Date:** 2026-11-20  

---

## 1. Project Phase Breakdown

```
Phase 1: Architecture, Schemas & Immutable Storage (COMPLETE)
Phase 2: Ingestion, Framing & Real-Time Protocol (COMPLETE)
Phase 3: Public Dataset Extraction & Adapters (COMPLETE)
Phase 4: Preprocessing & Feature Extraction Pipelines (COMPLETE)
Phase 5: Model Implementation & Strict LOSO Benchmarking (COMPLETE)
Phase 6: Deterministic Safety Engine & Fusion (COMPLETE)
Phase 7: Hardware-in-the-Loop Integration (READY / SPECIFIED)
Phase 8: Clinical Trial & Assistive Deployment (FUTURE WORK)
```

---

## 2. Completed Milestones & Deliverables

### Milestone 1: Data Contracts & Storage Subsystem
- **Standardized Schema:** Defined unified 8-column data representation (`sample_index, host_timestamp, EEG_Fp1, EEG_Fp2, EMG_submental, EMG_orbicularis, EMG_zygomaticus, EMG_buccinator`).
- **Immutable Raw Storage:** Implemented `RawStore` preventing destructive writes or overwriting of source raw bio-potential files.
- **Protocol Definitions:** Event marker dictionary, trial validity codes, and pseudonymous metadata models.

### Milestone 2: Dataset Adapters & Extraction
- **DEAP Adaptation:**
  - Automated extraction of Fp1 and Fp2 channels.
  - Frequency upsampling from 128 Hz to 2000 Hz using `scipy.signal.resample`.
  - Continuous-to-discrete affective state transformation (`Neutral` vs. `Stressed`).
- **Gaddy Silent Speech sEMG Adaptation:**
  - Extraction of 4 facial speech channels from 8-channel recordings.
  - Text transcript lexical mapping to `YES`, `NO`, `HELP` discrete intent tokens.
  - Participant/session organization creating multi-subject datasets for cross-validation.

### Milestone 3: Machine Learning Model Zoo
- **EMG Intent Classifiers:**
  - `EMGSVMModel`: Classical feature baseline (RMS, MAV, WL, ZCR).
  - `EMGGRUModel`: Deep recurrent architecture for sequence modeling.
  - `EMGCNNGRUModel`: Hybrid convolutional-recurrent model.
- **EEG State Classifiers:**
  - `EEGSVMModel`: Frequency band-power model.
  - `EEGCNN1DModel`: Deep 1D temporal convolutional architecture.

### Milestone 4: Strict LOSO Cross-Validation & Safety Engine
- **Evaluation Engine (`scripts/run_experiments.py`):**
  - Implemented strict Leave-One-Subject-Out (LOSO) cross-validation across 6 subjects.
  - Achieved complete leakage isolation: zero subject overlap across training and test splits.
  - Saved full metric suite (`macro_f1`, `accuracy`, `precision`, `recall`, `confusion_matrix`, `help_recall`) to `experiments/results.json`.
- **Deterministic Safety Engine:**
  - Unit-tested and validated confidence gating ($0.70$ threshold).
  - Verified `NO_DECISION` triggering on low confidence, sensor lead-off, or temporal inconsistency.

---

## 3. Phase 7: Hardware-in-the-Loop (HIL) Integration Plan

### Target Hardware Environment
- **Microcontroller:** STM32H7 / STM32F4 / ESP32-S3 microcontroller running FreeRTOS.
- **Analog Front-End:** Texas Instruments ADS1298 24-bit 8-channel simultaneous sampling AFE.
- **Physical Interface:** USB CDC Virtual COM Port / High-Speed UART (921,600 baud) or SPI-over-FTDI.

### Step-by-Step Transition Roadmap
1. **Firmware Packet Generator Deployment:**
   - Configure ADS1298 DRDY interrupt line on STM32 to trigger SPI DMA read.
   - Pack 6 active channels, 32-bit sample counter, and 64-bit microsecond counter into 42-byte binary frames.
2. **Serial Stream Ingestion:**
   - Implement `SerialSource` in `backend.acquisition` replacing `MockSource`.
   - Utilize non-blocking ring-buffer reading with dedicated frame synchronization thread.
3. **End-to-End Latency Validation:**
   - Measure real hardware timestamp vs. host arrival timestamp using `LatencyMonitor`.
   - Verify overall pipeline processing completes within the <50 ms budget.

---

## 4. Acceptance Criteria & Gate Mapping

| Gate | Description | Validation Method | Status |
|---|---|---|---|
| **Gate 1** | Schema validation & immutability | `tests/dataset/test_raw_store.py` | **PASSED** |
| **Gate 2** | Frame parser & synchronization | `tests/acquisition/test_frame_parser.py` | **PASSED** |
| **Gate 3** | Preprocessing & normalization | `tests/preprocessing/test_*_pipeline.py` | **PASSED** |
| **Gate 4** | Zero data leakage verification | `tests/dataset/test_splitter.py` | **PASSED** |
| **Gate 5** | Strict LOSO cross-validation | `scripts/run_experiments.py` | **PASSED** |
| **Gate 6** | SafetyEngine confidence thresholding | `tests/safety/test_safety_engine.py` | **PASSED** |
| **Gate 7** | Full test suite integrity | `pytest tests/ -v` | **PASSED** |
