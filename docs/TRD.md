# Technical Requirements Document (TRD)
## Multimodal Assistive Silent Speech & Affective State Recognition (URF/A-SSR)

**System:** URF/A-SSR Production Pipeline  
**Version:** 1.0  
**Classification:** Scientific & Medical Assistive Technology  
**Date:** 2026-11-20  

---

## 1. Executive Summary & Objective

The **URF/A-SSR (Universal Research Framework for Assistive Silent Speech Recognition)** is an end-to-end multimodal bio-signal processing software pipeline. The platform integrates **surface Electromyography (sEMG)** from facial speech musculature and **Electroencephalography (EEG)** from prefrontal cortical leads (Fp1, Fp2) to provide communication and distress-signaling capabilities for individuals with severe speech and vocal impairments (e.g., ALS, locked-in syndrome, post-laryngectomy).

### Core Goals:
1. **Silent Speech Intent Recognition:** Decode subvocal articulations of discrete emergency and control tokens (`YES`, `NO`, `HELP`) using facial sEMG.
2. **Cognitive/Affective State Validation:** Detect distress/stress states from prefrontal EEG (`Neutral` vs. `Stressed`) to contextualize intent.
3. **Deterministic Safety Gating:** Prevent spurious activations of safety-critical alarms through a multi-tier confidence and temporal consistency engine, enforcing an absolute `NO_DECISION` fallback whenever signal quality or confidence is degraded.

---

## 2. System Architecture & Modality Specifications

```
  +-------------------------------------------------------------------------+
  |                        BIO-POTENTIAL FRONT END                          |
  |  ADS1298 / AFE (24-bit ADC, Simultaneous Sampling @ 2000 - 4000 SPS)    |
  |  CH1: EEG_Fp1 | CH2: EEG_Fp2 | CH3-CH6: sEMG (Sub, Orb, Zyg, Bucc)     |
  +-------------------------------------------------------------------------+
                                      │ (42-Byte Binary Frames / USB Serial)
                                      ▼
  +─────────────────────────────────────────────────────────────────────────+
  |                       INGESTION & PARSING ENGINE                        |
  |  - backend.acquisition.frame_parser (Frame sync, CRC/Footer check)      |
  |  - Lead-off bitfield extraction & Saturation monitoring                 |
  +─────────────────────────────────────────────────────────────────────────+
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
  +─────────────────────────+                 +─────────────────────────+
  |   sEMG PREPROCESSING    |                 |    EEG PREPROCESSING    |
  | - 50/60 Hz Notch        |                 | - 50/60 Hz Notch        |
  | - 20-450 Hz Bandpass    |                 | - 1-50 Hz Bandpass      |
  | - Z-score Normalization |                 | - Baseline Removal      |
  +─────────────────────────+                 +─────────────────────────+
            │                                                   │
            ▼                                                   ▼
  +─────────────────────────+                 +─────────────────────────+
  |  sEMG INFERENCE BRANCH  |                 |  EEG INFERENCE BRANCH   |
  | - SVM (RMS, MAV, WL, ZCR)|                 | - SVM (Theta, Alpha,    |
  | - Sequence GRU          |                 |   Beta Band Powers)     |
  | - Hybrid CNN-GRU        |                 | - Small 1D-CNN          |
  | Output: Intent, Conf    |                 | Output: State, Conf     |
  +─────────────────────────+                 +─────────────────────────+
            │                                                   │
            └─────────────────────────┬─────────────────────────┘
                                      │
                                      ▼
  +─────────────────────────────────────────────────────────────────────────+
  |                   DECISION FUSION & SAFETY ENGINE                       |
  |  - Temporal Consistency Validation (Sliding confirmation buffer)        |
  |  - Deterministic Safety Rules (min_confidence >= 0.70 threshold)        |
  |  - Intent Gating (HELP requires validated Stressed confirmation)        |
  |  - Fallback: NO_DECISION (Zero false distress action policy)            |
  +─────────────────────────────────────────────────────────────────────────+
```

---

## 3. Functional Requirements

### FR-01: Signal Ingestion & Channel Mapping
- The system shall ingest synchronized 6-channel biological data at a target sampling frequency of 2000 SPS (configurable up to 4000 SPS).
- **Channel Allocation:**
  - CH1: `EEG_Fp1` (Prefrontal Left EEG)
  - CH2: `EEG_Fp2` (Prefrontal Right EEG)
  - CH3: `EMG_submental` (Submental / Anterior Digastric muscle)
  - CH4: `EMG_orbicularis` (Orbicularis Oris muscle)
  - CH5: `EMG_zygomaticus` (Zygomaticus Major muscle)
  - CH6: `EMG_buccinator` (Buccinator muscle)
- Channels must be sampled simultaneously to guarantee zero phase skew between EEG and EMG.

### FR-02: Frame Parsing & Protocol Integrity
- Binary input streams shall adhere to the 42-byte packet structure:
  - Header: `0xAA 0xBB` (2 bytes)
  - Sample Index: `uint32` (4 bytes, little-endian)
  - Host Timestamp: `uint64` (8 bytes, microseconds, little-endian)
  - Channel Samples: 6 × `int32` (24 bytes, 24-bit sign-extended values)
  - Event Marker: `uint8` (1 byte, 0x00 to 0x0E)
  - Status Bitfield: `uint8` (1 byte, lead-off flags CH1-CH6 and saturation bit)
  - Footer: `0xCC 0xDD` (2 bytes)
- Gaps in `sample_index` must be flagged immediately as dropped frames.

### FR-03: Preprocessing Pipelines
- **EMG Filtering:** 4th-order zero-phase Butterworth bandpass filter with passband [20.0 Hz, 450.0 Hz], plus a 50 Hz/60 Hz notch filter with Q=30.
- **EEG Filtering:** 4th-order zero-phase Butterworth bandpass filter with passband [1.0 Hz, 50.0 Hz], plus line noise notch filtering.
- **Normalization:** Running Z-score normalization computed per-channel using strictly non-leaking statistics.

### FR-04: Feature Extraction
- **sEMG Features (Time-Domain):**
  - Root Mean Square (RMS)
  - Mean Absolute Value (MAV)
  - Waveform Length (WL)
  - Zero Crossing Rate (ZCR)
  - Output vector size: 16 features (4 features × 4 channels).
- **EEG Features (Frequency-Domain):**
  - Power Spectral Density (Welch's method) across standard bands:
    - Theta (4.0 - 8.0 Hz)
    - Alpha (8.0 - 13.0 Hz)
    - Beta (13.0 - 30.0 Hz)
  - Relative Band Power ratios ($P_{band} / P_{total}$).
  - Output vector size: 12 features (6 per prefrontal channel).

### FR-05: Model Inference Architectures
- **EMG Intent Classifiers:**
  - `EMGSVMModel`: Radial Basis Function (RBF) Support Vector Machine on 16D feature vectors.
  - `EMGGRUModel`: 2-layer Gated Recurrent Unit (hidden dimension 64, dropout 0.2) processing raw/windowed 4-channel sequences.
  - `EMGCNNGRUModel`: Hybrid 1D-CNN (32 filters, kernel 5) feature extractor followed by a 2-layer GRU for spatio-temporal modeling.
- **EEG State Classifiers:**
  - `EEGSVMModel`: RBF-SVM trained on 12D band-power features.
  - `EEGCNN1DModel`: 2-layer temporal 1D convolutional neural network (16 filters, kernel 7, dropout 0.3) for direct end-to-end 2-channel classification.

### FR-06: Deterministic Safety Engine & Thresholding
- Machine learning probabilities must never trigger safety-critical output directly.
- The `SafetyEngine` enforces explicit threshold rules:
  1. `intent == "HELP"`
  2. `state == "Stressed"`
  3. `intent_confidence >= min_intent_confidence` (nominal 0.70)
  4. `state_confidence >= min_state_confidence` (nominal 0.70)
  5. `signal_quality == "OK"` (Lead-off and saturation bits clear)
  6. `temporal_consistency_met == True` (Decision sustained across $N$ consecutive windows)
- If ANY condition fails, the system outputs `NO_DECISION` and logs the rejection cause to the audit log.

---

## 4. Non-Functional Requirements (NFR)

### NFR-01: Real-Time Latency Budget
- Maximum end-to-end processing latency from packet receipt to decision delivery: **< 50.0 ms**.
- Breakdown:
  - Frame Parsing & Ingestion: < 1.0 ms
  - Filtering & Normalization: < 5.0 ms
  - Feature Extraction / Windowing: < 8.0 ms
  - Model Inference (EMG + EEG): < 20.0 ms
  - Decision Fusion & Safety Logic: < 2.0 ms
  - Margin: 14.0 ms

### NFR-02: Zero Data Leakage Guarantee
- Subject-aware and session-aware dataset partitioning is mandatory.
- Cross-validation must strictly employ Leave-One-Subject-Out (LOSO).
- No test-fold samples or test-subject normalization parameters may ever enter the training pipeline.

### NFR-03: Fault Tolerance & Signal QC
- The system must detect flatlines ($Var < 10^{-6}$), ADC clipping/saturation, and electrode lead-off within 50 ms.
- Any degraded signal invalidates downstream model inference and yields `NO_DECISION`.

### NFR-04: Auditability & Security
- Every decision emitted by the pipeline must be recorded with:
  - Exact timestamp (ISO 8601 UTC)
  - Input probabilities and confidence values
  - Applied rule identifier
  - Hardware signal quality status
- Participant data must maintain pseudonymous identification (`Subject_XX`) without personally identifiable information (PII).
