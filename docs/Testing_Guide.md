# Testing Guide & Quality Assurance Protocol
## Multimodal Assistive Silent Speech & Affective State Recognition (URF/A-SSR)

**Document Version:** 1.0  
**Date:** 2026-11-20  

---

## 1. Testing Philosophy & Verification Strategy

The URF/A-SSR verification harness enforces strict safety-critical software engineering standards:
1. **Determinism:** Tests must yield 100% reproducible outcomes regardless of execution order or random seed.
2. **Zero Data Leakage:** Mathematical and structural verification that no participant or session data overlaps between training, validation, and test splits.
3. **Explicit Fallback Paths:** The safety-critical `NO_DECISION` fallback must be rigorously exercised across all fault modalities (low confidence, lead-off, saturation, temporal disagreement).
4. **Hardware-Decoupled Verification:** Hardware interfaces must be simulated using bit-exact binary synthesizers and high-fidelity CSV replays before deployment to physical microcontrollers.

---

## 2. Test Suite Architecture

The repository contains 12 dedicated test modules structured under `tests/`:

```
tests/
├── acquisition/
│   ├── test_events.py              # Event markers, logging, roundtrip serialization
│   ├── test_frame_parser.py        # Binary wire parsing, header/footer, gap detection
│   ├── test_hardware_validator.py  # H1-H5 hardware validation acceptance checks
│   └── test_session_manager.py     # Session state machine, subject validation
├── dataset/
│   ├── test_pipeline_config.py     # Configuration parsing & TBD gate enforcement
│   ├── test_randomization.py       # Deterministic pseudo-random trial sequences
│   ├── test_raw_store.py           # Immutable raw data storage policy enforcement
│   ├── test_schema.py              # Subject, Session, Trial, Sample model validation
│   └── test_splitter.py            # Subject-held-out, session-held-out, LOSO leakage checks
├── features/                       # Feature extraction shape and computation tests
├── fusion/
│   ├── test_confidence.py          # Multimodal confidence scaling & calibration
│   ├── test_decision_fusion.py     # Multimodal rule combinations & decision matrix
│   └── test_temporal_consistency.py# Running window confirmation & debounce logic
├── models/
│   ├── test_eeg_models.py          # EEGSVMModel and EEGCNN1DModel fit/predict
│   ├── test_emg_models.py          # EMGSVMModel, EMGGRUModel, EMGCNNGRUModel
│   ├── test_metrics.py             # Macro-F1, Confusion matrix, HELP-recall tests
│   ├── test_model_registry.py      # Artifact tracking & version compatibility
│   └── test_model_selector.py      # Model selection constraints & criteria freezing
├── preprocessing/
│   ├── test_eeg_pipeline.py        # Filtering, detrending, NaN checks
│   ├── test_emg_pipeline.py        # Notch, bandpass, Z-score scaling
│   └── test_windowing.py           # Window segmentation, stride, boundary policies
├── qc/
│   ├── test_signal_qc.py           # Lead-off detection, saturation, flatline checks
│   └── test_trial_validator.py     # Trial integrity & invalid reason codes
├── realtime/
│   └── test_pipeline.py            # End-to-end streaming latency & pipeline throughput
├── replay/
│   └── test_replay.py              # CSV replay timing, rate limiting, and playback
└── safety/
    └── test_safety_engine.py       # Deterministic safety rule enforcement & audit logs
```

---

## 3. Running the Test Suite

### 3.1 Standard Execution
To run all automated unit and integration tests:

```bash
pytest tests/ -v
```

Expected Output:
```
============================= test session starts ==============================
...
======================= 261 passed, 8 skipped in ~3.0s =========================
```

### 3.2 Running Specific Test Subsystems
```bash
# Verify Safety Engine deterministic behavior:
pytest tests/safety/test_safety_engine.py -v

# Verify Zero Data Leakage across dataset splits:
pytest tests/dataset/test_splitter.py -v

# Verify Model Zoo architectures:
pytest tests/models/ -v

# Verify Frame Parsing and Protocol integrity:
pytest tests/acquisition/test_frame_parser.py -v
```

---

## 4. Key Invariant Checks

### Invariant 1: Immutable Raw Data Storage (`test_raw_store.py`)
```python
def test_immutability_blocks_overwrite(tmp_path):
    store = RawStore(tmp_path)
    with store.open_recording("S001", "Session_01") as writer:
        writer.write_sample(sample)
    # Re-opening an existing session for writing must raise RawStoreError
    with pytest.raises(RawStoreError, match="immutable"):
        store.open_recording("S001", "Session_01")
```

### Invariant 2: Zero Subject Leakage in LOSO (`test_splitter.py`)
```python
def test_loso_no_leakage_per_fold():
    subjects = ["S001", "S002", "S003", "S004"]
    splitter = DatasetSplitter(seed=42)
    folds = splitter.loso_splits(subjects, sessions)
    for fold in folds:
        train = set(fold["train"]["subjects"])
        test = set(fold["test"]["subjects"])
        assert len(train & test) == 0, "Subject leakage detected!"
```

### Invariant 3: Safety Engine Confidence Thresholding (`test_safety_engine.py`)
```python
def test_low_intent_confidence_gives_no_decision():
    cfg = SafetyConfig(enabled=True, trigger_intent="HELP", min_intent_confidence=0.70)
    engine = SafetyEngine(cfg)
    decision = engine.evaluate(intent="HELP", state="Stressed", intent_confidence=0.65, ...)
    assert decision.output == "NO_DECISION"
    assert decision.rule_applied == "intent_confidence_below_threshold"
```
