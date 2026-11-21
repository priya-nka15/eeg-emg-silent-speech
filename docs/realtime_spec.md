# URF/A-SSR Real-Time System Specification

**Version:** 1.0  
**Date:** 2026-10-08

---

## Pipeline Architecture

```
PCB
 ↓
ESP32-S3
 ↓
stream (protocol: TBD)
 ↓
Python receiver
 ↓
buffer
 ↓
QC (per-sample, per-window)
 ↓
EEG/EMG preprocessing
 ↓
windowing
 ↓
selected models (EMG: TBD, EEG: TBD)
 ↓
probabilities
 ↓
fusion
 ↓
temporal consistency
 ↓
safety engine
 ↓
GUI / output
```

ML inference runs on the PC initially. ESP32 deployment is not a prerequisite for the first complete real-time demonstration.

---

## Operating Modes

### Mock Mode
- No hardware required
- Uses `backend/realtime/mock_source.py`
- All data explicitly labelled `MOCK_DATA`
- Mock data must NEVER enter official research results
- Used for UI development and software testing

### Replay Mode
- Loads a recorded raw session
- Runs the exact production pipeline (no hardware)
- Mandatory for reproducibility and debugging
- Every replay of the same recording must produce the same output

### Live Mode
- Uses actual hardware streaming
- Requires `human_use_ready: true`
- Requires SELECTED models in model registry

---

## UI Status Display (Live Mode)

The UI must display in real time:
- Connection status
- Six channel status (each: OK / lead-off / fault)
- Sample rate (measured)
- Dropped sample count
- Recording state (IDLE / RECORDING / etc.)
- Current event
- EMG intent (YES / NO / HELP / NO_DECISION)
- EEG state (TBD labels / NO_DECISION)
- Confidence (per modality)
- Fusion result
- Safety state
- NO_DECISION state and reason

---

## Real-Time Measurements

Measure, do not assume. Do not invent a latency target before the complete pipeline has been measured.

### Acquisition
- Sample rate (measured from counter)
- Dropped sample count
- Packet loss rate
- Buffer overflow events

### Processing
- Preprocessing latency (ms)
- Feature extraction latency (ms)
- Model inference latency (ms per model)
- Fusion latency (ms)

### End-to-End
- Acquisition event to output latency (ms)

### Quality
- False activations
- Command stability (temporal consistency)
- HELP recall in real-time conditions
- NO_DECISION rate

---

## Buffer Management

- Input buffer: configurable size (TBD from latency/memory trade-off)
- Windowing stride determines how often inference runs
- Buffer overflow must be detected and reported, not silently ignored

---

## Error Handling

- Transport errors → NO_DECISION, log error
- Missing samples → flag in QC, may trigger NO_DECISION
- Model unavailable → NO_DECISION
- Safety engine fault → NO_DECISION, alert

---

## Reproducibility via Replay

The replay mode must:
1. Load raw session recording
2. Apply identical preprocessing, windowing, and inference
3. Produce the same output as the original live run
4. Allow timestamped comparison of live vs replay results

Any divergence between live and replay is a bug.
