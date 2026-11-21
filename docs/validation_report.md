# URF/A-SSR Hardware Validation Report v1

**Document:** hardware_validation_report_v1.md  
**Version:** TEMPLATE  
**Date:** 2026-10-08  
**Status:** NOT STARTED — requires physical hardware

---

## Validation Gates

| Gate | Description | Status | Date | Result |
|---|---|---|---|---|
| H1 | Channel mapping verification | NOT STARTED | — | — |
| H2 | Sampling rate measurement | NOT STARTED | — | — |
| H3 | Synchronization verification | NOT STARTED | — | — |
| H4 | Dropped sample detection | NOT STARTED | — | — |
| H5 | Saturation detection | NOT STARTED | — | — |
| H6 | Flatline detection | NOT STARTED | — | — |
| H7 | Noise characterization | NOT STARTED | — | — |
| H8 | Sustained streaming test | NOT STARTED | — | — |

Human data collection must not begin until all gates pass.

---

## H1 — Channel Mapping Verification

**Objective:** Confirm each software channel ID maps to the correct anatomical location.

| Channel | Expected Location | Verified | Method | Result |
|---|---|---|---|---|
| CH1 | Fp1 | [ ] | TBD | TBD |
| CH2 | Fp2 | [ ] | TBD | TBD |
| CH3 | Submental/anterior-digastric | [ ] | TBD | TBD |
| CH4 | Orbicularis oris | [ ] | TBD | TBD |
| CH5 | Zygomaticus major | [ ] | TBD | TBD |
| CH6 | Buccinator | [ ] | TBD | TBD |

---

## H2 — Sampling Rate Measurement

**Objective:** Measure actual samples/second/channel from counter/timestamps.

| Parameter | Target | Measured | Method | Acceptable |
|---|---|---|---|---|
| Sample rate (SPS) | 4,000 | TBD | Sample counter | TBD |
| Sample interval (µs) | 250 | TBD | Timestamp diff | TBD |
| Rate stability | ±TBD% | TBD | Jitter analysis | TBD |

**Do not assume the configured rate equals the measured rate.**

---

## H3 — Synchronization Verification

**Objective:** Confirm all six channels share one acquisition timeline.

Result: TBD

---

## H4 — Dropped Sample Detection

**Objective:** Compare expected and actual sample counts; detect counter gaps.

| Test Duration (s) | Expected Samples | Actual Samples | Gaps Detected | Result |
|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD |

---

## H5 — Saturation Detection

**Objective:** Detect repeated ADC limit hits.

| Channel | Saturation Level | Events in Test | Result |
|---|---|---|---|
| CH1 | TBD | TBD | TBD |
| CH2 | TBD | TBD | TBD |
| CH3 | TBD | TBD | TBD |
| CH4 | TBD | TBD | TBD |
| CH5 | TBD | TBD | TBD |
| CH6 | TBD | TBD | TBD |

---

## H6 — Flatline Detection

**Objective:** Detect suspiciously low-variance channels.

Result: TBD

---

## H7 — Noise Characterization

**Objective:** Test electronics-only, contact/electrode, and controlled-signal conditions.

| Condition | RMS Noise | Notes |
|---|---|---|
| Electronics-only (no electrodes) | TBD | TBD |
| Contact/electrode (no signal) | TBD | TBD |
| Controlled signal injection | TBD | TBD |

---

## H8 — Sustained Streaming Test

**Objective:** Stress test long enough to expose buffer overflow, transport loss, memory leaks, firmware instability.

| Duration | Dropped Samples | Buffer Overflows | Firmware Restarts | Result |
|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD |

---

## Human Safety Review

| Item | Status | Date | Reviewer |
|---|---|---|---|
| Electrical safety review | NOT COMPLETED | — | — |
| Experimental safety review | NOT COMPLETED | — | — |
| `human_use_ready` set to true | NO | — | — |

---

## Hardware Identity (to be completed)

| Parameter | Value |
|---|---|
| AFE part number | TBD |
| MCU module | TBD |
| PCB revision | TBD |
| Firmware revision | TBD |
