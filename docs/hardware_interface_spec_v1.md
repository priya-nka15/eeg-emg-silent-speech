# URF/A-SSR Hardware-Software Interface Specification v1

**Document version:** 1  
**Date:** 2026-10-08  
**Status:** DRAFT — hardware-dependent fields are TBD

All TBD fields must be filled from verified hardware documentation.
Do not invent values. Record updates in `docs/decision_log.md`.

---

## 1. Hardware Identity

| Parameter | Value | Status |
|---|---|---|
| AFE part number | TBD | HARDWARE-TBD |
| AFE family | ADS1298-class | EVIDENCE-BACKED |
| AFE channels | 8 (6 used) | EVIDENCE-BACKED |
| AFE resolution | 24-bit | EVIDENCE-BACKED |
| AFE acquisition mode | Simultaneous | EVIDENCE-BACKED |
| MCU | ESP32-S3 | FROZEN |
| MCU module/variant | TBD | HARDWARE-TBD |
| PCB revision | TBD | HARDWARE-TBD |
| Firmware revision | TBD | HARDWARE-TBD |

---

## 2. Channel Mapping

| Software Channel | Modality | Anatomical Target | AFE Input | IN+ | IN− | Status |
|---|---|---|---|---|---|---|
| CH1 (EEG_Fp1) | EEG | Fp1 | TBD | TBD | TBD | HARDWARE-TBD |
| CH2 (EEG_Fp2) | EEG | Fp2 | TBD | TBD | TBD | HARDWARE-TBD |
| CH3 (EMG_Submental) | EMG | Submental/anterior-digastric | TBD | TBD | TBD | HARDWARE-TBD |
| CH4 (EMG_Orbicularis) | EMG | Orbicularis oris | TBD | TBD | TBD | HARDWARE-TBD |
| CH5 (EMG_Zygomaticus) | EMG | Zygomaticus major | TBD | TBD | TBD | HARDWARE-TBD |
| CH6 (EMG_Buccinator) | EMG | Buccinator | TBD | TBD | TBD | HARDWARE-TBD |

**Note:** IN+/IN− assignments must come from the actual PCB schematic.
Do not infer that a differential pair means two electrodes physically at the same named 10-20 location.

---

## 3. Reference Electrodes

| Reference | Role | Physical Location | Status |
|---|---|---|---|
| EEG reference | Common EEG reference | TBD | HARDWARE-TBD |
| EMG reference | Common EMG reference | TBD | HARDWARE-TBD |

**Note:** The EEG reference electrode is NOT automatically a motion-artifact reference for LMS/NLMS.

---

## 4. Sampling

| Parameter | Value | Status |
|---|---|---|
| Target sample rate | 4,000 SPS/channel | PROJECT-TARGET |
| Sample interval | 250 µs | Derived from 4 kSPS |
| Simultaneous sampling | Yes (all 6 channels) | FROZEN |
| Measured sample rate | TBD | HARDWARE-TBD — must be verified by counter |
| ADC resolution | 24-bit | EVIDENCE-BACKED |

---

## 5. Gain

| Channel | Gain | Status |
|---|---|---|
| EEG (CH1, CH2) | TBD | HARDWARE-TBD |
| EMG (CH3–CH6) | TBD | HARDWARE-TBD |

---

## 6. Transport Protocol

| Parameter | Value | Status |
|---|---|---|
| Interface | TBD | HARDWARE-TBD |
| Protocol | TBD | HARDWARE-TBD |
| Baud rate / speed | TBD | HARDWARE-TBD |
| Frame format version | TBD | HARDWARE-TBD |

---

## 7. Frame / Sample Format

The preferred synchronized sample schema:

```
Field           Type        Notes
─────────────────────────────────────────────────────────────
sample_id       uint32      Rolling counter; gap = dropped sample
timestamp       uint64      Microseconds since session start (TBD epoch)
EEG_Fp1         int32       24-bit signed ADC value (TBD scaling)
EEG_Fp2         int32       24-bit signed ADC value (TBD scaling)
EMG_Submental   int32       24-bit signed ADC value (TBD scaling)
EMG_Orbicularis int32       24-bit signed ADC value (TBD scaling)
EMG_Zygomaticus int32       24-bit signed ADC value (TBD scaling)
EMG_Buccinator  int32       24-bit signed ADC value (TBD scaling)
event_marker    uint8       0 = no event; see event table
status          uint8       Bitfield: lead-off, saturation, etc.
─────────────────────────────────────────────────────────────
```

**Minimum required fields:** sample_id, timestamp, all 6 signal channels.  
**Exact wire encoding (endianness, packing):** TBD from firmware documentation.

---

## 8. Sample Counter / Timing

| Parameter | Notes | Status |
|---|---|---|
| Counter field | sample_id uint32 | IMPLEMENTED |
| Counter behavior | Monotonically increasing; wrap = gap | IMPLEMENTED |
| Timestamp epoch | TBD | HARDWARE-TBD |
| Timestamp resolution | TBD | HARDWARE-TBD |
| PC-side timestamp | Applied on receive as fallback | IMPLEMENTED |

---

## 9. Event Marker Mechanism

| Event | Code | Notes |
|---|---|---|
| SESSION_START | 0x01 | — |
| BASELINE_START | 0x02 | — |
| BASELINE_END | 0x03 | — |
| TRIAL_START | 0x04 | — |
| CUE | 0x05 | — |
| ARTICULATION_START | 0x06 | — |
| ARTICULATION_END | 0x07 | — |
| RECOVERY_START | 0x08 | — |
| TRIAL_END | 0x09 | — |
| SESSION_END | 0x0A | — |
| STATE_TASK_START | 0x0B | — |
| STATE_TASK_END | 0x0C | — |
| STATE_RECOVERY_START | 0x0D | — |
| STATE_RECOVERY_END | 0x0E | — |
| (none) | 0x00 | No event |

Software-injected events and hardware-injected events must both reference sample_id for synchronization.

---

## 10. Status / Lead-Off Bitfield

| Bit | Meaning | Status |
|---|---|---|
| 0 | CH1 lead-off | HARDWARE-TBD |
| 1 | CH2 lead-off | HARDWARE-TBD |
| 2 | CH3 lead-off | HARDWARE-TBD |
| 3 | CH4 lead-off | HARDWARE-TBD |
| 4 | CH5 lead-off | HARDWARE-TBD |
| 5 | CH6 lead-off | HARDWARE-TBD |
| 6 | Saturation flag | HARDWARE-TBD |
| 7 | Reserved | — |

Exact bit assignments must be confirmed from firmware documentation.

---

## 11. Hardware Preprocessing

See `configs/hardware_preprocessing.yaml`.  
All filter parameters (bandpass, notch, LMS/NLMS, wavelet) are **TBD**.  
Software preprocessing must not blindly repeat hardware filters.

---

## 12. Data Representation

| Parameter | Value | Status |
|---|---|---|
| Raw ADC units | TBD | HARDWARE-TBD |
| Voltage conversion factor | TBD | HARDWARE-TBD |
| Output representation | TBD (raw or filtered) | HARDWARE-TBD |

---

## 13. Discrepancy Protocol

If any value in this document conflicts with a verified hardware document:
1. The verified hardware document is authoritative.
2. Record the discrepancy in `docs/decision_log.md`.
3. Update this document.
4. Do not silently rewrite assumptions.
