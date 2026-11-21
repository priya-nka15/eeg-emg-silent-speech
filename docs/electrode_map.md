# URF/A-SSR Electrode Map

**Version:** 1  
**Date:** 2026-10-08  
**Status:** DRAFT — IN+/IN− wiring is TBD until PCB schematic is supplied

Do not invent electrode wiring. Do not infer that a differential pair means two electrodes at the same named 10-20 location.

---

## Electrode Summary

| # | Role | Anatomical Target / Location | Status |
|---|---|---|---|
| E01 | CH1 IN+ | TBD — Fp1 region, exact position from schematic | HARDWARE-TBD |
| E02 | CH1 IN− | TBD — from schematic | HARDWARE-TBD |
| E03 | CH2 IN+ | TBD — Fp2 region, exact position from schematic | HARDWARE-TBD |
| E04 | CH2 IN− | TBD — from schematic | HARDWARE-TBD |
| E05 | CH3 IN+ | TBD — Submental/anterior-digastric, from schematic | HARDWARE-TBD |
| E06 | CH3 IN− | TBD — from schematic | HARDWARE-TBD |
| E07 | CH4 IN+ | TBD — Orbicularis oris, from schematic | HARDWARE-TBD |
| E08 | CH4 IN− | TBD — from schematic | HARDWARE-TBD |
| E09 | CH5 IN+ | TBD — Zygomaticus major, from schematic | HARDWARE-TBD |
| E10 | CH5 IN− | TBD — from schematic | HARDWARE-TBD |
| E11 | CH6 IN+ | TBD — Buccinator, from schematic | HARDWARE-TBD |
| E12 | CH6 IN− | TBD — from schematic | HARDWARE-TBD |
| E13 | EEG reference | TBD | HARDWARE-TBD |
| E14 | EMG reference | TBD | HARDWARE-TBD |

**Total: 14 electrodes** (12 signal + 1 EEG ref + 1 EMG ref)

---

## Channel Details

### CH1 — EEG Fp1
- **Modality:** EEG
- **Anatomical target:** Fp1 (international 10-20 system)
- **IN+ physical location:** TBD from schematic
- **IN− physical location:** TBD from schematic
- **Reference:** EEG reference (E13)
- **Notes:** Frontal EEG site; susceptible to ocular and facial EMG artifact. See PRD §22.

### CH2 — EEG Fp2
- **Modality:** EEG
- **Anatomical target:** Fp2 (international 10-20 system)
- **IN+ physical location:** TBD from schematic
- **IN− physical location:** TBD from schematic
- **Reference:** EEG reference (E13)
- **Notes:** Frontal EEG site; susceptible to ocular and facial EMG artifact. See PRD §22.

### CH3 — EMG Submental
- **Modality:** EMG
- **Anatomical target:** Submental region / anterior-digastric muscle
- **IN+ physical location:** TBD from schematic
- **IN− physical location:** TBD from schematic
- **Reference:** EMG reference (E14)
- **Notes:** Pilot must determine usability and stability.

### CH4 — EMG Orbicularis Oris
- **Modality:** EMG
- **Anatomical target:** Orbicularis oris muscle (lips)
- **IN+ physical location:** TBD from schematic
- **IN− physical location:** TBD from schematic
- **Reference:** EMG reference (E14)
- **Notes:** Pilot must determine usability and stability.

### CH5 — EMG Zygomaticus Major
- **Modality:** EMG
- **Anatomical target:** Zygomaticus major muscle (cheek/smile)
- **IN+ physical location:** TBD from schematic
- **IN− physical location:** TBD from schematic
- **Reference:** EMG reference (E14)
- **Notes:** Pilot must determine usability and stability.

### CH6 — EMG Buccinator
- **Modality:** EMG
- **Anatomical target:** Buccinator muscle (cheek)
- **IN+ physical location:** TBD from schematic
- **IN− physical location:** TBD from schematic
- **Reference:** EMG reference (E14)
- **Notes:** Pilot must determine usability and stability.

---

## Reference Electrodes

### EEG Reference (E13)
- **Role:** Common reference for EEG channels (CH1, CH2)
- **Physical location:** TBD from schematic
- **Status:** HARDWARE-TBD
- **Important:** This electrode is NOT automatically a motion-artifact reference for LMS/NLMS adaptive filtering.

### EMG Reference (E14)
- **Role:** Common reference for EMG channels (CH3–CH6)
- **Physical location:** TBD from schematic
- **Status:** HARDWARE-TBD

---

## Topology Notes

- **6 differential signal channels**, each requiring 2 signal electrodes.
- **2 common reference electrodes** (1 EEG, 1 EMG).
- **14 physical electrodes** total in the current topology.
- The PCB schematic is the authoritative source for all wiring details.
- If the schematic conflicts with any assumption here, the schematic wins. Record in `decision_log.md`.

---

## Update Instructions

When the PCB schematic becomes available:
1. Fill in all TBD fields in this document.
2. Update `configs/electrodes.yaml` correspondingly.
3. Add a decision log entry (DL-009 superseded).
4. Run hardware validation Gate H1 to verify channel mapping.
