# API Contract & Wire Protocol Specification
## Multimodal Assistive Silent Speech & Affective State Recognition (URF/A-SSR)

**Document Version:** 1.0  
**Date:** 2026-11-20  

---

## 1. Binary Wire Packet Specification (42 Bytes)

This specification defines the hardware-to-software transport contract for synchronized 6-channel bio-potential streaming from the acquisition front-end (AFE) microcontroller to the host pipeline.

```
Byte Offset:   0   1   2   3   4   5   6           13  14                  37  38  39  40  41
             ┌───┬───┬───┬───┬───┬───┬───────────────┬───────────────────────┬───┬───┬───┬───┐
Field:       │HDR│HDR│  sample_id    │ timestamp_us  │  CH1 - CH6 ADC Data   │EVT│STS│FTR│FTR│
             └───┴───┴───┴───┴───┴───┴───────────────┴───────────────────────┴───┴───┴───┴───┘
Size (bytes):  1   1       4                 8                   24             1   1   1   1
```

### Frame Field Layout

| Offset | Field Name | Data Type | Encoding | Description |
|---|---|---|---|---|
| `0..1` | Header | `uint16` | Big-Endian (`0xAA 0xBB`) | Frame synchronization preamble |
| `2..5` | `sample_id` | `uint32` | Little-Endian | Rolling monotonic sample counter. Gap $>1$ indicates dropped frame. |
| `6..13` | `timestamp_us`| `uint64` | Little-Endian | Hardware microsecond timer value since session onset |
| `14..17`| `EEG_Fp1` | `int32` | Little-Endian | 24-bit sign-extended ADC count from Channel 1 |
| `18..21`| `EEG_Fp2` | `int32` | Little-Endian | 24-bit sign-extended ADC count from Channel 2 |
| `22..25`| `EMG_Submental`| `int32`| Little-Endian | 24-bit sign-extended ADC count from Channel 3 |
| `26..29`| `EMG_Orbicularis`| `int32`| Little-Endian| 24-bit sign-extended ADC count from Channel 4 |
| `30..33`| `EMG_Zygomaticus`| `int32`| Little-Endian| 24-bit sign-extended ADC count from Channel 5 |
| `34..37`| `EMG_Buccinator`| `int32` | Little-Endian| 24-bit sign-extended ADC count from Channel 6 |
| `38` | `event_marker`| `uint8` | Byte value | Experimental or hardware stimulus event code |
| `39` | `status` | `uint8` | Bitfield | Lead-off and ADC saturation diagnostic status |
| `40..41`| Footer | `uint16` | Big-Endian (`0xCC 0xDD`) | Frame termination delimiter |

### Status Bitfield Mapping (`status`)

| Bit Position | Mask | Flag Name | Description (1 = Active Fault, 0 = Normal) |
|---|---|---|---|
| Bit 0 | `0x01` | `CH1_LEAD_OFF` | High impedance / lead disconnection on EEG Fp1 |
| Bit 1 | `0x02` | `CH2_LEAD_OFF` | High impedance / lead disconnection on EEG Fp2 |
| Bit 2 | `0x04` | `CH3_LEAD_OFF` | Lead disconnection on Submental sEMG |
| Bit 3 | `0x08` | `CH4_LEAD_OFF` | Lead disconnection on Orbicularis Oris sEMG |
| Bit 4 | `0x10` | `CH5_LEAD_OFF` | Lead disconnection on Zygomaticus Major sEMG |
| Bit 5 | `0x20` | `CH6_LEAD_OFF` | Lead disconnection on Buccinator sEMG |
| Bit 6 | `0x40` | `SATURATION` | ADC rail limit reached on one or more channels |
| Bit 7 | `0x80` | `RESERVED` | Reserved for future hardware diagnostic flags |

### Event Marker Codes (`event_marker`)

| Code (Hex) | Enumeration Name | Description |
|---|---|---|
| `0x00` | `EVENT_NONE` | Continuous streaming baseline |
| `0x01` | `SESSION_START` | Session initialization event |
| `0x02` | `BASELINE_START` | Physiological resting baseline recording begin |
| `0x03` | `BASELINE_END` | Physiological resting baseline recording end |
| `0x04` | `TRIAL_START` | Beginning of experimental trial |
| `0x05` | `CUE_ONSET` | Visual / auditory stimulus presentation onset |
| `0x06` | `ARTICULATION_START` | Participant initiates silent speech articulation |
| `0x07` | `ARTICULATION_END` | Participant completes silent articulation |
| `0x08` | `RECOVERY_START` | Post-trial recovery period start |
| `0x09` | `TRIAL_END` | Trial completion delimiter |
| `0x0A` | `SESSION_END` | End of experimental recording session |

---

## 2. Standardized `raw_signals.csv` Schema

Every processed and recorded session folder contains `raw_signals.csv` matching this exact schema:

```csv
sample_index,host_timestamp,EEG_Fp1,EEG_Fp2,EMG_submental,EMG_orbicularis,EMG_zygomaticus,EMG_buccinator
0,0.000000,4.0521,3.0533,0.0000,0.0000,0.0000,0.0000
1,0.000500,4.1686,3.2392,0.0000,0.0000,0.0000,0.0000
...
```

### Column Specifications

| Column Name | Type | Unit | Range / Constraints |
|---|---|---|---|
| `sample_index` | `int64` | count | Monotonically increasing from 0, no negative values |
| `host_timestamp` | `float64` | seconds | Wall-clock elapsed time with microsecond precision |
| `EEG_Fp1` | `float64` | $\mu\text{V}$ | Prefrontal Left EEG potential |
| `EEG_Fp2` | `float64` | $\mu\text{V}$ | Prefrontal Right EEG potential |
| `EMG_submental` | `float64` | $\mu\text{V}$ | Submental sEMG potential |
| `EMG_orbicularis` | `float64` | $\mu\text{V}$ | Orbicularis Oris sEMG potential |
| `EMG_zygomaticus` | `float64` | $\mu\text{V}$ | Zygomaticus Major sEMG potential |
| `EMG_buccinator` | `float64` | $\mu\text{V}$ | Buccinator sEMG potential |

---

## 3. Directory Hierarchy Contract

```
data/
├── public_datasets/
│   ├── DEAP/
│   │   └── data_preprocessed_python/  # s01.dat ... s32.dat
│   └── Gaddy/
│       ├── emg_data/                  # _emg.npy and _info.json
│       └── text_alignments/
└── processed/
    ├── DEAP/
    │   └── Subject_01/
    │       ├── Session_01/
    │       │   ├── raw_signals.csv
    │       │   ├── trials.json
    │       │   └── metadata.json
    │       └── Session_02/
    │           └── ...
    └── Gaddy/
        └── Subject_01/
            ├── Session_01/
            │   ├── raw_signals.csv
            │   ├── trials.json
            │   └── metadata.json
            └── Session_02/
                └── ...
```
