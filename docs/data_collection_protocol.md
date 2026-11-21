# URF/A-SSR Data Collection Protocol

**Version:** 1.0  
**Date:** 2026-10-08  
**Status:** DRAFT — exact durations TBD from pilot

---

## Safety Requirements

**Human use requires `human_use_ready = true`.**  
The hardware team/institution must complete electrical and experimental safety review before any human participant session.  
This system is a research prototype. It is not a medical device.

---

## Pilot Phase (Gate 5)

**Target:** 2–3 participants  
**Purpose:** Engineering and protocol validation, NOT final performance evaluation

### EMG Pilot
- ~20–30 trials per class (YES, NO, HELP) per session
- Assess: channel usability, contact stability, silent articulation consistency, command distinguishability, cross-talk

### EEG Pilot
- Assess Fp1/Fp2 usability
- Assess facial/ocular/movement contamination
- Compare: rest | cognitive task without articulation | silent articulation | cognitive + articulation
- Candidate conditions (not frozen): rest/baseline, focused attention, cognitive workload levels, positive affect, controlled stress-related condition

**Do not freeze EEG labels before the pilot.**

---

## Trial Protocol

Each trial follows this sequence:

```
REST (duration: TBD from pilot)
 ↓
CUE (command displayed to participant)
 ↓
SILENT ARTICULATION (duration: TBD from pilot)
 ↓
REST / RECOVERY (duration: TBD from pilot)
```

**Commands:** YES | NO | HELP  
**Exact durations:** TBD — must be determined during pilot, then frozen (Gate 6)  
**Do not copy another paper's trial timing without justification.**

---

## Session Structure

1. **Consent and briefing**
2. **Electrode placement** (record actual placement per `electrode_map.md`)
3. **Impedance check** (TBD threshold from hardware team)
4. **Baseline recording** (rest, eyes open; duration TBD from pilot)
5. **Practice trials** (not included in analysis)
6. **Experimental blocks** (randomized trial sequence)
7. **Break(s)** (as needed)
8. **Debrief and notes**

---

## Randomization

- Do NOT record all YES then all NO then all HELP.
- Generate a reproducible block-randomized or fully randomized sequence.
- Save the sequence in `metadata/randomization_sequence.json`.
- Use a fixed seed recorded in session metadata for reproducibility.

---

## Main Dataset Targets (Gate 7)

| Parameter | Target | Status |
|---|---|---|
| Participants | 12 | PROJECT-TARGET |
| Sessions per participant | 2 | FROZEN |
| Commands | YES, NO, HELP | FROZEN |
| Valid trials per class per session | ≥100 | PROJECT-TARGET |
| Total valid EMG trials | ≥7,200 | Derived |
| Collection allowance | ~110–120 attempts/class/session | Working target |

**Two sessions required** to test session variation from electrode repositioning and participant return.

---

## Trial Validity

A trial is marked INVALID if any of the following apply:
- Participant vocalized
- Wrong command performed
- Excessive movement artifact
- Swallowing during trial
- Coughing during trial
- Electrode detachment
- Poor electrode contact
- Signal saturation
- Sample counter gaps
- Transmission failure
- Timing error in event markers
- External interruption
- Protocol violation

**NEVER silently delete invalid trials. Preserve original signal. Record reason.**

---

## Operator Responsibilities

- Record actual electrode placement for every session.
- Record any anomalies in session notes.
- Verify channel mapping at session start (H1).
- Verify measured sample rate (H2).
- Monitor signal quality in real time.
- Mark any trials requiring validity review.

---

## Post-Session

- Verify raw recording is complete.
- Run QC report.
- Back up raw data.
- Do NOT overwrite any raw data files.
