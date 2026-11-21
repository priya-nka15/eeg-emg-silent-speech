# URF/A-SSR Limitations

**Version:** 1.0  
**Date:** 2026-10-08

---

## Important Disclaimers

**This system is a research prototype.**  
It has not been clinically validated and is not a certified medical device.  
It must not be used or described as a clinical tool, diagnostic device, or assistive device without appropriate regulatory approval.

---

## Current Known Limitations

### Hardware
- Exact AFE part number not yet confirmed (TBD).
- Exact MCU module/variant not yet confirmed (TBD).
- IN+/IN− electrode wiring not yet confirmed from schematic (TBD).
- All hardware filter parameters (bandpass, notch, LMS, wavelet) not yet confirmed (TBD).
- Gain (EEG, EMG) not yet confirmed (TBD).
- `human_use_ready` is currently `false` — human participant sessions cannot begin until safety review is completed.

### Signal Acquisition
- Fp1/Fp2 frontal EEG sites are susceptible to ocular artifact, facial EMG contamination, and movement artifact.
- The extent of contamination in the EEG signal during silent articulation has not yet been assessed (requires EEG contamination pilot, §22).
- Actual sample rate under real operating conditions must be verified by counter, not assumed to be exactly 4,000 SPS.

### EMG
- The four EMG channel locations are the current proposed configuration. The pilot must determine whether they are sufficiently useful and stable.
- Silent articulation performance has not yet been measured for any participant.
- Cross-talk between facial EMG channels has not yet been assessed.

### EEG
- EEG task/labels are NOT frozen. Candidate conditions listed in the PRD are candidates only.
- No EEG pilot has been conducted. EEG feasibility for cognitive/affective discrimination at Fp1/Fp2 is unproven for this system.
- Two-channel frontal EEG provides very limited spatial information. ICA is not mandated and may not be appropriate for a 2-channel system.
- EEG generalisation across sessions and subjects may be poor. If this is the case, it will be reported honestly.

### Models
- No models have been trained on real data. All model implementations are untested against actual biosignals.
- Performance numbers from other published work are not transferable to this system (different hardware, participants, protocols).
- EEG model labels cannot be frozen until the pilot is complete.

### Fusion and Safety
- Confidence thresholds for rejection are TBD — not yet validated.
- Temporal consistency window count is TBD — not yet validated.
- Safety engine `trigger_state` is TBD — EEG label is not frozen.
- Safety engine is disabled by default and must not be enabled until thresholds are validated.

### Evaluation
- No data has been collected. No models have been evaluated on real data.
- Final test set has not yet been evaluated.
- Robustness tests (§43) have not yet been conducted.

### Timeline
- All pilot and main dataset collection is pending hardware readiness and safety review.

---

## What This System Does Not Claim

- This system does not claim to be a certified medical device.
- This system does not claim to be clinically safe for all users.
- This system does not claim any specific accuracy performance until trained and evaluated on real data.
- This system does not claim to have invented silent-speech recognition.
- This system does not reproduce or rely on performance numbers from other published work.
