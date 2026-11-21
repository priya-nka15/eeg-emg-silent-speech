# Dataset Adaptation Log & Benchmark Manifest
## Multimodal Assistive Silent Speech & Affective State Recognition (URF/A-SSR)

**Document Version:** 1.0  
**Date:** 2026-11-20  

---

## 1. Overview

To validate the multimodal URF/A-SSR software architecture prior to physical patient trials, the pipeline incorporates two major open-access biosignal benchmarks:
1. **DEAP:** A Dataset for Emotion Analysis using Physiological Signals (Queen Mary University of London, University of Geneva, EPFL).
2. **Gaddy Silent Speech Dataset:** Digital Voicing of Silent Speech sEMG dataset (UC Berkeley).

The adaptation script `scripts/adapt_datasets.py` converts both datasets into the unified `raw_signals.csv` schema and stores them in `data/processed/`.

---

## 2. DEAP Dataset Adaptation (EEG Branch)

### Source Dataset Characteristics
- **Participants:** 32 healthy subjects.
- **Trial Design:** 40 music video stimuli per subject. Each trial is 63 seconds long (3-second baseline + 60-second video stimulus).
- **Sampling Rate:** 128 Hz downsampled preprocessed data.
- **Channels:** 40 channels (first 32 are 10-20 EEG leads; channels 0 and 1 correspond to Fp1 and Fp2).
- **Labels:** Continuous self-assessment ratings for Valence, Arousal, Dominance, and Liking on a continuous 1.0 to 9.0 scale.

### Processing & Transformation Protocol
1. **Channel Selection:** Strictly extracted channel index 0 (`Fp1`) and channel index 1 (`Fp2`). All other 38 channels discarded to mirror the target wearable headband topology.
2. **Temporal Windowing:** A 10.0-second active stimulus segment was extracted from each trial, starting at $t=3.0\text{ s}$ to isolate active cognitive/affective response from the initial rest baseline.
3. **Upsampling via FFT Interpolation:**
   - Input length: $10\text{ s} \times 128\text{ Hz} = 1,280\text{ samples}$.
   - Target sampling frequency: $2,000\text{ Hz}$.
   - Output length: $10\text{ s} \times 2,000\text{ Hz} = 20,000\text{ samples}$ per trial.
   - Performed using `scipy.signal.resample` to preserve spectral harmonics and phase fidelity.
4. **Affective State Binarization:**
   - Valence threshold: Midpoint $5.0$.
   - **`Stressed`:** Valence $< 5.0$ (negative affective response).
   - **`Neutral`:** Valence $\ge 5.0$ (calm / positive affective response).
5. **Channel Nulling:** EMG columns (`EMG_submental`, `EMG_orbicularis`, `EMG_zygomaticus`, `EMG_buccinator`) populated with `0.0000`.

### Summary Statistics
- **Subjects Processed:** 6 subjects (`Subject_01` to `Subject_06`).
- **Sessions Generated:** 12 sessions (2 sessions per subject, 20 trials per session).
- **Total Valid Trials:** 240 trials.
- **Class Breakdown:**
  - `Neutral`: 136 trials (56.7%)
  - `Stressed`: 104 trials (43.3%)

---

## 3. Gaddy Silent Speech Dataset Adaptation (sEMG Branch)

### Source Dataset Characteristics
- **Dataset:** Gaddy & Klein (Interspeech 2020), open-access silent and vocalized speech electromyography recordings.
- **Sampling Rate:** 1,000 Hz.
- **Channels:** 8 surface EMG electrodes placed over facial and articulatory musculature.
- **Modalities:** Parallel and nonparallel speech, books, and interactive prompt reading.

### Processing & Transformation Protocol
1. **Channel Selection & Anatomical Mapping:**
   - First 4 channels mapped directly to the URF/A-SSR articulatory configuration:
     - Channel 0 $\rightarrow$ `EMG_submental`
     - Channel 1 $\rightarrow$ `EMG_orbicularis`
     - Channel 2 $\rightarrow$ `EMG_zygomaticus`
     - Channel 3 $\rightarrow$ `EMG_buccinator`
   - Channels 4-7 discarded.
2. **Resampling:** Resampled from 1000 Hz to 2000 Hz using `scipy.signal.resample` to match the 2000 SPS pipeline standard.
3. **Vocabulary Mapping & Keyword Extraction:**
   - Analyzed utterance transcripts from `_info.json` files.
   - Priority 1: Utterances containing `"yes"` mapped to **`YES`**.
   - Priority 2: Utterances containing assistance keywords (`"help"`, `"stop"`, `"please"`, `"urgent"`, `"care"`, `"pain"`, `"doctor"`, `"nurse"`, `"assist"`, `"call"`, `"danger"`, `"trouble"`, `"emergency"`) mapped to **`HELP`**.
   - Priority 3: Utterances containing negation (`"no"`, `"not"`, `"never"`) mapped to **`NO`**.
4. **Subject & Session Partitioning:**
   - Grouped 12 distinct recording sessions into 6 independent subjects with 2 sessions each:
     - `Subject_01`: `4-21` & `4-23`
     - `Subject_02`: `4-24` & `4-25`
     - `Subject_03`: `4-26` & `4-27`
     - `Subject_04`: `4-29` & `4-30`
     - `Subject_05`: `5-2` & `4-25_2`
     - `Subject_06`: `5-10_silent` & `5-5_silent`
5. **Channel Nulling:** EEG columns (`EEG_Fp1`, `EEG_Fp2`) populated with `0.0000`.

### Summary Statistics
- **Subjects Processed:** 6 subjects (`Subject_01` to `Subject_06`).
- **Sessions Generated:** 12 sessions.
- **Total Valid Trials:** 370 trials.
- **Class Breakdown:**
  - `YES`: 101 trials (27.3%)
  - `NO`: 144 trials (38.9%)
  - `HELP`: 125 trials (33.8%)

---

## 4. Adaptation Manifest File

A permanent machine-readable summary is maintained at `data/processed/adaptation_summary.json` containing:
- Exact source directories and version timestamps.
- Target sampling rate (2000 Hz).
- Trial counts and class distributions per modality.
- Column header order for schema verification.
