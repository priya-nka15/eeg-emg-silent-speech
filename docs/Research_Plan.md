# Research Plan & Scientific Evaluation Protocol
## Multimodal Assistive Silent Speech & Affective State Recognition (URF/A-SSR)

**Document Version:** 1.0  
**Date:** 2026-11-20  

---

## 1. Scientific Background & Research Questions

Surface electromyography (sEMG) captures neuromuscular motor unit action potentials generated during silent articulation. Electroencephalography (EEG) records cortical electrical activity reflecting cognitive and emotional states. Prior literature has predominantly treated speech decoding and affective computing as isolated domains. URF/A-SSR pioneers a unified multimodal framework addressing the following core Research Questions (RQs):

- **RQ1 (Subject-Independent Silent Speech Decoding):**  
  *Can non-invasive facial sEMG achieve robust, statistically significant classification of discrete silent speech intents (`YES`, `NO`, `HELP`) across unseen subjects without participant-specific retraining?*
- **RQ2 (Multimodal False-Positive Reduction):**  
  *Does decision-level fusion with prefrontal EEG affective state estimation (`Neutral` vs. `Stressed`) significantly decrease false-positive activations of high-risk emergency commands (`HELP`) compared to unimodal EMG triggers?*
- **RQ3 (Model Topology & Latency Trade-Offs):**  
  *How do lightweight deep sequence models (GRU, CNN-GRU) compare to classical feature-engineered models (RBF-SVM) under strict Leave-One-Subject-Out cross-validation and real-time execution constraints (<50 ms)?*

---

## 2. Evaluation Strategy: Strict Leave-One-Subject-Out (LOSO)

Standard random train/test splits (e.g., shuffling individual windows or trials) in bio-potential processing lead to severe **optimistic bias and data leakage** due to intra-subject correlation, shared sensor placement characteristics, and temporal autocorrelation.

### LOSO Cross-Validation Mathematical Formulation
Given a dataset $\mathcal{D} = \bigcup_{k=1}^{K} \mathcal{S}_k$, where $\mathcal{S}_k$ represents all trials, sessions, and samples from Subject $k$:
For each fold $k \in \{1, 2, \dots, K\}$:
$$\mathcal{D}_{\text{train}}^{(k)} = \bigcup_{j \neq k} \mathcal{S}_j$$
$$\mathcal{D}_{\text{test}}^{(k)} = \mathcal{S}_k$$

- **Leakage Prevention Guarantees:**
  1. $\mathcal{D}_{\text{train}}^{(k)} \cap \mathcal{D}_{\text{test}}^{(k)} = \emptyset$ at the subject level.
  2. All feature scaling parameters ($\mu, \sigma$) and model hyperparameters are fit strictly on $\mathcal{D}_{\text{train}}^{(k)}$.
  3. No temporal or spatial information from $\mathcal{S}_k$ is exposed to the model prior to testing.

---

## 3. Evaluation Metrics

### Primary Metric: Macro-Averaged F1 Score
To ensure equal sensitivity across all vocabulary classes irrespective of sample class imbalance:
$$\text{Macro-F1} = \frac{1}{C} \sum_{c=1}^{C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
where $P_c$ and $R_c$ are the precision and recall for class $c$, and $C$ is the number of classes.

### Safety-Critical Metric: HELP-Recall ($R_{\text{HELP}}$)
For assistive applications, failing to trigger an emergency call is critical:
$$R_{\text{HELP}} = \frac{\text{True Positives}_{\text{HELP}}}{\text{True Positives}_{\text{HELP}} + \text{False Negatives}_{\text{HELP}}}$$

### False Activation Count ($FA_{\text{HELP}}$)
The absolute count and rate of non-help utterances spuriously decoded as `HELP`:
$$FA_{\text{HELP}} = \sum (\hat{y} = \text{HELP} \mid y \neq \text{HELP})$$

---

## 4. Benchmark Results on Adapted Public Datasets

The models were evaluated under strict LOSO cross-validation across 6 independent subjects (370 EMG trials, 240 EEG trials).

### EMG Silent Speech Branch (Adapted Gaddy Benchmark, 3-Class)

| Architecture | Input Representation | Macro-F1 | Accuracy | Precision | Recall | HELP-Recall |
|---|---|---|---|---|---|---|
| **SVM (RBF)** | 16D Feature Vector (RMS, MAV, WL, ZCR) | **0.4525** | 0.4486 | 0.4551 | 0.4527 | 0.4800 |
| **GRU** | 250 × 4 Temporal Sequence | 0.3475 | 0.3757 | 0.3547 | 0.3587 | **0.5040** |
| **CNN-GRU** | 250 × 4 Temporal Sequence | 0.4110 | **0.4838** | **0.4681** | 0.4438 | 0.4000 |

### EEG Affective Branch (Adapted DEAP Benchmark, 2-Class)

| Architecture | Input Representation | Macro-F1 | Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| **SVM (RBF)** | 12D Band-Power Features | 0.3617 | **0.5667** | 0.2833 | 0.5000 |
| **Small 1D-CNN** | 250 × 2 Temporal Sequence | **0.4646** | 0.5333 | **0.4916** | 0.4943 |

### Deterministic Safety Engine Verification
- **Validation Outcome:** 9 / 9 test scenarios PASSED (100% compliance).
- Confirmed that whenever intent or state confidence falls below the validated **0.70 threshold**, the engine deterministically returns `NO_DECISION`, suppressing false alarm propagation.
