#!/usr/bin/env python3
"""
scripts/run_experiments.py
==========================
URF/A-SSR Experiment Runner with Strict Leave-One-Subject-Out (LOSO) Cross-Validation.

Pipelines evaluated:
  1. EMG Branch on adapted Gaddy dataset:
     - SVM (Feature-based: RMS, MAV, WL, ZCR)
     - GRU (Sequence-based: 4 channels)
     - CNN-GRU (Hybrid temporal-sequential: 4 channels)
  2. EEG Branch on adapted DEAP dataset:
     - SVM (Feature-based: theta, alpha, beta power)
     - Small 1D-CNN (Temporal convolutional: 2 channels)
  3. Safety Engine Verification:
     - Enforces deterministic thresholding (min_confidence = 0.70)
     - Verifies NO_DECISION output whenever model confidence drops below threshold

Metrics calculated and saved to experiments/results.json:
  - macro_f1
  - accuracy
  - precision
  - recall
  - confusion_matrix
  - help_recall (for EMG branch)
  - per-class breakdown & LOSO fold details
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from backend.datasets.splitter import DatasetSplitter
from backend.features.eeg_features import EEGFeatureConfig, EEGFeatureExtractor
from backend.features.emg_features import EMGFeatureConfig, EMGFeatureExtractor
from backend.models.eeg.cnn1d_model import EEGCNN1DModel
from backend.models.eeg.svm_model import EEGSVMModel
from backend.models.emg.cnn_gru_model import EMGCNNGRUModel
from backend.models.emg.gru_model import EMGGRUModel
from backend.models.emg.svm_model import EMGSVMModel
from backend.safety.safety_engine import SafetyConfig, SafetyEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("run_experiments")

EMG_LABELS = ["YES", "NO", "HELP"]
EEG_LABELS = ["Neutral", "Stressed"]

# ─────────────────────────────────────────────────────────────────────────────
# Data Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_emg_dataset(processed_base: Path, subjects: List[str], seq_len: int = 250) -> Dict[str, dict]:
    """
    Load EMG trials for each subject.
    Returns: subject_id -> {"features": np.ndarray, "sequences": np.ndarray, "labels": List[str]}
    """
    extractor = EMGFeatureExtractor(EMGFeatureConfig(
        feature_version="v1",
        sampling_rate_hz=2000.0,
        use_rms=True,
        use_mav=True,
        use_wl=True,
        use_zcr=True,
    ))
    
    gaddy_dir = processed_base / "Gaddy"
    subject_data = {}
    
    for subj in subjects:
        subj_dir = gaddy_dir / subj
        features_list = []
        seq_list = []
        labels_list = []
        
        for sess_dir in sorted(subj_dir.iterdir()):
            if not sess_dir.is_dir():
                continue
            csv_path = sess_dir / "raw_signals.csv"
            trials_path = sess_dir / "trials.json"
            if not csv_path.exists() or not trials_path.exists():
                continue
                
            # Load raw signals
            data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
            # EMG columns are indices 4, 5, 6, 7
            emg_signals = data[:, 4:8]
            
            with open(trials_path) as tf:
                trials = json.load(tf)
                
            for tr in trials:
                start = tr["start_sample"]
                end = tr["end_sample"] + 1
                intent = tr["intent"]
                if intent not in EMG_LABELS:
                    continue
                
                trial_emg = emg_signals[start:end]
                if len(trial_emg) < 10:
                    continue
                    
                # Feature vector
                feat = extractor.extract_window(trial_emg)
                features_list.append(feat)
                
                # Sequence representation: center slice or resample to seq_len
                if len(trial_emg) >= seq_len:
                    mid = len(trial_emg) // 2
                    h = seq_len // 2
                    seq = trial_emg[mid - h:mid + h]
                else:
                    # Pad
                    pad = np.zeros((seq_len - len(trial_emg), 4))
                    seq = np.vstack([trial_emg, pad])
                seq_list.append(seq)
                labels_list.append(intent)
                
        subject_data[subj] = {
            "features": np.array(features_list),
            "sequences": np.array(seq_list),
            "labels": np.array(labels_list),
        }
        logger.info("Loaded EMG %s: %d trials", subj, len(labels_list))
        
    return subject_data


def load_eeg_dataset(processed_base: Path, subjects: List[str], seq_len: int = 250) -> Dict[str, dict]:
    """
    Load EEG trials for each subject.
    Returns: subject_id -> {"features": np.ndarray, "sequences": np.ndarray, "labels": List[str]}
    """
    extractor = EEGFeatureExtractor(EEGFeatureConfig(
        feature_version="v1",
        sampling_rate_hz=2000.0,
        use_theta_power=True,
        use_alpha_power=True,
        use_beta_power=True,
        use_relative_bands=True,
    ))
    
    deap_dir = processed_base / "DEAP"
    subject_data = {}
    
    for subj in subjects:
        subj_dir = deap_dir / subj
        features_list = []
        seq_list = []
        labels_list = []
        
        for sess_dir in sorted(subj_dir.iterdir()):
            if not sess_dir.is_dir():
                continue
            csv_path = sess_dir / "raw_signals.csv"
            trials_path = sess_dir / "trials.json"
            if not csv_path.exists() or not trials_path.exists():
                continue
                
            data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
            # EEG columns are indices 2, 3 (Fp1, Fp2)
            eeg_signals = data[:, 2:4]
            
            with open(trials_path) as tf:
                trials = json.load(tf)
                
            for tr in trials:
                start = tr["start_sample"]
                end = tr["end_sample"] + 1
                state = tr["state"]
                if state not in EEG_LABELS:
                    continue
                
                trial_eeg = eeg_signals[start:end]
                if len(trial_eeg) < 10:
                    continue
                    
                # Feature vector
                feat = extractor.extract_window(trial_eeg)
                features_list.append(feat)
                
                # Sequence representation: center slice or resample to seq_len
                if len(trial_eeg) >= seq_len:
                    mid = len(trial_eeg) // 2
                    h = seq_len // 2
                    seq = trial_eeg[mid - h:mid + h]
                else:
                    pad = np.zeros((seq_len - len(trial_eeg), 2))
                    seq = np.vstack([trial_eeg, pad])
                seq_list.append(seq)
                labels_list.append(state)
                
        subject_data[subj] = {
            "features": np.array(features_list),
            "sequences": np.array(seq_list),
            "labels": np.array(labels_list),
        }
        logger.info("Loaded EEG %s: %d trials", subj, len(labels_list))
        
    return subject_data


# ─────────────────────────────────────────────────────────────────────────────
# Strict LOSO Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_loso(
    subject_data: Dict[str, dict],
    model_type: str,
    modality: str,
    labels: List[str],
) -> dict:
    """
    Run strict Leave-One-Subject-Out (LOSO) cross-validation.
    
    Guarantees:
      - Training set contains only N-1 subjects
      - Held-out subject is strictly used for testing
      - Zero overlap / zero data leakage
    """
    subjects = sorted(subject_data.keys())
    all_y_true = []
    all_y_pred = []
    fold_summaries = []
    
    logger.info("Starting LOSO cross-validation for %s (%s) over %d subjects", model_type, modality, len(subjects))
    
    label_map = {lbl: idx for idx, lbl in enumerate(labels)}
    
    for fold_idx, held_out in enumerate(subjects):
        train_subjs = [s for s in subjects if s != held_out]
        
        # Prepare training data
        if "SVM" in model_type:
            X_train = np.vstack([subject_data[s]["features"] for s in train_subjs])
            X_test = subject_data[held_out]["features"]
        else:
            X_train = np.vstack([subject_data[s]["sequences"] for s in train_subjs])
            X_test = subject_data[held_out]["sequences"]
            
        y_train = np.concatenate([subject_data[s]["labels"] for s in train_subjs])
        y_test = subject_data[held_out]["labels"]
        
        # Instantiate fresh model per fold
        if model_type == "SVM" and modality == "EMG":
            model = EMGSVMModel(label_map=label_map, model_id=f"emg-svm-fold{fold_idx}")
        elif model_type == "GRU" and modality == "EMG":
            model = EMGGRUModel(label_map=label_map, model_id=f"emg-gru-fold{fold_idx}", hidden_size=32, num_layers=1, n_epochs=5, batch_size=32)
        elif model_type == "CNN-GRU" and modality == "EMG":
            model = EMGCNNGRUModel(label_map=label_map, model_id=f"emg-cnn-gru-fold{fold_idx}", cnn_filters=16, gru_hidden_size=32, gru_num_layers=1, n_epochs=5, batch_size=32)
        elif model_type == "SVM" and modality == "EEG":
            model = EEGSVMModel(label_map=label_map, model_id=f"eeg-svm-fold{fold_idx}")
        elif model_type == "Small 1D-CNN" and modality == "EEG":
            model = EEGCNN1DModel(label_map=label_map, model_id=f"eeg-cnn1d-fold{fold_idx}", cnn_filters=16, n_epochs=5, batch_size=32)
        else:
            raise ValueError(f"Unknown model_type={model_type} or modality={modality}")
            
        # Fit on training subjects only
        model.fit(X_train, y_train)
        
        # Predict on held-out subject
        preds = model.predict(X_test)
        pred_labels = [p.label for p in preds]
        
        fold_acc = float(accuracy_score(y_test, pred_labels))
        fold_f1 = float(f1_score(y_test, pred_labels, average="macro", zero_division=0))
        logger.info("  [%s %s] Fold %d/%d (held-out: %s) -> Acc: %.3f, Macro-F1: %.3f", modality, model_type, fold_idx + 1, len(subjects), held_out, fold_acc, fold_f1)
        
        fold_summaries.append({
            "fold": fold_idx + 1,
            "held_out_subject": held_out,
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "accuracy": fold_acc,
            "macro_f1": fold_f1,
        })
        
        all_y_true.extend(y_test)
        all_y_pred.extend(pred_labels)
        
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    # Calculate overall metrics
    macro_f1 = float(f1_score(all_y_true, all_y_pred, average="macro", zero_division=0))
    accuracy = float(accuracy_score(all_y_true, all_y_pred))
    precision = float(precision_score(all_y_true, all_y_pred, average="macro", zero_division=0))
    recall = float(recall_score(all_y_true, all_y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(all_y_true, all_y_pred, labels=labels).tolist()
    
    # Per-class metrics
    per_class = {}
    for lbl in labels:
        lbl_true = (all_y_true == lbl).astype(int)
        lbl_pred = (all_y_pred == lbl).astype(int)
        p = float(precision_score(lbl_true, lbl_pred, zero_division=0))
        r = float(recall_score(lbl_true, lbl_pred, zero_division=0))
        f = float(f1_score(lbl_true, lbl_pred, zero_division=0))
        per_class[lbl] = {"precision": p, "recall": r, "f1": f, "support": int(np.sum(lbl_true))}
        
    result = {
        "modality": modality,
        "model_type": model_type,
        "validation_strategy": "strict_LOSO",
        "n_folds": len(subjects),
        "total_samples": len(all_y_true),
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
        "labels": labels,
        "per_class": per_class,
        "folds": fold_summaries,
    }
    
    # EMG Safety recall
    if modality == "EMG" and "HELP" in per_class:
        help_rec = per_class["HELP"]["recall"]
        result["help_recall"] = help_rec
        # False help activations: when predicted HELP but true was not HELP
        false_help = int(np.sum((all_y_pred == "HELP") & (all_y_true != "HELP")))
        result["false_help_activations"] = false_help
        result["help_support"] = per_class["HELP"]["support"]
        
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Safety Engine Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_safety_engine(threshold: float = 0.70) -> dict:
    """
    Verify SafetyEngine behavior against PRD §37 deterministic safety rules.
    Tests NO_DECISION output when confidence falls below the 0.70 threshold.
    """
    logger.info("Executing SafetyEngine validation tests with confidence threshold %.2f", threshold)
    
    cfg = SafetyConfig(
        enabled=True,
        trigger_intent="HELP",
        trigger_state="Stressed",
        min_intent_confidence=threshold,
        min_state_confidence=threshold,
        temporal_requirement=True,
    )
    engine = SafetyEngine(cfg)
    
    test_cases = [
        # (intent, state, intent_conf, state_conf, signal_qual, temp_met, expected_output, expected_rule)
        ("HELP", "Stressed", 0.85, 0.85, "OK", True, "ACTION", "all_rules_passed"),
        ("HELP", "Stressed", 0.65, 0.85, "OK", True, "NO_DECISION", "intent_confidence_below_threshold"),
        ("HELP", "Stressed", 0.85, 0.50, "OK", True, "NO_DECISION", "state_confidence_below_threshold"),
        ("HELP", "Stressed", 0.60, 0.60, "OK", True, "NO_DECISION", "intent_confidence_below_threshold"),
        ("YES",  "Stressed", 0.95, 0.90, "OK", True, "NO_DECISION", "intent_mismatch"),
        ("NO",   "Stressed", 0.95, 0.90, "OK", True, "NO_DECISION", "intent_mismatch"),
        ("HELP", "Neutral",  0.90, 0.90, "OK", True, "NO_DECISION", "state_mismatch"),
        ("HELP", "Stressed", 0.90, 0.90, "DEGRADED", True, "NO_DECISION", "signal_quality_not_ok"),
        ("HELP", "Stressed", 0.90, 0.90, "OK", False, "NO_DECISION", "temporal_consistency_not_met"),
    ]
    
    passed_cases = 0
    test_log = []
    
    for intent, state, i_conf, s_conf, sq, tm, exp_out, exp_rule in test_cases:
        dec = engine.evaluate(
            intent=intent,
            state=state,
            intent_confidence=i_conf,
            state_confidence=s_conf,
            signal_quality=sq,
            temporal_consistency_met=tm,
        )
        out_str = dec.output.value if hasattr(dec.output, "value") else str(dec.output)
        success = (out_str == exp_out and dec.rule_applied == exp_rule)
        if success:
            passed_cases += 1
        else:
            logger.error("Safety test failed: got %s (%s), expected %s (%s)", out_str, dec.rule_applied, exp_out, exp_rule)
            
        test_log.append({
            "intent": intent,
            "state": state,
            "intent_confidence": i_conf,
            "state_confidence": s_conf,
            "actual_output": out_str,
            "expected_output": exp_out,
            "rule_applied": dec.rule_applied,
            "passed": success,
        })
        
    assert passed_cases == len(test_cases), f"SafetyEngine validation failed: {passed_cases}/{len(test_cases)} passed"
    logger.info("All %d SafetyEngine validation tests passed successfully!", passed_cases)
    
    return {
        "status": "PASSED",
        "confidence_threshold": threshold,
        "total_tests": len(test_cases),
        "passed_tests": passed_cases,
        "no_decision_below_threshold_verified": True,
        "test_log": test_log,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Final Model Training and Persistence
# ─────────────────────────────────────────────────────────────────────────────

def train_and_save_final_models(emg_data: Dict[str, dict], eeg_data: Dict[str, dict], models_dir: Path) -> dict:
    """
    Train production models on the complete adapted datasets across all subjects
    and persist their weights to models/ in both directory bundles and direct weight files.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Training and saving production models to %s...", models_dir)

    # 1. Pool EMG Data
    emg_X_feat = np.vstack([d["features"] for d in emg_data.values()])
    emg_X_seq = np.vstack([d["sequences"] for d in emg_data.values()])
    emg_y = np.concatenate([d["labels"] for d in emg_data.values()])
    emg_label_map = {lbl: idx for idx, lbl in enumerate(EMG_LABELS)}

    logger.info("Pooled EMG dataset: %d samples, classes=%s", len(emg_y), EMG_LABELS)

    # EMG SVM
    logger.info("Training production EMG SVM...")
    emg_svm = EMGSVMModel(label_map=emg_label_map, model_id="emg-svm-prod")
    emg_svm.fit(emg_X_feat, emg_y)
    emg_svm_dir = models_dir / "emg_svm"
    emg_svm.save(emg_svm_dir)
    shutil.copy2(emg_svm_dir / "model.pkl", models_dir / "emg_svm.pkl")
    logger.info("Saved EMG SVM to %s and %s", emg_svm_dir, models_dir / "emg_svm.pkl")

    # EMG GRU
    logger.info("Training production EMG GRU...")
    emg_gru = EMGGRUModel(label_map=emg_label_map, model_id="emg-gru-prod", hidden_size=32, num_layers=1, n_epochs=6, batch_size=32)
    emg_gru.fit(emg_X_seq, emg_y)
    emg_gru_dir = models_dir / "emg_gru"
    emg_gru.save(emg_gru_dir)
    shutil.copy2(emg_gru_dir / "model.pt", models_dir / "emg_gru.pt")
    logger.info("Saved EMG GRU to %s and %s", emg_gru_dir, models_dir / "emg_gru.pt")

    # EMG CNN-GRU
    logger.info("Training production EMG CNN-GRU...")
    emg_cnngru = EMGCNNGRUModel(label_map=emg_label_map, model_id="emg-cnn-gru-prod", cnn_filters=16, gru_hidden_size=32, gru_num_layers=1, n_epochs=6, batch_size=32)
    emg_cnngru.fit(emg_X_seq, emg_y)
    emg_cnngru_dir = models_dir / "emg_cnn_gru"
    emg_cnngru.save(emg_cnngru_dir)
    shutil.copy2(emg_cnngru_dir / "model.pt", models_dir / "emg_cnn_gru.pt")
    logger.info("Saved EMG CNN-GRU to %s and %s", emg_cnngru_dir, models_dir / "emg_cnn_gru.pt")

    # 2. Pool EEG Data
    eeg_X_feat = np.vstack([d["features"] for d in eeg_data.values()])
    eeg_X_seq = np.vstack([d["sequences"] for d in eeg_data.values()])
    eeg_y = np.concatenate([d["labels"] for d in eeg_data.values()])
    eeg_label_map = {lbl: idx for idx, lbl in enumerate(EEG_LABELS)}

    logger.info("Pooled EEG dataset: %d samples, classes=%s", len(eeg_y), EEG_LABELS)

    # EEG SVM
    logger.info("Training production EEG SVM...")
    eeg_svm = EEGSVMModel(label_map=eeg_label_map, model_id="eeg-svm-prod")
    eeg_svm.fit(eeg_X_feat, eeg_y)
    eeg_svm_dir = models_dir / "eeg_svm"
    eeg_svm.save(eeg_svm_dir)
    shutil.copy2(eeg_svm_dir / "model.pkl", models_dir / "eeg_svm.pkl")
    logger.info("Saved EEG SVM to %s and %s", eeg_svm_dir, models_dir / "eeg_svm.pkl")

    # EEG Small 1D-CNN
    logger.info("Training production EEG Small 1D-CNN...")
    eeg_cnn1d = EEGCNN1DModel(label_map=eeg_label_map, model_id="eeg-cnn1d-prod", cnn_filters=16, n_epochs=6, batch_size=32)
    eeg_cnn1d.fit(eeg_X_seq, eeg_y)
    eeg_cnn1d_dir = models_dir / "eeg_cnn1d"
    eeg_cnn1d.save(eeg_cnn1d_dir)
    shutil.copy2(eeg_cnn1d_dir / "model.pt", models_dir / "eeg_cnn1d.pt")
    logger.info("Saved EEG Small 1D-CNN to %s and %s", eeg_cnn1d_dir, models_dir / "eeg_cnn1d.pt")

    return {
        "models_dir": str(models_dir),
        "emg_svm": str(emg_svm_dir),
        "emg_svm_pkl": str(models_dir / "emg_svm.pkl"),
        "emg_gru": str(emg_gru_dir),
        "emg_gru_pt": str(models_dir / "emg_gru.pt"),
        "emg_cnn_gru": str(emg_cnngru_dir),
        "emg_cnn_gru_pt": str(models_dir / "emg_cnn_gru.pt"),
        "eeg_svm": str(eeg_svm_dir),
        "eeg_svm_pkl": str(models_dir / "eeg_svm.pkl"),
        "eeg_cnn1d": str(eeg_cnn1d_dir),
        "eeg_cnn1d_pt": str(models_dir / "eeg_cnn1d.pt"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Experiment Execution
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate URF/A-SSR models using LOSO")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Path to adapted data directory"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("experiments/results.json"),
        help="Path to output results.json"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=250,
        help="Sequence length in samples for deep learning models"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save production model weights"
    )
    args = parser.parse_args()
    
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    subjects = [f"Subject_{i:02d}" for i in range(1, 7)]
    
    # 1. Load Datasets
    logger.info("Loading adapted datasets...")
    emg_data = load_emg_dataset(args.data_dir, subjects, seq_len=args.seq_len)
    eeg_data = load_eeg_dataset(args.data_dir, subjects, seq_len=args.seq_len)
    
    # 2. EMG Branch Models (SVM, GRU, CNN-GRU)
    logger.info("\n=== EVALUATING EMG BRANCH (Gaddy) ===")
    emg_svm_res = evaluate_loso(emg_data, "SVM", "EMG", EMG_LABELS)
    emg_gru_res = evaluate_loso(emg_data, "GRU", "EMG", EMG_LABELS)
    emg_cnngru_res = evaluate_loso(emg_data, "CNN-GRU", "EMG", EMG_LABELS)
    
    # 3. EEG Branch Models (SVM, Small 1D-CNN)
    logger.info("\n=== EVALUATING EEG BRANCH (DEAP) ===")
    eeg_svm_res = evaluate_loso(eeg_data, "SVM", "EEG", EEG_LABELS)
    eeg_cnn1d_res = evaluate_loso(eeg_data, "Small 1D-CNN", "EEG", EEG_LABELS)
    
    # 4. Safety Engine Verification
    logger.info("\n=== VERIFYING SAFETY ENGINE ===")
    safety_res = verify_safety_engine(threshold=0.70)

    # 5. Train and Save Production Models
    logger.info("\n=== TRAINING AND SAVING PRODUCTION MODELS ===")
    saved_models_info = train_and_save_final_models(emg_data, eeg_data, args.models_dir)
    
    # 6. Compile and Save Results
    results = {
        "evaluation_protocol": "Strict Leave-One-Subject-Out (LOSO) Cross-Validation",
        "description": "Zero data leakage cross-subject validation across 6 subjects",
        "emg_branch": {
            "SVM": emg_svm_res,
            "GRU": emg_gru_res,
            "CNN-GRU": emg_cnngru_res,
        },
        "eeg_branch": {
            "SVM": eeg_svm_res,
            "Small 1D-CNN": eeg_cnn1d_res,
        },
        "safety_engine": safety_res,
        "saved_models": saved_models_info,
    }
    
    with open(args.output_file, "w") as rf:
        json.dump(results, rf, indent=2)
        
    logger.info("Saved complete evaluation results to %s", args.output_file)
    
    # Console Summary
    print("\n" + "="*70)
    print("           URF/A-SSR MODEL BENCHMARK RESULTS (STRICT LOSO)")
    print("="*70)
    
    print("\n--- EMG Silent Speech Branch (3-Class: YES / NO / HELP) ---")
    print(f"{'Model':<12} | {'Macro-F1':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'HELP Recall':<12}")
    print("-" * 70)
    for name, r in [("SVM", emg_svm_res), ("GRU", emg_gru_res), ("CNN-GRU", emg_cnngru_res)]:
        print(f"{name:<12} | {r['macro_f1']:<10.4f} | {r['accuracy']:<10.4f} | {r['precision']:<10.4f} | {r['recall']:<10.4f} | {r['help_recall']:<12.4f}")
        
    print("\n--- EEG Affective/Cognitive Branch (2-Class: Neutral / Stressed) ---")
    print(f"{'Model':<12} | {'Macro-F1':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 56)
    for name, r in [("SVM", eeg_svm_res), ("Small 1D-CNN", eeg_cnn1d_res)]:
        print(f"{name:<12} | {r['macro_f1']:<10.4f} | {r['accuracy']:<10.4f} | {r['precision']:<10.4f} | {r['recall']:<10.4f}")
        
    print("\n--- Deterministic Safety Engine Verification (Threshold = 0.70) ---")
    print(f"Status: {safety_res['status']}")
    print(f"Total Test Cases: {safety_res['total_tests']} | Passed: {safety_res['passed_tests']}")
    print(f"NO_DECISION on Low Confidence (<0.70): VERIFIED")

    print("\n--- Saved Production Models (models/) ---")
    for k, v in saved_models_info.items():
        print(f"  {k:<16}: {v}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
