#!/usr/bin/env python3
"""
scripts/adapt_datasets.py
=========================
URF/A-SSR Dataset Adapter Script.

Adapts public DEAP (EEG) and Gaddy (silent speech sEMG) datasets into
the standardized URF/A-SSR raw_signals.csv format and directory hierarchy:

  data/processed/
    DEAP/
      Subject_01/
        Session_01/
          raw_signals.csv
          trials.json
          metadata.json
        Session_02/
          ...
    Gaddy/
      Subject_01/
        Session_01/
          raw_signals.csv
          trials.json
          metadata.json

Schema for raw_signals.csv:
  sample_index, host_timestamp, EEG_Fp1, EEG_Fp2, EMG_submental, EMG_orbicularis, EMG_zygomaticus, EMG_buccinator

Adapter rules:
  1. DEAP:
     - Extracts ONLY Fp1 and Fp2 channels (indices 0 and 1).
     - Upsamples 128Hz data to 2000Hz using scipy.signal.resample.
     - Maps Valence/Arousal to Neutral/Stressed state labels.
     - Sets all 4 EMG columns to 0.0.
  2. Gaddy:
     - Extracts facial sEMG channels.
     - Maps first 4 channels to EMG_submental, EMG_orbicularis, EMG_zygomaticus, EMG_buccinator.
     - Sets the 2 EEG columns to 0.0.
     - Maps words to YES / NO / HELP intent labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import re
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Tuple

import numpy as np
from scipy import signal as sp_signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("adapt_datasets")

CSV_HEADER = [
    "sample_index",
    "host_timestamp",
    "EEG_Fp1",
    "EEG_Fp2",
    "EMG_submental",
    "EMG_orbicularis",
    "EMG_zygomaticus",
    "EMG_buccinator",
]

# ─────────────────────────────────────────────────────────────────────────────
# DEAP Adapter
# ─────────────────────────────────────────────────────────────────────────────

def adapt_deap(
    deap_dir: Path,
    output_base: Path,
    max_subjects: int = 6,
    target_fs: float = 2000.0,
    trial_duration_sec: float = 10.0,  # 10s slice of active stimulus per trial
) -> Dict[str, any]:
    """
    Adapt DEAP .dat files into URF/A-SSR processed format.
    
    DEAP .dat structure:
      data: shape (40, 40, 8064) -> (trials, channels, 63s at 128Hz)
      labels: shape (40, 4) -> [valence, arousal, dominance, liking] (1 to 9)
    """
    logger.info("Starting DEAP dataset adaptation from %s", deap_dir)
    deap_out = output_base / "DEAP"
    deap_out.mkdir(parents=True, exist_ok=True)
    
    files = sorted(deap_dir.glob("s*.dat"))
    if not files:
        raise FileNotFoundError(f"No s*.dat files found in {deap_dir}")
    
    selected_files = files[:max_subjects]
    orig_fs = 128.0
    # Samples in active stimulus window (start at 3s to skip pre-trial baseline)
    start_sample_orig = int(3.0 * orig_fs)
    samples_orig = int(trial_duration_sec * orig_fs)
    end_sample_orig = start_sample_orig + samples_orig
    target_samples = int(trial_duration_sec * target_fs)
    
    stats = {"subjects": len(selected_files), "sessions": 0, "trials": 0, "classes": {"Neutral": 0, "Stressed": 0}}
    
    for subj_idx, filepath in enumerate(selected_files, start=1):
        subj_name = f"Subject_{subj_idx:02d}"
        logger.info("Processing DEAP %s (%s)...", subj_name, filepath.name)
        
        with open(filepath, "rb") as f:
            obj = pickle.load(f, encoding="latin1")
        
        trials_data = obj["data"]    # (40, 40, 8064)
        trials_labels = obj["labels"]  # (40, 4)
        
        # Split 40 trials into 2 sessions of 20 trials each
        session_splits = [
            ("Session_01", range(0, 20)),
            ("Session_02", range(20, 40)),
        ]
        
        for sess_name, trial_indices in session_splits:
            sess_dir = deap_out / subj_name / sess_name
            sess_dir.mkdir(parents=True, exist_ok=True)
            stats["sessions"] += 1
            
            csv_path = sess_dir / "raw_signals.csv"
            trials_meta = []
            
            all_rows = []
            current_sample_idx = 0
            
            for t_idx in trial_indices:
                stats["trials"] += 1
                val = float(trials_labels[t_idx, 0])
                arousal = float(trials_labels[t_idx, 1])
                
                # Affective mapping: Low valence (< 5.0) mapped to Stressed, >= 5.0 to Neutral
                state_label = "Stressed" if val < 5.0 else "Neutral"
                stats["classes"][state_label] += 1
                
                # Extract Fp1 (ch 0) and Fp2 (ch 1)
                raw_fp1 = trials_data[t_idx, 0, start_sample_orig:end_sample_orig]
                raw_fp2 = trials_data[t_idx, 1, start_sample_orig:end_sample_orig]
                
                # Resample 128Hz -> 2000Hz
                resampled_fp1 = sp_signal.resample(raw_fp1, target_samples)
                resampled_fp2 = sp_signal.resample(raw_fp2, target_samples)
                
                trial_start = current_sample_idx
                trial_end = trial_start + target_samples - 1
                
                trials_meta.append({
                    "trial_id": f"T{t_idx+1:03d}",
                    "state": state_label,
                    "valence": val,
                    "arousal": arousal,
                    "start_sample": trial_start,
                    "end_sample": trial_end,
                    "start_time_s": trial_start / target_fs,
                    "end_time_s": trial_end / target_fs,
                })
                
                # Build rows: sample_index, host_timestamp, EEG_Fp1, EEG_Fp2, 4x EMG (0.0)
                t_indices = np.arange(current_sample_idx, current_sample_idx + target_samples)
                timestamps = t_indices / target_fs
                zeros = np.zeros(target_samples)
                
                trial_matrix = np.column_stack([
                    t_indices,
                    timestamps,
                    resampled_fp1,
                    resampled_fp2,
                    zeros,
                    zeros,
                    zeros,
                    zeros,
                ])
                all_rows.append(trial_matrix)
                current_sample_idx += target_samples
            
            session_data = np.vstack(all_rows)
            # Write CSV
            with open(csv_path, "w", newline="") as csv_f:
                writer = csv.writer(csv_f)
                writer.writerow(CSV_HEADER)
                for row in session_data:
                    writer.writerow([int(row[0]), f"{row[1]:.6f}"] + [f"{v:.4f}" for v in row[2:]])
            
            # Write trial metadata
            with open(sess_dir / "trials.json", "w") as jf:
                json.dump(trials_meta, jf, indent=2)
                
            # Write session metadata
            with open(sess_dir / "metadata.json", "w") as mf:
                json.dump({
                    "modality": "EEG",
                    "subject_id": subj_name,
                    "session_id": sess_name,
                    "sampling_rate_hz": target_fs,
                    "source_dataset": "DEAP",
                    "source_file": filepath.name,
                    "n_samples": len(session_data),
                    "n_trials": len(trials_meta),
                }, mf, indent=2)
                
    logger.info("DEAP adaptation complete: %s", stats)
    return stats

# ─────────────────────────────────────────────────────────────────────────────
# Gaddy Adapter
# ─────────────────────────────────────────────────────────────────────────────

def _classify_gaddy_text(text: str) -> str | None:
    """Map utterance transcript to YES, NO, or HELP intent."""
    words = set(re.findall(r"\b\w+\b", text.lower()))
    if not words:
        return None
    
    # Priority 1: Direct YES
    if "yes" in words or text.lower().startswith("yes") or text.lower().startswith("“yes"):
        return "YES"
    
    # Priority 2: HELP and emergency assistance vocabulary
    help_keywords = {
        "help", "stop", "please", "urgent", "care", "pain", "doctor",
        "nurse", "assist", "call", "danger", "trouble", "emergency", "hurry"
    }
    if words & help_keywords:
        return "HELP"
    
    # Priority 3: Direct NO / negation
    if "no" in words or "not" in words or "never" in words:
        return "NO"
        
    return None


def adapt_gaddy(
    gaddy_dir: Path,
    output_base: Path,
    max_subjects: int = 6,
    target_fs: float = 2000.0,
    orig_fs: float = 1000.0,
) -> Dict[str, any]:
    """
    Adapt Gaddy sEMG .npy files into URF/A-SSR processed format.
    
    Organizes 12 chosen sessions into 6 subjects x 2 sessions.
    Maps first 4 EMG channels to required schema, sets EEG to 0.0.
    """
    logger.info("Starting Gaddy dataset adaptation from %s", gaddy_dir)
    gaddy_out = output_base / "Gaddy"
    gaddy_out.mkdir(parents=True, exist_ok=True)
    
    # Select balanced sessions with high trial variety
    session_mapping = [
        ("Subject_01", [("Session_01", "nonparallel_data/4-21"), ("Session_02", "nonparallel_data/4-23")]),
        ("Subject_02", [("Session_01", "nonparallel_data/4-24"), ("Session_02", "nonparallel_data/4-25")]),
        ("Subject_03", [("Session_01", "nonparallel_data/4-26"), ("Session_02", "nonparallel_data/4-27")]),
        ("Subject_04", [("Session_01", "nonparallel_data/4-29"), ("Session_02", "nonparallel_data/4-30")]),
        ("Subject_05", [("Session_01", "nonparallel_data/5-2"), ("Session_02", "nonparallel_data/4-25_2")]),
        ("Subject_06", [("Session_01", "silent_parallel_data/5-10_silent"), ("Session_02", "silent_parallel_data/5-5_silent")]),
    ][:max_subjects]
    
    stats = {"subjects": len(session_mapping), "sessions": 0, "trials": 0, "classes": {"YES": 0, "NO": 0, "HELP": 0}}
    
    for subj_name, sess_list in session_mapping:
        logger.info("Processing Gaddy %s...", subj_name)
        for sess_name, rel_path in sess_list:
            src_session_dir = gaddy_dir / rel_path
            if not src_session_dir.exists():
                logger.warning("Source session directory %s does not exist, skipping", src_session_dir)
                continue
                
            sess_dir = gaddy_out / subj_name / sess_name
            sess_dir.mkdir(parents=True, exist_ok=True)
            stats["sessions"] += 1
            
            # Find all info files
            info_files = sorted(src_session_dir.glob("*_info.json"))
            valid_trials = []
            
            # Collect candidates
            candidates_by_class = {"YES": [], "NO": [], "HELP": []}
            for info_p in info_files:
                emg_p = info_p.parent / info_p.name.replace("_info.json", "_emg.npy")
                if not emg_p.exists():
                    continue
                try:
                    with open(info_p) as f:
                        meta = json.load(f)
                    text = meta.get("text", "")
                    intent = _classify_gaddy_text(text)
                    if intent:
                        candidates_by_class[intent].append((emg_p, meta, text, intent))
                except Exception:
                    continue
            
            # Balance candidates per session: up to 10 YES, 10 HELP, 10 NO
            balanced_candidates = []
            for c_label in ["YES", "HELP", "NO"]:
                c_list = candidates_by_class[c_label]
                # Take up to 12 of each class to form a rich session
                balanced_candidates.extend(c_list[:12])
                
            if not balanced_candidates:
                logger.warning("No valid trials found in %s", src_session_dir)
                continue
                
            csv_path = sess_dir / "raw_signals.csv"
            trials_meta = []
            all_rows = []
            current_sample_idx = 0
            
            for t_num, (emg_path, meta, text, intent) in enumerate(balanced_candidates, start=1):
                raw_emg = np.load(emg_path)  # shape (time, 8)
                if raw_emg.ndim != 2 or raw_emg.shape[1] < 4:
                    continue
                    
                # Extract first 4 channels: submental, orbicularis, zygomaticus, buccinator
                emg_4ch = raw_emg[:, :4]
                
                # Upsample from 1000Hz to 2000Hz if needed
                if orig_fs != target_fs:
                    n_target = int(len(emg_4ch) * (target_fs / orig_fs))
                    emg_resampled = sp_signal.resample(emg_4ch, n_target, axis=0)
                else:
                    emg_resampled = emg_4ch
                    n_target = len(emg_resampled)
                    
                trial_start = current_sample_idx
                trial_end = trial_start + n_target - 1
                
                stats["trials"] += 1
                stats["classes"][intent] += 1
                
                trials_meta.append({
                    "trial_id": f"T{t_num:03d}",
                    "intent": intent,
                    "text": text,
                    "start_sample": trial_start,
                    "end_sample": trial_end,
                    "start_time_s": trial_start / target_fs,
                    "end_time_s": trial_end / target_fs,
                })
                
                t_indices = np.arange(current_sample_idx, current_sample_idx + n_target)
                timestamps = t_indices / target_fs
                zeros = np.zeros(n_target)
                
                # Schema: sample_index, host_timestamp, EEG_Fp1, EEG_Fp2, EMG_sub, EMG_orb, EMG_zyg, EMG_buc
                trial_matrix = np.column_stack([
                    t_indices,
                    timestamps,
                    zeros,  # EEG_Fp1 = 0
                    zeros,  # EEG_Fp2 = 0
                    emg_resampled[:, 0],
                    emg_resampled[:, 1],
                    emg_resampled[:, 2],
                    emg_resampled[:, 3],
                ])
                all_rows.append(trial_matrix)
                current_sample_idx += n_target
                
            if not all_rows:
                continue
                
            session_data = np.vstack(all_rows)
            # Write CSV
            with open(csv_path, "w", newline="") as csv_f:
                writer = csv.writer(csv_f)
                writer.writerow(CSV_HEADER)
                for row in session_data:
                    writer.writerow([int(row[0]), f"{row[1]:.6f}"] + [f"{v:.4f}" for v in row[2:]])
            
            with open(sess_dir / "trials.json", "w") as jf:
                json.dump(trials_meta, jf, indent=2)
                
            with open(sess_dir / "metadata.json", "w") as mf:
                json.dump({
                    "modality": "EMG",
                    "subject_id": subj_name,
                    "session_id": sess_name,
                    "sampling_rate_hz": target_fs,
                    "source_dataset": "Gaddy",
                    "source_session": rel_path,
                    "n_samples": len(session_data),
                    "n_trials": len(trials_meta),
                }, mf, indent=2)
                
    logger.info("Gaddy adaptation complete: %s", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt DEAP and Gaddy datasets into URF/A-SSR schema")
    parser.add_argument(
        "--deap-dir",
        type=Path,
        default=Path("data/public_datasets/DEAP/data_preprocessed_python"),
        help="Path to DEAP preprocessed .dat directory"
    )
    parser.add_argument(
        "--gaddy-dir",
        type=Path,
        default=Path("data/public_datasets/Gaddy/emg_data/emg_data"),
        help="Path to Gaddy emg_data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Target output base directory"
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=6,
        help="Number of subjects to adapt per dataset (default: 6 for strict LOSO)"
    )
    parser.add_argument(
        "--target-fs",
        type=float,
        default=2000.0,
        help="Target sampling rate in Hz (default: 2000.0)"
    )
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    deap_stats = adapt_deap(args.deap_dir, args.output_dir, max_subjects=args.max_subjects, target_fs=args.target_fs)
    gaddy_stats = adapt_gaddy(args.gaddy_dir, args.output_dir, max_subjects=args.max_subjects, target_fs=args.target_fs)
    
    summary_path = args.output_dir / "adaptation_summary.json"
    with open(summary_path, "w") as sf:
        json.dump({
            "DEAP": deap_stats,
            "Gaddy": gaddy_stats,
            "target_fs": args.target_fs,
            "schema": CSV_HEADER,
        }, sf, indent=2)
        
    print("\n================ DATASET ADAPTATION SUMMARY ================")
    print(f"Output directory: {args.output_dir}")
    print(f"DEAP:  {deap_stats['subjects']} subjects, {deap_stats['trials']} trials, Classes: {deap_stats['classes']}")
    print(f"Gaddy: {gaddy_stats['subjects']} subjects, {gaddy_stats['trials']} trials, Classes: {gaddy_stats['classes']}")
    print("============================================================\n")


if __name__ == "__main__":
    main()
