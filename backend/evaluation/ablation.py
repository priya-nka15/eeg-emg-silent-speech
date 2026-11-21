"""
URF/A-SSR Channel Ablation
PRD §42 — EMG: all-4, leave-one-out; EEG: Fp1+Fp2, Fp1-only, Fp2-only.
Ablation is a secondary research phase, not a prerequisite.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from backend.evaluation.metrics import EvaluationResult, evaluate


EMG_CHANNEL_NAMES = [
    "EMG_Submental",
    "EMG_Orbicularis",
    "EMG_Zygomaticus",
    "EMG_Buccinator",
]

EEG_CHANNEL_NAMES = ["EEG_Fp1", "EEG_Fp2"]


def emg_ablation_subsets() -> List[Dict[str, List[str]]]:
    """
    Return list of channel subsets for EMG ablation.
    PRD §42: all 4 channels + leave-one-out.
    """
    subsets = [{"name": "all_channels", "channels": list(EMG_CHANNEL_NAMES)}]
    for ch in EMG_CHANNEL_NAMES:
        remaining = [c for c in EMG_CHANNEL_NAMES if c != ch]
        subsets.append({"name": f"leave_out_{ch}", "channels": remaining})
    return subsets


def eeg_ablation_subsets() -> List[Dict[str, List[str]]]:
    """
    Return EEG channel subsets for ablation.
    PRD §42: Fp1+Fp2, Fp1-only, Fp2-only.
    """
    return [
        {"name": "fp1_fp2", "channels": ["EEG_Fp1", "EEG_Fp2"]},
        {"name": "fp1_only", "channels": ["EEG_Fp1"]},
        {"name": "fp2_only", "channels": ["EEG_Fp2"]},
    ]
