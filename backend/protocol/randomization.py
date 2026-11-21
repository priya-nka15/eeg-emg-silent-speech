"""
URF/A-SSR Trial Randomization
PRD §19 — Reproducible randomized/block-randomized trial sequence, saved in metadata.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional


def generate_block_randomized_sequence(
    classes: List[str],
    trials_per_class: int,
    seed: int,
    block_size: Optional[int] = None,
) -> List[str]:
    """
    Generate a reproducible block-randomized trial sequence.

    Each block contains exactly one occurrence of each class, shuffled.
    This guarantees balanced exposure across the session and prevents
    recording all YES then all NO then all HELP.

    Args:
        classes: List of class labels, e.g. ["YES", "NO", "HELP"]
        trials_per_class: Number of trials per class.
        seed: Random seed for reproducibility.
        block_size: Number of classes per block. Defaults to len(classes).

    Returns:
        List of class labels in randomized order.
    """
    if not classes:
        raise ValueError("classes must not be empty")
    if trials_per_class <= 0:
        raise ValueError("trials_per_class must be positive")

    rng = random.Random(seed)
    block_size = block_size or len(classes)
    n_blocks = trials_per_class  # one occurrence per class per block

    sequence: List[str] = []
    for _ in range(n_blocks):
        block = list(classes)
        rng.shuffle(block)
        sequence.extend(block)

    return sequence


def generate_fully_randomized_sequence(
    classes: List[str],
    trials_per_class: int,
    seed: int,
) -> List[str]:
    """
    Generate a fully randomized sequence (no block structure).

    Args:
        classes: List of class labels.
        trials_per_class: Number of trials per class.
        seed: Random seed.

    Returns:
        Fully shuffled list of class labels.
    """
    pool = classes * trials_per_class
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool


def save_sequence(
    sequence: List[str],
    seed: int,
    strategy: str,
    output_path: Path,
    metadata: Optional[dict] = None,
) -> None:
    """Save the trial sequence to JSON for session metadata."""
    payload = {
        "strategy": strategy,
        "seed": seed,
        "n_trials": len(sequence),
        "sequence": sequence,
        "metadata": metadata or {},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def load_sequence(path: Path) -> dict:
    """Load a previously saved randomization sequence."""
    with open(path) as f:
        return json.load(f)
