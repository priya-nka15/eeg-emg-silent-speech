"""
URF/A-SSR Dataset Splitter
PRD §28 — Subject/session-aware splitting. Never random window splits.
Generates split_manifest.json.
"""
from __future__ import annotations

import json
import random
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class SplitStrategy(str, Enum):
    SESSION_HELD_OUT = "session_held_out"
    SUBJECT_HELD_OUT = "subject_held_out"
    LOSO             = "LOSO"              # Leave-One-Subject-Out


class SplitLeakageError(Exception):
    """Raised when a proposed split would create train/test leakage."""


def _check_no_subject_leakage(
    train_subjects: List[str],
    test_subjects: List[str],
) -> None:
    overlap = set(train_subjects) & set(test_subjects)
    if overlap:
        raise SplitLeakageError(
            f"Subject(s) {overlap} appear in both train and test sets. "
            "This creates leakage — use subject-aware splitting."
        )


def _check_no_session_leakage(
    train_sessions: List[Tuple[str, str]],
    test_sessions: List[Tuple[str, str]],
) -> None:
    train_set = set(train_sessions)
    test_set = set(test_sessions)
    overlap = train_set & test_set
    if overlap:
        raise SplitLeakageError(
            f"Sessions {overlap} appear in both train and test. "
            "Use session-aware splitting."
        )


class DatasetSplitter:
    """
    Creates leakage-safe train/validation/test splits.
    Subject/session granularity — never individual window granularity.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def split_by_subject(
        self,
        subjects: List[str],
        sessions_per_subject: Dict[str, List[str]],
        val_fraction: float = 0.15,
        test_fraction: float = 0.20,
    ) -> dict:
        """
        Hold out complete subjects for validation and test.
        All sessions from a subject stay together.
        """
        rng = random.Random(self.seed)
        shuffled = list(subjects)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_test = max(1, int(n * test_fraction))
        n_val = max(1, int(n * val_fraction))
        n_train = n - n_test - n_val

        if n_train < 1:
            raise ValueError(
                f"Not enough subjects ({n}) for train/val/test split."
            )

        train_subjects = shuffled[:n_train]
        val_subjects = shuffled[n_train:n_train + n_val]
        test_subjects = shuffled[n_train + n_val:]

        _check_no_subject_leakage(train_subjects, test_subjects)
        _check_no_subject_leakage(train_subjects, val_subjects)
        _check_no_subject_leakage(val_subjects, test_subjects)

        return self._build_manifest(
            strategy=SplitStrategy.SUBJECT_HELD_OUT,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=test_subjects,
            sessions_per_subject=sessions_per_subject,
        )

    def split_by_session(
        self,
        subjects: List[str],
        sessions_per_subject: Dict[str, List[str]],
        held_out_session: str = "Session_02",
    ) -> dict:
        """
        Hold out a specific session for test (e.g., Session_02).
        Train on Session_01, test on Session_02.
        """
        train_sessions = []
        test_sessions = []
        for subj in subjects:
            sessions = sessions_per_subject.get(subj, [])
            for sess in sessions:
                if sess == held_out_session:
                    test_sessions.append((subj, sess))
                else:
                    train_sessions.append((subj, sess))

        _check_no_session_leakage(train_sessions, test_sessions)

        return {
            "manifest_version": "1",
            "split_strategy": SplitStrategy.SESSION_HELD_OUT,
            "seed": self.seed,
            "held_out_session": held_out_session,
            "train": {
                "subjects": sorted(set(s[0] for s in train_sessions)),
                "sessions": [{"subject": s, "session": sess} for s, sess in train_sessions],
            },
            "validation": {"subjects": [], "sessions": []},
            "test": {
                "subjects": sorted(set(s[0] for s in test_sessions)),
                "sessions": [{"subject": s, "session": sess} for s, sess in test_sessions],
            },
            "test_locked": True,
            "notes": f"Session {held_out_session} held out for test.",
        }

    def loso_splits(
        self,
        subjects: List[str],
        sessions_per_subject: Dict[str, List[str]],
    ) -> List[dict]:
        """
        Leave-One-Subject-Out: returns one split manifest per fold.
        """
        manifests = []
        for held_out in subjects:
            train_subjects = [s for s in subjects if s != held_out]
            test_subjects = [held_out]
            _check_no_subject_leakage(train_subjects, test_subjects)
            manifests.append(
                self._build_manifest(
                    strategy=SplitStrategy.LOSO,
                    train_subjects=train_subjects,
                    val_subjects=[],
                    test_subjects=test_subjects,
                    sessions_per_subject=sessions_per_subject,
                    notes=f"LOSO fold: held out {held_out}",
                )
            )
        return manifests

    def _build_manifest(
        self,
        strategy: SplitStrategy,
        train_subjects: List[str],
        val_subjects: List[str],
        test_subjects: List[str],
        sessions_per_subject: Dict[str, List[str]],
        notes: str = "",
    ) -> dict:
        def sessions_for(subjs):
            result = []
            for s in subjs:
                for sess in sessions_per_subject.get(s, []):
                    result.append({"subject": s, "session": sess})
            return result

        return {
            "manifest_version": "1",
            "split_strategy": strategy,
            "seed": self.seed,
            "train": {
                "subjects": train_subjects,
                "sessions": sessions_for(train_subjects),
            },
            "validation": {
                "subjects": val_subjects,
                "sessions": sessions_for(val_subjects),
            },
            "test": {
                "subjects": test_subjects,
                "sessions": sessions_for(test_subjects),
            },
            "test_locked": True,
            "notes": notes,
        }

    def save_manifest(self, manifest: dict, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def load_manifest(self, path: Path) -> dict:
        with open(path) as f:
            return json.load(f)
