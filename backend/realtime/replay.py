"""
URF/A-SSR Replay Mode
PRD §40 — Loads a recorded raw session and runs the exact production pipeline.
Replay is mandatory for reproducibility and debugging.
Any divergence between live and replay output is a bug.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, List

from backend.acquisition.frame_parser import Sample
from backend.datasets.raw_store import RawStore


class ReplaySource:
    """
    Loads samples from a raw CSV recording and replays them.
    Uses the exact same Sample representation as live acquisition.
    Does NOT use real hardware.
    """

    def __init__(self, raw_csv_path: Path) -> None:
        self.raw_csv_path = Path(raw_csv_path)
        if not self.raw_csv_path.exists():
            raise FileNotFoundError(f"Replay file not found: {self.raw_csv_path}")
        self._is_replay = True

    def stream(self) -> Iterator[Sample]:
        """Yield samples from the recording in order."""
        with open(self.raw_csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield Sample.from_dict(row)

    def load_all(self) -> List[Sample]:
        """Load all samples into memory."""
        return list(self.stream())

    @property
    def is_replay(self) -> bool:
        return True

    @classmethod
    def from_store(
        cls, store: RawStore, subject_id: str, session_id: str
    ) -> "ReplaySource":
        raw_csv = store.raw_dir(subject_id, session_id) / "recording.csv"
        return cls(raw_csv)
