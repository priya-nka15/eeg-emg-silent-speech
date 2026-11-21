"""
URF/A-SSR Raw Data Store
PRD §13 — Immutable raw data policy. Never overwrite raw/.
Every processed artifact identifies its source raw recording and preprocessing version.
"""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from backend.acquisition.frame_parser import Sample, CHANNEL_NAMES


# CSV column order (matches PRD §12 Sample schema)
CSV_COLUMNS = [
    "sample_id", "timestamp_us",
] + CHANNEL_NAMES + ["event_marker", "status"]


class RawStoreError(Exception):
    """Raised when a raw store operation would violate the immutability policy."""


class RawStore:
    """
    Manages raw recording storage under the directory structure:

        URF_DATA/Subject_NN/Session_NN/
            raw/            <- IMMUTABLE
            metadata/
            events/
            qc/
            processed/<version>/
            features/<version>/

    Writing to an existing raw/ directory is prohibited.
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)

    def session_dir(self, subject_id: str, session_id: str) -> Path:
        return self.base_dir / subject_id / session_id

    def raw_dir(self, subject_id: str, session_id: str) -> Path:
        return self.session_dir(subject_id, session_id) / "raw"

    def processed_dir(
        self, subject_id: str, session_id: str, version: str
    ) -> Path:
        return self.session_dir(subject_id, session_id) / "processed" / version

    def features_dir(
        self, subject_id: str, session_id: str, version: str
    ) -> Path:
        return self.session_dir(subject_id, session_id) / "features" / version

    # ── Writing ───────────────────────────────────────────────────────────────

    def open_recording(
        self, subject_id: str, session_id: str
    ) -> "RecordingWriter":
        """
        Open a new recording session for writing.
        Raises RawStoreError if raw/ already exists (immutability policy).
        """
        raw = self.raw_dir(subject_id, session_id)
        if raw.exists() and any(raw.iterdir()):
            raise RawStoreError(
                f"Raw data already exists at {raw}. "
                "Raw data is immutable — never overwrite. "
                "Create a new session directory instead."
            )
        raw.mkdir(parents=True, exist_ok=True)
        return RecordingWriter(raw)

    def write_processed(
        self,
        subject_id: str,
        session_id: str,
        version: str,
        data: np.ndarray,
        manifest: dict,
        filename: str = "processed.npy",
    ) -> Path:
        """Write a preprocessed array. Version-stamped, never overwrites raw."""
        out_dir = self.processed_dir(subject_id, session_id, version)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / filename, data)
        manifest["source_raw"] = str(self.raw_dir(subject_id, session_id))
        manifest["preprocessing_version"] = version
        manifest["created"] = datetime.utcnow().isoformat()
        with open(out_dir / "processing_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        return out_dir / filename

    def write_features(
        self,
        subject_id: str,
        session_id: str,
        version: str,
        features: np.ndarray,
        labels: np.ndarray,
        manifest: dict,
    ) -> Path:
        """Write feature array. Version-stamped."""
        out_dir = self.features_dir(subject_id, session_id, version)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "features.npy", features)
        np.save(out_dir / "labels.npy", labels)
        manifest["source_raw"] = str(self.raw_dir(subject_id, session_id))
        manifest["feature_version"] = version
        manifest["created"] = datetime.utcnow().isoformat()
        with open(out_dir / "feature_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        return out_dir / "features.npy"

    # ── Reading ───────────────────────────────────────────────────────────────

    def read_raw_csv(
        self, subject_id: str, session_id: str
    ) -> Iterator[dict]:
        """Iterate over raw samples as dicts (generator, no full load)."""
        raw_file = self.raw_dir(subject_id, session_id) / "recording.csv"
        if not raw_file.exists():
            raise FileNotFoundError(f"Raw recording not found: {raw_file}")
        with open(raw_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield {k: int(v) if k != "timestamp_us" else int(v) for k, v in row.items()}

    def read_metadata(self, subject_id: str, session_id: str) -> dict:
        meta_dir = self.session_dir(subject_id, session_id) / "metadata"
        result = {}
        for fname in ("session.json", "trials.json", "subject.json"):
            p = meta_dir / fname
            if p.exists():
                with open(p) as f:
                    result[fname.replace(".json", "")] = json.load(f)
        return result

    def read_events(self, subject_id: str, session_id: str) -> list:
        events_file = (
            self.session_dir(subject_id, session_id) / "events" / "events.json"
        )
        if not events_file.exists():
            return []
        with open(events_file) as f:
            return json.load(f)

    def list_sessions(self, subject_id: str) -> List[str]:
        subject_path = self.base_dir / subject_id
        if not subject_path.exists():
            return []
        return sorted(
            d.name for d in subject_path.iterdir() if d.is_dir()
        )

    def list_subjects(self) -> List[str]:
        if not self.base_dir.exists():
            return []
        return sorted(
            d.name for d in self.base_dir.iterdir()
            if d.is_dir() and d.name.startswith("S")
        )


class RecordingWriter:
    """
    Context manager for writing raw samples to CSV.
    CSV is append-only during session; closed on exit.
    """

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir
        self._csv_path = raw_dir / "recording.csv"
        self._file = None
        self._writer = None
        self._n_samples = 0

    def __enter__(self) -> "RecordingWriter":
        self._file = open(self._csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._file:
            self._file.flush()
            self._file.close()

    def write_sample(self, sample: Sample) -> None:
        if self._writer is None:
            raise RuntimeError("RecordingWriter not opened. Use as context manager.")
        self._writer.writerow(sample.to_dict())
        self._n_samples += 1

    def write_samples(self, samples: List[Sample]) -> None:
        for s in samples:
            self.write_sample(s)

    @property
    def n_samples(self) -> int:
        return self._n_samples
