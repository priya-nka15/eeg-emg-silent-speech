"""
URF/A-SSR Experiment Tracker
PRD §44 — Every experiment records: experiment_id, git_commit, dataset_version,
hardware_version, firmware_version, preprocessing_version, windowing_version,
feature_version, model, seed, split_manifest, metrics.
"""
from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN"


class ExperimentTracker:
    """
    Creates and saves experiment_manifest.json for every training run.
    PRD §44, §62.
    """

    def __init__(self, experiments_dir: Path) -> None:
        self.experiments_dir = Path(experiments_dir)

    def create(
        self,
        model: str,
        dataset_version: str,
        preprocessing_version: str,
        windowing_version: str,
        feature_version: str,
        split_manifest_path: str,
        seed: int,
        hardware_version: str = "TBD",
        firmware_version: str = "TBD",
        hyperparams: Optional[dict] = None,
        metrics: Optional[dict] = None,
        notes: str = "",
    ) -> dict:
        """
        Create a new experiment manifest.
        Returns the manifest dict and saves it to disk.
        """
        experiment_id = f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        manifest = {
            "experiment_id": experiment_id,
            "created": datetime.utcnow().isoformat(),
            "git_commit": _get_git_commit(),
            "dataset_version": dataset_version,
            "hardware_version": hardware_version,
            "firmware_version": firmware_version,
            "preprocessing_version": preprocessing_version,
            "windowing_version": windowing_version,
            "feature_version": feature_version,
            "model": model,
            "seed": seed,
            "split_manifest": split_manifest_path,
            "hyperparams": hyperparams or {},
            "metrics": metrics or {},
            "notes": notes,
        }

        out_dir = self.experiments_dir / experiment_id
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "experiment_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def load(self, experiment_id: str) -> dict:
        path = self.experiments_dir / experiment_id / "experiment_manifest.json"
        with open(path) as f:
            return json.load(f)

    def list_all(self) -> list:
        if not self.experiments_dir.exists():
            return []
        manifests = []
        for exp_dir in sorted(self.experiments_dir.iterdir()):
            manifest_file = exp_dir / "experiment_manifest.json"
            if manifest_file.exists():
                with open(manifest_file) as f:
                    manifests.append(json.load(f))
        return manifests
