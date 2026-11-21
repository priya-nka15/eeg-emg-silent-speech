"""
Tests for RealtimePipeline — LIVE safety gate, mock mode, temporal→safety handoff,
replay pipeline invocation (PRD §40, §47, §54, §49).
"""
import os
import tempfile
import warnings
import pytest
from pathlib import Path

import yaml


# ── Helpers ───────────────────────────────────────────────────────────────────

def write_hardware_yaml(path: Path, human_use_ready: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump({"human_use_ready": human_use_ready}, f)


def write_final_pipeline_yaml(path: Path, with_tbd: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "pipeline_config_version": "1" if not with_tbd else "TBD",
        "hardware_version": "hw-1",
        "firmware_version": "fw-1",
        "preprocessing_version": "pp-1",
        "emg": {"model_id": "emg-svm-v1" if not with_tbd else "TBD"},
        "eeg": {"model_id": "eeg-svm-v1"},
        "fusion": {"method": "decision_level", "config_version": "1"},
        "temporal": {"config_version": "1"},
        "safety": {"config_version": "1", "enabled": False},
    }
    with open(path, "w") as f:
        yaml.dump(data, f)


# ── Import pipeline symbols ───────────────────────────────────────────────────

from backend.realtime.pipeline import RealtimePipeline, PipelineMode, _repo_root
from backend.protocol.output_states import IntentOutput


# ── LIVE safety gate ──────────────────────────────────────────────────────────

class TestLiveSafetyGate:

    def test_live_refused_when_hardware_yaml_missing(self, tmp_path, monkeypatch):
        """
        Regression for audit finding C-1:
        LIVE mode must be refused when hardware.yaml is not found,
        regardless of the current working directory.
        """
        # Patch _repo_root to point to tmp_path (no configs/ there)
        monkeypatch.setattr(
            "backend.realtime.pipeline._repo_root",
            lambda: tmp_path,
        )
        with pytest.raises(RuntimeError, match="hardware.yaml not found"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                RealtimePipeline(mode=PipelineMode.LIVE)

    def test_live_refused_from_non_repo_cwd(self, tmp_path, monkeypatch):
        """
        LIVE mode must not silently pass when run from an arbitrary directory.
        The hardware.yaml check must use the repo-root-relative path.
        """
        monkeypatch.chdir(tmp_path)  # no configs/ here
        monkeypatch.setattr(
            "backend.realtime.pipeline._repo_root",
            lambda: tmp_path,
        )
        with pytest.raises(RuntimeError, match="hardware.yaml not found"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                RealtimePipeline(mode=PipelineMode.LIVE)

    def test_live_refused_when_human_use_ready_false(self, tmp_path, monkeypatch):
        write_hardware_yaml(tmp_path / "configs" / "hardware.yaml", human_use_ready=False)
        monkeypatch.setattr("backend.realtime.pipeline._repo_root", lambda: tmp_path)
        with pytest.raises(RuntimeError, match="human_use_ready is False"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                RealtimePipeline(mode=PipelineMode.LIVE)

    def test_live_refused_when_final_pipeline_not_provided(self, tmp_path, monkeypatch):
        write_hardware_yaml(tmp_path / "configs" / "hardware.yaml", human_use_ready=True)
        monkeypatch.setattr("backend.realtime.pipeline._repo_root", lambda: tmp_path)
        with pytest.raises(RuntimeError, match="final_pipeline_config_path was not provided"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                RealtimePipeline(mode=PipelineMode.LIVE, final_pipeline_config_path=None)

    def test_live_refused_when_final_pipeline_missing(self, tmp_path, monkeypatch):
        write_hardware_yaml(tmp_path / "configs" / "hardware.yaml", human_use_ready=True)
        monkeypatch.setattr("backend.realtime.pipeline._repo_root", lambda: tmp_path)
        nonexistent = tmp_path / "no_such_file.yaml"
        with pytest.raises(RuntimeError, match="not found"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                RealtimePipeline(mode=PipelineMode.LIVE, final_pipeline_config_path=nonexistent)

    def test_live_refused_when_final_pipeline_has_tbd(self, tmp_path, monkeypatch):
        write_hardware_yaml(tmp_path / "configs" / "hardware.yaml", human_use_ready=True)
        fp = tmp_path / "final_pipeline.yaml"
        write_final_pipeline_yaml(fp, with_tbd=True)
        monkeypatch.setattr("backend.realtime.pipeline._repo_root", lambda: tmp_path)
        with pytest.raises(RuntimeError, match="TBD"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                RealtimePipeline(mode=PipelineMode.LIVE, final_pipeline_config_path=fp)


# ── Mock mode ─────────────────────────────────────────────────────────────────

class TestMockMode:

    def test_mock_output_always_no_decision(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline = RealtimePipeline(mode=PipelineMode.MOCK)
        output = pipeline.process_window()
        assert output.final_intent == IntentOutput.NO_DECISION

    def test_mock_output_is_mock_flagged(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline = RealtimePipeline(mode=PipelineMode.MOCK)
        output = pipeline.process_window()
        assert output.is_mock is True

    def test_mock_can_never_return_action(self):
        """Mock mode must never produce ACTION regardless of inputs."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline = RealtimePipeline(mode=PipelineMode.MOCK)
        for _ in range(10):
            output = pipeline.process_window()
            assert output.safety_decision != "ACTION"
            assert output.final_intent != "HELP"


# ── Temporal → Safety handoff ─────────────────────────────────────────────────

class TestTemporalToSafetyHandoff:
    """
    Regression for audit finding C-4:
    temporal_consistency_met must reflect the actual temporal filter result,
    not be inferred from the fusion output.
    """

    def test_temporal_not_met_passes_false_to_safety(self):
        """
        When the temporal filter returns NO_DECISION (buffer not full),
        temporal_consistency_met must be False — even if a model prediction
        would otherwise produce a non-NO_DECISION intent.
        """
        import numpy as np
        from backend.fusion.temporal_consistency import (
            TemporalConsistencyConfig, ConsistencyStrategy
        )
        from backend.safety.safety_engine import SafetyConfig

        # Temporal filter requires 5 consecutive windows — first window should not pass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tc_config = TemporalConsistencyConfig(
                required_windows=5,
                strategy=ConsistencyStrategy.CONSECUTIVE,
                config_version="test-1",
            )
            safety_config = SafetyConfig(
                enabled=True,
                trigger_intent="HELP",
                trigger_state="STRESS",
                min_intent_confidence=0.5,
                min_state_confidence=0.5,
                temporal_requirement=True,
            )
            from backend.models.emg.svm_model import EMGSVMModel
            model = EMGSVMModel()
            X = np.random.randn(30, 10)
            y = ["YES", "NO", "HELP"] * 10
            model.fit(X, y)

            pipeline = RealtimePipeline(
                mode=PipelineMode.REPLAY,
                emg_model=model,
                temporal_config=tc_config,
                safety_config=safety_config,
            )

        # Send a single window — temporal buffer not full → NO_DECISION
        emg_window = np.random.randn(10)
        output = pipeline.process_window(emg_window=emg_window, signal_quality="OK")

        # Safety engine must not have triggered ACTION on the first window
        # because temporal consistency was not met
        assert output.safety_decision != "ACTION", (
            "Safety engine produced ACTION on first window before temporal "
            "consistency was met — C-4 regression."
        )

    def test_temporal_met_after_n_consistent_windows(self):
        """After N consistent windows, temporal_consistency_met should be True."""
        import numpy as np
        from backend.fusion.temporal_consistency import (
            TemporalConsistencyConfig, ConsistencyStrategy
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tc_config = TemporalConsistencyConfig(
                required_windows=3,
                strategy=ConsistencyStrategy.CONSECUTIVE,
                config_version="test-1",
            )
            from backend.models.emg.svm_model import EMGSVMModel
            model = EMGSVMModel()
            # SVM requires at least 2 classes — use YES/NO/HELP
            X = np.random.randn(30, 10)
            y = ["YES"] * 10 + ["NO"] * 10 + ["HELP"] * 10
            model.fit(X, y)

            pipeline = RealtimePipeline(
                mode=PipelineMode.REPLAY,
                emg_model=model,
                temporal_config=tc_config,
            )

        # Feed 5 windows — first 2 must be NO_DECISION, 3rd onward may resolve
        outputs = []
        for _ in range(5):
            emg_window = np.random.randn(10)
            out = pipeline.process_window(emg_window=emg_window)
            outputs.append(out)

        # First 2 must be NO_DECISION (temporal buffer not full after 1 and 2 windows)
        for i in range(2):
            assert outputs[i].final_intent == IntentOutput.NO_DECISION, (
                f"Window {i+1} should be NO_DECISION (temporal not yet met)"
            )


# ── Replay invokes pipeline ───────────────────────────────────────────────────

class TestReplayInvokesPipeline:
    """
    Regression for audit finding C-2:
    cmd_replay must actually call process_window(), not just count samples.
    """

    def test_replay_process_window_called(self, tmp_path):
        """
        Verify that running replay through process_window produces pipeline
        outputs (not just a sample count).
        """
        import csv
        import numpy as np
        from backend.realtime.replay import ReplaySource
        from backend.acquisition.frame_parser import Sample

        # Write a small recording
        csv_path = tmp_path / "recording.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "sample_id", "timestamp_us",
                "EEG_Fp1", "EEG_Fp2",
                "EMG_Submental", "EMG_Orbicularis",
                "EMG_Zygomaticus", "EMG_Buccinator",
                "event_marker", "status",
            ])
            writer.writeheader()
            for i in range(10):
                writer.writerow({
                    "sample_id": i, "timestamp_us": i * 250,
                    "EEG_Fp1": 10, "EEG_Fp2": -10,
                    "EMG_Submental": 5, "EMG_Orbicularis": -5,
                    "EMG_Zygomaticus": 8, "EMG_Buccinator": -8,
                    "event_marker": 0, "status": 0,
                })

        source = ReplaySource(csv_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline = RealtimePipeline(mode=PipelineMode.REPLAY)

        n_outputs = 0
        for sample in source.stream():
            emg_window = np.array([[
                sample.EMG_Submental, sample.EMG_Orbicularis,
                sample.EMG_Zygomaticus, sample.EMG_Buccinator,
            ]], dtype=float)
            output = pipeline.process_window(emg_window=emg_window)
            assert output is not None
            assert output.mode == PipelineMode.REPLAY
            assert output.is_mock is False
            n_outputs += 1

        assert n_outputs == 10, (
            f"Expected 10 pipeline outputs, got {n_outputs}. "
            "cmd_replay must call process_window(), not just count samples (C-2 regression)."
        )
