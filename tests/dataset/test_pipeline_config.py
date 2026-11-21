"""
Tests for final_pipeline.yaml TBD enforcement — REQ-054, PRD §47, §49.
Verifies that LIVE mode refuses to start when mandatory fields are TBD,
and that the _flatten helper correctly detects nested TBD values.
"""
import tempfile
import warnings
import pytest
from pathlib import Path

import yaml

from backend.realtime.pipeline import _flatten, RealtimePipeline, PipelineMode


def write_hardware_yaml(path: Path, human_use_ready: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump({"human_use_ready": human_use_ready}, f)


class TestFlattenHelper:
    def test_detects_top_level_tbd(self):
        cfg = {"key": "TBD", "other": "value"}
        flat = _flatten(cfg)
        assert flat["key"] == "TBD"
        assert flat["other"] == "value"

    def test_detects_nested_tbd(self):
        cfg = {"emg": {"model_id": "TBD"}, "eeg": {"model_id": "eeg-svm-v1"}}
        flat = _flatten(cfg)
        assert flat["emg.model_id"] == "TBD"
        assert flat["eeg.model_id"] == "eeg-svm-v1"

    def test_no_tbd_returns_empty_tbd_list(self):
        cfg = {"key": "value", "nested": {"inner": "ok"}}
        tbd = [k for k, v in _flatten(cfg).items() if v == "TBD"]
        assert tbd == []


class TestPipelineConfigEnforcement:

    def test_live_refused_tbd_emg_model(self, tmp_path, monkeypatch):
        write_hardware_yaml(tmp_path / "configs" / "hardware.yaml", human_use_ready=True)
        fp = tmp_path / "final_pipeline.yaml"
        cfg = {
            "pipeline_config_version": "1",
            "hardware_version": "hw-1",
            "firmware_version": "fw-1",
            "preprocessing_version": "pp-1",
            "emg": {"model_id": "TBD"},     # TBD field
            "eeg": {"model_id": "eeg-svm-v1"},
            "fusion": {"method": "decision_level", "config_version": "1"},
            "temporal": {"config_version": "1"},
            "safety": {"config_version": "1", "enabled": False},
        }
        with open(fp, "w") as f:
            yaml.dump(cfg, f)
        monkeypatch.setattr("backend.realtime.pipeline._repo_root", lambda: tmp_path)
        with pytest.raises(RuntimeError, match="TBD"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                RealtimePipeline(mode=PipelineMode.LIVE, final_pipeline_config_path=fp)

    def test_live_refused_when_pipeline_config_is_none(self, tmp_path, monkeypatch):
        write_hardware_yaml(tmp_path / "configs" / "hardware.yaml", human_use_ready=True)
        monkeypatch.setattr("backend.realtime.pipeline._repo_root", lambda: tmp_path)
        with pytest.raises(RuntimeError, match="not provided"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                RealtimePipeline(mode=PipelineMode.LIVE, final_pipeline_config_path=None)

    def test_live_refused_when_config_file_missing(self, tmp_path, monkeypatch):
        write_hardware_yaml(tmp_path / "configs" / "hardware.yaml", human_use_ready=True)
        monkeypatch.setattr("backend.realtime.pipeline._repo_root", lambda: tmp_path)
        with pytest.raises(RuntimeError, match="not found"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                RealtimePipeline(
                    mode=PipelineMode.LIVE,
                    final_pipeline_config_path=tmp_path / "nonexistent.yaml"
                )

    def test_mock_and_replay_not_blocked_by_config(self):
        """MOCK and REPLAY modes must not require final_pipeline.yaml."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mock_p = RealtimePipeline(mode=PipelineMode.MOCK)
            replay_p = RealtimePipeline(mode=PipelineMode.REPLAY)
        assert mock_p.mode == PipelineMode.MOCK
        assert replay_p.mode == PipelineMode.REPLAY
