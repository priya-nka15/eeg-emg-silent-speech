"""
Tests for replay mode reproducibility (PRD §49 Replay).
"""
import csv
import tempfile
import pytest
from pathlib import Path

from backend.realtime.replay import ReplaySource
from backend.acquisition.frame_parser import Sample


def write_test_recording(path: Path, n=20):
    samples = []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_id", "timestamp_us",
            "EEG_Fp1", "EEG_Fp2",
            "EMG_Submental", "EMG_Orbicularis",
            "EMG_Zygomaticus", "EMG_Buccinator",
            "event_marker", "status",
        ])
        writer.writeheader()
        for i in range(n):
            row = {
                "sample_id": i,
                "timestamp_us": i * 250,
                "EEG_Fp1": i * 10,
                "EEG_Fp2": i * -10,
                "EMG_Submental": i,
                "EMG_Orbicularis": i + 1,
                "EMG_Zygomaticus": i + 2,
                "EMG_Buccinator": i + 3,
                "event_marker": 0,
                "status": 0,
            }
            writer.writerow(row)
            samples.append(row)
    return samples


class TestReplaySource:

    def test_loads_all_samples(self, tmp_path):
        csv_path = tmp_path / "recording.csv"
        write_test_recording(csv_path, n=20)
        source = ReplaySource(csv_path)
        samples = source.load_all()
        assert len(samples) == 20

    def test_sample_values_correct(self, tmp_path):
        csv_path = tmp_path / "recording.csv"
        write_test_recording(csv_path, n=5)
        source = ReplaySource(csv_path)
        samples = source.load_all()
        assert samples[0].sample_id == 0
        assert samples[4].sample_id == 4
        assert samples[2].EEG_Fp1 == 20
        assert samples[2].EEG_Fp2 == -20

    def test_reproducible_replay(self, tmp_path):
        csv_path = tmp_path / "recording.csv"
        write_test_recording(csv_path, n=10)
        source = ReplaySource(csv_path)
        run1 = source.load_all()
        run2 = source.load_all()
        for s1, s2 in zip(run1, run2):
            assert s1.sample_id == s2.sample_id
            assert s1.EEG_Fp1 == s2.EEG_Fp1
            assert s1.EMG_Buccinator == s2.EMG_Buccinator

    def test_is_replay_flag(self, tmp_path):
        csv_path = tmp_path / "recording.csv"
        write_test_recording(csv_path, n=1)
        source = ReplaySource(csv_path)
        assert source.is_replay is True

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ReplaySource(tmp_path / "nonexistent.csv")

    def test_stream_yields_samples(self, tmp_path):
        csv_path = tmp_path / "recording.csv"
        write_test_recording(csv_path, n=5)
        source = ReplaySource(csv_path)
        count = sum(1 for _ in source.stream())
        assert count == 5
