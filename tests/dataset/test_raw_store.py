"""
Tests for RawStore immutability policy (PRD §13, §49).
"""
import pytest
from pathlib import Path
import tempfile

from backend.datasets.raw_store import RawStore, RawStoreError
from backend.acquisition.frame_parser import Sample


def make_sample(sample_id: int) -> Sample:
    return Sample(
        sample_id=sample_id,
        timestamp_us=sample_id * 250,
        EEG_Fp1=10, EEG_Fp2=20,
        EMG_Submental=30, EMG_Orbicularis=40,
        EMG_Zygomaticus=50, EMG_Buccinator=60,
    )


class TestRawStore:

    def test_write_and_read_recording(self, tmp_path):
        store = RawStore(tmp_path)
        with store.open_recording("S001", "Session_01") as writer:
            for i in range(5):
                writer.write_sample(make_sample(i))
            assert writer.n_samples == 5

        samples = list(store.read_raw_csv("S001", "Session_01"))
        assert len(samples) == 5
        assert samples[0]["sample_id"] == 0
        assert samples[4]["sample_id"] == 4

    def test_immutability_blocks_overwrite(self, tmp_path):
        store = RawStore(tmp_path)
        # Write once
        with store.open_recording("S001", "Session_01") as writer:
            writer.write_sample(make_sample(0))
        # Attempting to write again to the same session must raise
        with pytest.raises(RawStoreError, match="immutable"):
            store.open_recording("S001", "Session_01")

    def test_different_sessions_independent(self, tmp_path):
        store = RawStore(tmp_path)
        with store.open_recording("S001", "Session_01") as w:
            w.write_sample(make_sample(0))
        with store.open_recording("S001", "Session_02") as w:
            w.write_sample(make_sample(0))
        # Both sessions should exist
        assert store.raw_dir("S001", "Session_01").exists()
        assert store.raw_dir("S001", "Session_02").exists()

    def test_missing_recording_raises(self, tmp_path):
        store = RawStore(tmp_path)
        with pytest.raises(FileNotFoundError):
            list(store.read_raw_csv("S999", "Session_01"))

    def test_list_subjects(self, tmp_path):
        store = RawStore(tmp_path)
        for subj in ["S001", "S002", "S003"]:
            with store.open_recording(subj, "Session_01") as w:
                w.write_sample(make_sample(0))
        subjects = store.list_subjects()
        assert "S001" in subjects
        assert "S002" in subjects
        assert "S003" in subjects

    def test_list_sessions(self, tmp_path):
        store = RawStore(tmp_path)
        for sess in ["Session_01", "Session_02"]:
            with store.open_recording("S001", sess) as w:
                w.write_sample(make_sample(0))
        sessions = store.list_sessions("S001")
        assert "Session_01" in sessions
        assert "Session_02" in sessions
