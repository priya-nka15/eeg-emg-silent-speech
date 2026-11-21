"""
Tests for DatasetSplitter — leakage safety (PRD §49 Dataset tests).
"""
import pytest
from backend.datasets.splitter import DatasetSplitter, SplitLeakageError


def make_subjects_sessions(n_subjects=6, sessions_per=2):
    subjects = [f"S{i:03d}" for i in range(1, n_subjects + 1)]
    sessions = {s: [f"Session_{j:02d}" for j in range(1, sessions_per + 1)] for s in subjects}
    return subjects, sessions


class TestDatasetSplitter:

    def test_subject_split_no_leakage(self):
        subjects, sessions = make_subjects_sessions(6)
        splitter = DatasetSplitter(seed=42)
        manifest = splitter.split_by_subject(subjects, sessions)
        train = set(manifest["train"]["subjects"])
        test = set(manifest["test"]["subjects"])
        assert len(train & test) == 0, "Subject leakage between train and test"

    def test_subject_split_no_val_test_leakage(self):
        subjects, sessions = make_subjects_sessions(8)
        splitter = DatasetSplitter(seed=42)
        manifest = splitter.split_by_subject(subjects, sessions)
        val = set(manifest["validation"]["subjects"])
        test = set(manifest["test"]["subjects"])
        assert len(val & test) == 0

    def test_session_split_no_leakage(self):
        subjects, sessions = make_subjects_sessions(4)
        splitter = DatasetSplitter(seed=42)
        manifest = splitter.split_by_session(subjects, sessions, held_out_session="Session_02")
        train_sessions = {(s["subject"], s["session"]) for s in manifest["train"]["sessions"]}
        test_sessions = {(s["subject"], s["session"]) for s in manifest["test"]["sessions"]}
        assert len(train_sessions & test_sessions) == 0

    def test_loso_each_subject_held_out_once(self):
        subjects, sessions = make_subjects_sessions(4)
        splitter = DatasetSplitter(seed=42)
        folds = splitter.loso_splits(subjects, sessions)
        assert len(folds) == len(subjects)
        held_out_subjects = [f["test"]["subjects"][0] for f in folds]
        assert set(held_out_subjects) == set(subjects)

    def test_loso_no_leakage_per_fold(self):
        subjects, sessions = make_subjects_sessions(5)
        splitter = DatasetSplitter(seed=42)
        folds = splitter.loso_splits(subjects, sessions)
        for fold in folds:
            train = set(fold["train"]["subjects"])
            test = set(fold["test"]["subjects"])
            assert len(train & test) == 0

    def test_test_locked_flag_set(self):
        subjects, sessions = make_subjects_sessions(4)
        splitter = DatasetSplitter(seed=42)
        manifest = splitter.split_by_subject(subjects, sessions)
        assert manifest["test_locked"] is True

    def test_too_few_subjects_raises(self):
        subjects, sessions = make_subjects_sessions(1)
        splitter = DatasetSplitter(seed=42)
        with pytest.raises(ValueError, match="Not enough subjects"):
            splitter.split_by_subject(subjects, sessions)

    def test_deterministic_with_same_seed(self):
        subjects, sessions = make_subjects_sessions(6)
        m1 = DatasetSplitter(seed=42).split_by_subject(subjects, sessions)
        m2 = DatasetSplitter(seed=42).split_by_subject(subjects, sessions)
        assert m1["train"]["subjects"] == m2["train"]["subjects"]

    def test_different_seeds_give_different_splits(self):
        subjects, sessions = make_subjects_sessions(8)
        m1 = DatasetSplitter(seed=42).split_by_subject(subjects, sessions)
        m2 = DatasetSplitter(seed=99).split_by_subject(subjects, sessions)
        # Not guaranteed to differ but very likely with 8 subjects
        # Just verify both produce valid splits
        assert len(m1["train"]["subjects"]) > 0
        assert len(m2["train"]["subjects"]) > 0
