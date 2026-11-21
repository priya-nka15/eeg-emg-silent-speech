"""
Tests for SessionManager — session lifecycle, timestamps, consent enforcement,
event ordering, trial preservation (PRD §12, §14, §15, §49).
"""
import tempfile
import pytest
from pathlib import Path

from backend.protocol.session_manager import (
    SessionManager, SubjectRecord, SessionRecord, TrialRecord, TrialValidity
)
from backend.protocol.events import EventCode


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def consented_subject():
    return SubjectRecord("S001", "CONSENTED")


@pytest.fixture
def session_record():
    return SessionRecord("S001_Sess01", "S001", "2026-10-08", "09:00")


@pytest.fixture
def manager(consented_subject, session_record, tmp_path):
    return SessionManager(consented_subject, session_record, tmp_path)


# ── Consent enforcement ───────────────────────────────────────────────────────

class TestConsentEnforcement:
    def test_consented_subject_ok(self, tmp_path):
        s = SubjectRecord("S001", "CONSENTED")
        sess = SessionRecord("S001_Sess01", "S001", "2026-10-08", "09:00")
        mgr = SessionManager(s, sess, tmp_path)
        assert mgr is not None

    def test_pending_consent_blocked(self, tmp_path):
        s = SubjectRecord("S001", "PENDING")
        sess = SessionRecord("S001_Sess01", "S001", "2026-10-08", "09:00")
        with pytest.raises(ValueError, match="has not consented"):
            SessionManager(s, sess, tmp_path)

    def test_withdrawn_consent_blocked(self, tmp_path):
        s = SubjectRecord("S001", "WITHDRAWN")
        sess = SessionRecord("S001_Sess01", "S001", "2026-10-08", "09:00")
        with pytest.raises(ValueError, match="has not consented"):
            SessionManager(s, sess, tmp_path)


# ── Timestamp sentinel ────────────────────────────────────────────────────────

class TestSessionTimestamps:
    def test_now_us_before_start_raises(self, manager):
        """
        Calling _now_us() before start_session() must raise RuntimeError.
        Prevents events from receiving epoch-relative timestamps (audit C-H7).
        """
        with pytest.raises(RuntimeError, match="start_session"):
            manager._now_us()

    def test_start_trial_before_start_session_raises(self, manager):
        """start_trial calls _now_us() internally."""
        with pytest.raises(RuntimeError, match="start_session"):
            manager.start_trial("YES", randomization_position=0)

    def test_now_us_after_start_is_small_positive(self, manager):
        manager.start_session()
        t = manager._now_us()
        assert t >= 0
        assert t < 10_000_000  # less than 10 seconds in µs

    def test_timestamps_increase_across_events(self, manager):
        manager.start_session()
        t1 = manager._now_us()
        manager.advance_sample(100)
        t2 = manager._now_us()
        # t2 >= t1 (time only moves forward)
        assert t2 >= t1

    def test_event_timestamps_are_session_relative(self, manager):
        manager.start_session()
        events = manager.event_log.get_by_code(EventCode.SESSION_START)
        assert len(events) == 1
        # Should be close to 0 µs (session just started)
        assert events[0].timestamp_us >= 0
        assert events[0].timestamp_us < 1_000_000  # less than 1 second


# ── Session lifecycle ─────────────────────────────────────────────────────────

class TestSessionLifecycle:
    def test_session_start_records_event(self, manager):
        manager.start_session()
        events = manager.event_log.get_by_code(EventCode.SESSION_START)
        assert len(events) == 1

    def test_session_end_saves_metadata(self, manager, tmp_path):
        manager.start_session()
        manager.end_session()
        assert (tmp_path / "metadata" / "session.json").exists()
        assert (tmp_path / "metadata" / "subject.json").exists()
        assert (tmp_path / "metadata" / "trials.json").exists()
        assert (tmp_path / "events" / "events.json").exists()

    def test_baseline_events_recorded(self, manager):
        manager.start_session()
        manager.start_baseline()
        manager.end_baseline()
        assert len(manager.event_log.get_by_code(EventCode.BASELINE_START)) == 1
        assert len(manager.event_log.get_by_code(EventCode.BASELINE_END)) == 1


# ── Trial lifecycle ───────────────────────────────────────────────────────────

class TestTrialLifecycle:
    def test_trial_records_all_events(self, manager):
        manager.start_session()
        trial = manager.start_trial("YES", randomization_position=0)
        manager.cue(trial)
        manager.articulation_start(trial)
        manager.articulation_end(trial)
        manager.recovery_start(trial)
        manager.end_trial(trial)
        assert len(manager.event_log.get_by_code(EventCode.TRIAL_START)) == 1
        assert len(manager.event_log.get_by_code(EventCode.CUE)) == 1
        assert len(manager.event_log.get_by_code(EventCode.ARTICULATION_START)) == 1
        assert len(manager.event_log.get_by_code(EventCode.TRIAL_END)) == 1

    def test_trial_ids_are_unique(self, manager):
        manager.start_session()
        t1 = manager.start_trial("YES", 0)
        manager.end_trial(t1)
        t2 = manager.start_trial("NO", 1)
        manager.end_trial(t2)
        assert t1.trial_id != t2.trial_id

    def test_trial_sample_ids_recorded(self, manager):
        manager.start_session()
        manager.advance_sample(100)
        trial = manager.start_trial("HELP", 0)
        manager.advance_sample(200)
        manager.end_trial(trial)
        assert trial.start_sample == 100
        assert trial.end_sample == 300


# ── Invalid trial preservation ────────────────────────────────────────────────

class TestInvalidTrialPreservation:
    def test_mark_invalid_preserves_trial(self, manager):
        """Invalid trials must be kept, not deleted (PRD §15, rule 11)."""
        manager.start_session()
        trial = manager.start_trial("YES", 0)
        manager.end_trial(trial)
        manager.mark_trial_invalid(trial, "saturation")
        # Trial still in the list
        assert trial in manager.trials
        assert trial.validity == TrialValidity.INVALID
        assert trial.invalid_reason == "saturation"

    def test_invalid_reason_required(self, manager):
        manager.start_session()
        trial = manager.start_trial("YES", 0)
        manager.end_trial(trial)
        with pytest.raises(ValueError, match="Unknown reason"):
            manager.mark_trial_invalid(trial, "bad_invented_reason")

    def test_all_invalid_reasons_accepted(self, manager):
        from backend.protocol.session_manager import INVALID_REASONS
        manager.start_session()
        for reason in INVALID_REASONS:
            trial = manager.start_trial("YES", 0)
            manager.end_trial(trial)
            manager.mark_trial_invalid(trial, reason)
            assert trial.validity == TrialValidity.INVALID

    def test_trial_counts_separated_by_validity(self, manager):
        manager.start_session()
        t1 = manager.start_trial("YES", 0)
        manager.end_trial(t1)
        t2 = manager.start_trial("YES", 1)
        manager.end_trial(t2)
        manager.mark_trial_invalid(t2, "vocalization")
        counts = manager.trial_counts()
        assert counts["valid"]["YES"] == 1
        assert counts["invalid"]["YES"] == 1
