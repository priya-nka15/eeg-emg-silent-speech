"""
Tests for the event system (PRD §49 Acquisition).
"""
import pytest
from backend.protocol.events import Event, EventCode, EventLog


class TestEventSystem:

    def test_all_event_codes_defined(self):
        required = [
            "SESSION_START", "BASELINE_START", "BASELINE_END",
            "TRIAL_START", "CUE", "ARTICULATION_START", "ARTICULATION_END",
            "RECOVERY_START", "TRIAL_END", "SESSION_END",
            "STATE_TASK_START", "STATE_TASK_END",
            "STATE_RECOVERY_START", "STATE_RECOVERY_END",
        ]
        for name in required:
            assert hasattr(EventCode, name), f"Missing EventCode: {name}"

    def test_event_creation(self):
        e = Event(
            code=EventCode.TRIAL_START,
            sample_id=1000,
            timestamp_us=250000,
            label="YES",
        )
        assert e.code == EventCode.TRIAL_START
        assert e.sample_id == 1000
        assert e.label == "YES"

    def test_event_to_dict(self):
        e = Event(code=EventCode.CUE, sample_id=42, timestamp_us=10500)
        d = e.to_dict()
        assert d["code"] == int(EventCode.CUE)
        assert d["code_name"] == "CUE"
        assert d["sample_id"] == 42

    def test_event_roundtrip(self):
        e = Event(code=EventCode.SESSION_END, sample_id=9999, timestamp_us=2500000)
        recovered = Event.from_dict(e.to_dict())
        assert recovered.code == e.code
        assert recovered.sample_id == e.sample_id

    def test_event_log_record_and_retrieve(self):
        log = EventLog()
        e1 = Event(code=EventCode.TRIAL_START, sample_id=0, timestamp_us=0, label="YES")
        e2 = Event(code=EventCode.TRIAL_END, sample_id=400, timestamp_us=100000, label="YES")
        log.record(e1)
        log.record(e2)
        assert len(log) == 2
        assert log.get_by_code(EventCode.TRIAL_START)[0].label == "YES"

    def test_event_log_get_in_range(self):
        log = EventLog()
        for i, code in enumerate([EventCode.TRIAL_START, EventCode.CUE, EventCode.TRIAL_END]):
            log.record(Event(code=code, sample_id=i * 100, timestamp_us=i * 25000))
        events = log.get_in_range(50, 150)
        assert len(events) == 1
        assert events[0].code == EventCode.CUE

    def test_event_log_serialization(self):
        log = EventLog()
        log.record(Event(code=EventCode.SESSION_START, sample_id=0, timestamp_us=0))
        log.record(Event(code=EventCode.SESSION_END, sample_id=4000, timestamp_us=1000000))
        data = log.to_list()
        restored = EventLog.from_list(data)
        assert len(restored) == 2
        assert restored.get_by_code(EventCode.SESSION_START)[0].sample_id == 0
