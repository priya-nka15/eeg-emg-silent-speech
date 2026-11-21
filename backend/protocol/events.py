"""
URF/A-SSR Event System
PRD §14 — Events synchronized to signal samples.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class EventCode(IntEnum):
    """Numeric event codes embedded in the sample stream (event_marker field)."""
    NONE                 = 0x00
    SESSION_START        = 0x01
    BASELINE_START       = 0x02
    BASELINE_END         = 0x03
    TRIAL_START          = 0x04
    CUE                  = 0x05
    ARTICULATION_START   = 0x06
    ARTICULATION_END     = 0x07
    RECOVERY_START       = 0x08
    TRIAL_END            = 0x09
    SESSION_END          = 0x0A
    STATE_TASK_START     = 0x0B
    STATE_TASK_END       = 0x0C
    STATE_RECOVERY_START = 0x0D
    STATE_RECOVERY_END   = 0x0E

    @classmethod
    def from_name(cls, name: str) -> "EventCode":
        return cls[name.upper()]

    @classmethod
    def name_to_code(cls) -> dict[str, int]:
        return {e.name: int(e) for e in cls}


@dataclass
class Event:
    """
    A single event, always associated with a sample_id so it can be
    precisely aligned to the signal timeline.
    """
    code: EventCode
    sample_id: int                       # sample_id at which this event fired
    timestamp_us: int                    # microseconds since session start
    label: Optional[str] = None          # e.g. "YES", "NO", "HELP", EEG condition
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "code": int(self.code),
            "code_name": self.code.name,
            "sample_id": self.sample_id,
            "timestamp_us": self.timestamp_us,
            "label": self.label,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(
            code=EventCode(d["code"]),
            sample_id=d["sample_id"],
            timestamp_us=d["timestamp_us"],
            label=d.get("label"),
            metadata=d.get("metadata", {}),
        )


class EventLog:
    """Ordered log of events for a session, indexed by sample_id."""

    def __init__(self) -> None:
        self._events: list[Event] = []

    def record(self, event: Event) -> None:
        self._events.append(event)

    def get_all(self) -> list[Event]:
        return list(self._events)

    def get_by_code(self, code: EventCode) -> list[Event]:
        return [e for e in self._events if e.code == code]

    def get_in_range(self, start_sample: int, end_sample: int) -> list[Event]:
        return [e for e in self._events
                if start_sample <= e.sample_id <= end_sample]

    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self._events]

    @classmethod
    def from_list(cls, data: list[dict]) -> "EventLog":
        log = cls()
        for d in data:
            log.record(Event.from_dict(d))
        return log

    def __len__(self) -> int:
        return len(self._events)
