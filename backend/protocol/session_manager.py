"""
URF/A-SSR Session Manager
PRD §12, §14, §18, §20, §24 — Session/trial workflow, event markers, baseline,
randomized sequences, metadata.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import List, Optional

from backend.protocol.events import Event, EventCode, EventLog
from backend.protocol.output_states import VALID_INTENT_LABELS


class TrialValidity(str, Enum):
    VALID   = "VALID"
    INVALID = "INVALID"


INVALID_REASONS = frozenset([
    "vocalization",
    "wrong_command",
    "movement",
    "swallowing",
    "coughing",
    "electrode_detachment",
    "poor_contact",
    "saturation",
    "missing_samples",
    "transmission_failure",
    "timing_error",
    "interruption",
    "protocol_violation",
])


@dataclass
class SubjectRecord:
    """PRD §12 Subject data model. Pseudonymous IDs only."""
    subject_id: str                      # e.g. "S001" — no real names
    consent_status: str = "PENDING"      # CONSENTED | WITHDRAWN | PENDING
    allowed_demographic_metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        valid = {"CONSENTED", "WITHDRAWN", "PENDING"}
        if self.consent_status not in valid:
            raise ValueError(f"consent_status must be one of {valid}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SubjectRecord":
        return cls(**d)


@dataclass
class SessionRecord:
    """PRD §12 Session data model."""
    session_id: str
    subject_id: str
    date: str                            # ISO 8601
    time: str                            # ISO 8601
    hardware_version: str = "TBD"
    firmware_version: str = "TBD"
    electrode_configuration: str = "1"
    gain_eeg: object = "TBD"
    gain_emg: object = "TBD"
    sampling_rate_hz: int = 4000
    measured_sampling_rate_hz: Optional[float] = None
    preprocessing_version: str = "TBD"
    protocol_version: str = "1.0"
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SessionRecord":
        return cls(**d)


@dataclass
class TrialRecord:
    """
    PRD §12, §15 Trial data model.
    Invalid trials are preserved with original signal and reason.
    """
    trial_id: str
    subject_id: str
    session_id: str
    intent: str                          # YES | NO | HELP | BASELINE | EEG_TASK
    state: str = "TBD"                   # EEG condition label (TBD until pilot)
    start_sample: int = 0
    end_sample: int = 0
    start_timestamp: int = 0             # µs
    end_timestamp: int = 0               # µs
    validity: str = TrialValidity.VALID
    invalid_reason: Optional[str] = None
    randomization_position: int = 0
    notes: Optional[str] = None

    def __post_init__(self):
        if self.intent not in VALID_INTENT_LABELS | {"BASELINE", "EEG_TASK"}:
            raise ValueError(f"Unknown intent: {self.intent!r}")
        if self.validity == TrialValidity.INVALID and not self.invalid_reason:
            raise ValueError("invalid_reason required when validity is INVALID")
        if self.invalid_reason and self.invalid_reason not in INVALID_REASONS:
            raise ValueError(f"Unknown invalid_reason: {self.invalid_reason!r}. "
                             f"Valid reasons: {sorted(INVALID_REASONS)}")

    def mark_invalid(self, reason: str) -> None:
        if reason not in INVALID_REASONS:
            raise ValueError(f"Unknown reason: {reason!r}")
        self.validity = TrialValidity.INVALID
        self.invalid_reason = reason

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrialRecord":
        return cls(**d)


class SessionManager:
    """
    Manages the lifecycle of a recording session.
    Produces event-annotated trial metadata for raw storage.
    """

    def __init__(
        self,
        subject: SubjectRecord,
        session: SessionRecord,
        output_dir: Path,
    ) -> None:
        if subject.consent_status != "CONSENTED":
            raise ValueError(
                f"Subject {subject.subject_id} has not consented "
                f"(status: {subject.consent_status})"
            )
        self.subject = subject
        self.session = session
        self.output_dir = output_dir
        self.event_log = EventLog()
        self.trials: List[TrialRecord] = []
        self._current_sample_id: int = 0
        # Sentinel: -1 means start_session() has not been called.
        # _now_us() will raise RuntimeError if called before start_session().
        self._session_start_us: int = -1

    # ── sample counter ────────────────────────────────────────────────────────

    def advance_sample(self, n: int = 1) -> None:
        """Called by the acquisition layer to advance the sample counter."""
        self._current_sample_id += n

    def _now_us(self) -> int:
        """
        Return microseconds elapsed since start_session() was called.

        Raises RuntimeError if called before start_session() — this prevents
        events from receiving wildly incorrect epoch-relative timestamps
        (audit finding H-7).
        """
        if self._session_start_us == -1:
            raise RuntimeError(
                "SessionManager._now_us() called before start_session(). "
                "Call start_session() first to establish the session timeline."
            )
        return int((time.monotonic_ns() // 1000) - self._session_start_us)

    # ── session lifecycle ─────────────────────────────────────────────────────

    def start_session(self) -> None:
        self._session_start_us = time.monotonic_ns() // 1000
        self._record_event(EventCode.SESSION_START)

    def end_session(self) -> None:
        self._record_event(EventCode.SESSION_END)
        self._save_metadata()

    # ── baseline ──────────────────────────────────────────────────────────────

    def start_baseline(self) -> int:
        """Returns sample_id of baseline start event."""
        return self._record_event(EventCode.BASELINE_START)

    def end_baseline(self) -> int:
        return self._record_event(EventCode.BASELINE_END)

    # ── trial lifecycle ───────────────────────────────────────────────────────

    def start_trial(
        self,
        intent: str,
        randomization_position: int,
        state: str = "TBD",
    ) -> TrialRecord:
        """Begin a new trial. Returns the TrialRecord."""
        trial_id = f"{self.session.session_id}_T{len(self.trials) + 1:04d}"
        start_sample = self._current_sample_id
        start_ts = self._now_us()
        self._record_event(EventCode.TRIAL_START, label=intent)

        trial = TrialRecord(
            trial_id=trial_id,
            subject_id=self.subject.subject_id,
            session_id=self.session.session_id,
            intent=intent,
            state=state,
            start_sample=start_sample,
            start_timestamp=start_ts,
            randomization_position=randomization_position,
        )
        self.trials.append(trial)
        return trial

    def cue(self, trial: TrialRecord) -> None:
        self._record_event(EventCode.CUE, label=trial.intent)

    def articulation_start(self, trial: TrialRecord) -> None:
        self._record_event(EventCode.ARTICULATION_START, label=trial.intent)

    def articulation_end(self, trial: TrialRecord) -> None:
        self._record_event(EventCode.ARTICULATION_END, label=trial.intent)

    def recovery_start(self, trial: TrialRecord) -> None:
        self._record_event(EventCode.RECOVERY_START)

    def end_trial(self, trial: TrialRecord) -> None:
        trial.end_sample = self._current_sample_id
        trial.end_timestamp = self._now_us()
        self._record_event(EventCode.TRIAL_END, label=trial.intent)

    def mark_trial_invalid(self, trial: TrialRecord, reason: str) -> None:
        """
        Mark a trial invalid. Preserves original signal (PRD §15, rule 11).
        Never silently deletes data.
        """
        trial.mark_invalid(reason)

    # ── EEG state task ────────────────────────────────────────────────────────

    def start_state_task(self, label: str) -> None:
        self._record_event(EventCode.STATE_TASK_START, label=label)

    def end_state_task(self) -> None:
        self._record_event(EventCode.STATE_TASK_END)

    def start_state_recovery(self) -> None:
        self._record_event(EventCode.STATE_RECOVERY_START)

    def end_state_recovery(self) -> None:
        self._record_event(EventCode.STATE_RECOVERY_END)

    # ── internals ─────────────────────────────────────────────────────────────

    def _record_event(self, code: EventCode, label: Optional[str] = None) -> int:
        sample_id = self._current_sample_id
        ts = self._now_us()
        self.event_log.record(Event(
            code=code,
            sample_id=sample_id,
            timestamp_us=ts,
            label=label,
        ))
        return sample_id

    def _save_metadata(self) -> None:
        meta_dir = self.output_dir / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        events_dir = self.output_dir / "events"
        events_dir.mkdir(parents=True, exist_ok=True)

        with open(meta_dir / "session.json", "w") as f:
            json.dump(self.session.to_dict(), f, indent=2)

        with open(meta_dir / "subject.json", "w") as f:
            json.dump(self.subject.to_dict(), f, indent=2)

        with open(meta_dir / "trials.json", "w") as f:
            json.dump([t.to_dict() for t in self.trials], f, indent=2)

        with open(events_dir / "events.json", "w") as f:
            json.dump(self.event_log.to_list(), f, indent=2)

    # ── statistics ────────────────────────────────────────────────────────────

    def trial_counts(self) -> dict:
        counts: dict[str, int] = {}
        valid: dict[str, int] = {}
        invalid: dict[str, int] = {}
        for t in self.trials:
            counts[t.intent] = counts.get(t.intent, 0) + 1
            if t.validity == TrialValidity.VALID:
                valid[t.intent] = valid.get(t.intent, 0) + 1
            else:
                invalid[t.intent] = invalid.get(t.intent, 0) + 1
        return {"total": counts, "valid": valid, "invalid": invalid}
