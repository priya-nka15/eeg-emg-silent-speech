"""
URF/A-SSR Dataset Schema Validation
PRD §12 — Subject, Session, Trial, Sample data models with validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Re-export from session_manager for schema validation use
from backend.protocol.session_manager import (
    SubjectRecord,
    SessionRecord,
    TrialRecord,
    TrialValidity,
    INVALID_REASONS,
)
from backend.acquisition.frame_parser import Sample, CHANNEL_NAMES

__all__ = [
    "SubjectRecord",
    "SessionRecord",
    "TrialRecord",
    "TrialValidity",
    "INVALID_REASONS",
    "Sample",
    "CHANNEL_NAMES",
    "validate_sample",
    "validate_trial",
    "validate_session",
    "validate_subject",
]


def validate_subject(d: dict) -> list[str]:
    """Return list of validation errors for a subject dict."""
    errors = []
    if "subject_id" not in d:
        errors.append("Missing subject_id")
    elif not isinstance(d["subject_id"], str) or not d["subject_id"].startswith("S"):
        errors.append(f"subject_id should be a string starting with 'S', got {d['subject_id']!r}")
    if d.get("consent_status") not in ("CONSENTED", "WITHDRAWN", "PENDING"):
        errors.append(f"Invalid consent_status: {d.get('consent_status')!r}")
    # Privacy check: no obvious real names
    sid = d.get("subject_id", "")
    if any(c.isspace() for c in sid):
        errors.append("subject_id must not contain spaces (use pseudonymous ID like S001)")
    return errors


def validate_session(d: dict) -> list[str]:
    """Return list of validation errors for a session dict."""
    errors = []
    for required in ("session_id", "subject_id", "date", "time"):
        if required not in d:
            errors.append(f"Missing required field: {required}")
    return errors


def validate_trial(d: dict) -> list[str]:
    """Return list of validation errors for a trial dict."""
    errors = []
    required = ("trial_id", "subject_id", "session_id", "intent",
                 "start_sample", "end_sample", "validity")
    for r in required:
        if r not in d:
            errors.append(f"Missing required field: {r}")

    intent = d.get("intent")
    valid_intents = {"YES", "NO", "HELP", "BASELINE", "EEG_TASK"}
    if intent and intent not in valid_intents:
        errors.append(f"Unknown intent: {intent!r}")

    validity = d.get("validity")
    if validity not in (TrialValidity.VALID, TrialValidity.INVALID):
        errors.append(f"Invalid validity: {validity!r}")

    if validity == TrialValidity.INVALID:
        reason = d.get("invalid_reason")
        if not reason:
            errors.append("invalid_reason required when validity is INVALID")
        elif reason not in INVALID_REASONS:
            errors.append(f"Unknown invalid_reason: {reason!r}")

    return errors


def validate_sample(d: dict) -> list[str]:
    """Return list of validation errors for a sample dict."""
    errors = []
    required = ["sample_id", "timestamp_us"] + CHANNEL_NAMES
    for r in required:
        if r not in d:
            errors.append(f"Missing field: {r}")
        elif not isinstance(d[r], (int, float)):
            errors.append(f"Field {r} must be numeric, got {type(d[r]).__name__}")

    for ch in CHANNEL_NAMES:
        val = d.get(ch)
        if val is not None:
            if not isinstance(val, (int, float)):
                errors.append(f"{ch}: not numeric")
            elif not (-2**23 <= val <= 2**23 - 1):
                # 24-bit signed range
                errors.append(f"{ch}: value {val} out of 24-bit signed range")

    return errors
