"""
URF/A-SSR Output States
PRD §38 — Final output and internal states.
"""
from __future__ import annotations

from enum import Enum


class IntentOutput(str, Enum):
    """User-facing intent output states."""
    YES         = "YES"
    NO          = "NO"
    HELP        = "HELP"
    NO_DECISION = "NO_DECISION"


class InternalState(str, Enum):
    """Internal diagnostic states, not exposed directly to users."""
    SIGNAL_INVALID  = "SIGNAL_INVALID"
    CHANNEL_FAULT   = "CHANNEL_FAULT"
    STATE_UNCERTAIN = "STATE_UNCERTAIN"
    MODEL_UNCERTAIN = "MODEL_UNCERTAIN"
    OK              = "OK"


VALID_INTENT_LABELS = frozenset(["YES", "NO", "HELP"])
"""The three EMG command classes. BASELINE and EEG_TASK are not intent labels."""
