"""
URF/A-SSR Decision-Level Fusion
PRD §34 — Decision-level fusion. First implementation.
No raw concatenation into a large multimodal network.
Supports missing/invalid modalities.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

from backend.models.base import Prediction
from backend.protocol.output_states import IntentOutput


@dataclass
class FusionResult:
    """Output of the fusion layer."""
    intent: str                      # YES | NO | HELP | NO_DECISION
    emg_prediction: Optional[Prediction] = None
    eeg_prediction: Optional[Prediction] = None
    fusion_method: str = "decision_level"
    reason: str = ""

    @property
    def is_decided(self) -> bool:
        return self.intent != IntentOutput.NO_DECISION


class DecisionFusion:
    """
    Decision-level fusion of EMG intent and EEG state predictions.

    Strategy:
    1. If EMG is unavailable → NO_DECISION (intent requires EMG)
    2. If EMG is NO_DECISION → NO_DECISION
    3. If EEG is unavailable → pass EMG intent through (with degraded confidence note)
    4. If EEG is NO_DECISION → pass EMG intent (EEG does not block intent)
    5. If both available → use EMG intent as primary; EEG state informs safety engine
    """

    def __init__(self, require_eeg: bool = False) -> None:
        """
        Args:
            require_eeg: If True, both EMG and EEG must be available.
                         Defaults to False (EMG drives intent; EEG is supplementary).
        """
        self.require_eeg = require_eeg

    def fuse(
        self,
        emg_prediction: Optional[Prediction],
        eeg_prediction: Optional[Prediction],
    ) -> FusionResult:
        """
        Fuse EMG intent and EEG state into a final decision.

        Args:
            emg_prediction: EMG model output, or None if EMG unavailable.
            eeg_prediction: EEG model output, or None if EEG unavailable.

        Returns:
            FusionResult
        """
        # EMG unavailable
        if emg_prediction is None:
            return FusionResult(
                intent=IntentOutput.NO_DECISION,
                eeg_prediction=eeg_prediction,
                reason="EMG unavailable",
            )

        # EMG is NO_DECISION
        if emg_prediction.label == IntentOutput.NO_DECISION:
            return FusionResult(
                intent=IntentOutput.NO_DECISION,
                emg_prediction=emg_prediction,
                eeg_prediction=eeg_prediction,
                reason=f"EMG NO_DECISION: {emg_prediction.notes}",
            )

        # EEG required but unavailable
        if self.require_eeg and eeg_prediction is None:
            return FusionResult(
                intent=IntentOutput.NO_DECISION,
                emg_prediction=emg_prediction,
                reason="EEG required but unavailable",
            )

        # EEG unavailable — proceed with EMG only
        if eeg_prediction is None:
            return FusionResult(
                intent=emg_prediction.label,
                emg_prediction=emg_prediction,
                eeg_prediction=None,
                reason="EEG unavailable — using EMG intent only (degraded)",
            )

        # Both available — EMG drives intent; EEG informs safety engine
        intent = emg_prediction.label
        reason = "EMG intent confirmed"
        if eeg_prediction.label == IntentOutput.NO_DECISION:
            reason += "; EEG NO_DECISION (not blocking)"

        return FusionResult(
            intent=intent,
            emg_prediction=emg_prediction,
            eeg_prediction=eeg_prediction,
            reason=reason,
        )
