"""
URF/A-SSR Confidence and Rejection
PRD §35 — Every model exposes predicted label, probability/confidence, quality state.
Returns NO_DECISION when confidence is too low, quality inadequate, etc.
Thresholds must be validated, not invented.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional

from backend.models.base import Prediction
from backend.protocol.output_states import IntentOutput


@dataclass
class ConfidenceConfig:
    """
    Confidence thresholds for rejection.
    All values are TBD until validated from pilot/validation data.
    """
    emg_min_confidence: Optional[float] = None  # PILOT-VALIDATED
    eeg_min_confidence: Optional[float] = None  # PILOT-VALIDATED
    config_version: str = "TBD"

    def validate(self) -> list:
        issues = []
        if self.emg_min_confidence is None:
            issues.append("emg_min_confidence is TBD")
        if self.eeg_min_confidence is None:
            issues.append("eeg_min_confidence is TBD")
        if self.config_version == "TBD":
            issues.append("config_version is TBD")
        return issues

    def is_configured(self) -> bool:
        return (
            self.emg_min_confidence is not None
            and self.eeg_min_confidence is not None
        )


class ConfidenceGate:
    """
    Applies confidence-based rejection.
    Returns NO_DECISION when evidence is insufficient.
    """

    def __init__(self, config: ConfidenceConfig) -> None:
        self.config = config
        issues = config.validate()
        if issues:
            warnings.warn(
                f"ConfidenceConfig has TBD fields: {issues}. "
                "Rejection thresholds not active until configured.",
                UserWarning,
                stacklevel=2,
            )

    def check_emg(
        self,
        prediction: Prediction,
        signal_quality_ok: bool = True,
    ) -> Prediction:
        """
        Gate EMG prediction. Returns NO_DECISION if:
        - signal quality is not OK
        - confidence < emg_min_confidence (if configured)
        """
        if not signal_quality_ok:
            return Prediction.no_decision("EMG signal quality not OK")

        if not prediction.quality_ok:
            return Prediction.no_decision("EMG prediction quality flag not OK")

        if (
            self.config.emg_min_confidence is not None
            and prediction.confidence < self.config.emg_min_confidence
        ):
            return Prediction.no_decision(
                f"EMG confidence {prediction.confidence:.3f} < "
                f"threshold {self.config.emg_min_confidence}"
            )

        return prediction

    def check_eeg(
        self,
        prediction: Prediction,
        signal_quality_ok: bool = True,
    ) -> Prediction:
        """Gate EEG prediction."""
        if not signal_quality_ok:
            return Prediction.no_decision("EEG signal quality not OK")

        if (
            self.config.eeg_min_confidence is not None
            and prediction.confidence < self.config.eeg_min_confidence
        ):
            return Prediction.no_decision(
                f"EEG confidence {prediction.confidence:.3f} < "
                f"threshold {self.config.eeg_min_confidence}"
            )

        return prediction
