"""
URF/A-SSR Temporal Consistency Layer
PRD §36 — Do not trigger from one noisy window.
Configurable strategy: consecutive predictions, majority vote, smoothing, hysteresis.
Window count must come from validation data, not be invented.
"""
from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, List, Optional

from backend.models.base import Prediction
from backend.protocol.output_states import IntentOutput


class ConsistencyStrategy(str, Enum):
    CONSECUTIVE = "consecutive"     # N consecutive matching predictions
    MAJORITY    = "majority_vote"   # majority in last N windows
    SMOOTHING   = "smoothing"       # probability smoothing
    HYSTERESIS  = "hysteresis"      # change only if different for N windows


@dataclass
class TemporalConsistencyConfig:
    """
    All window/threshold parameters are TBD until validated (PRD §36).
    """
    strategy: ConsistencyStrategy = ConsistencyStrategy.CONSECUTIVE
    required_windows: Optional[int] = None  # PILOT-VALIDATED
    smoothing_alpha: float = 0.5            # PILOT-VALIDATED
    config_version: str = "TBD"

    def validate(self) -> list:
        issues = []
        if self.required_windows is None:
            issues.append("required_windows is TBD — no temporal consistency active")
        if self.config_version == "TBD":
            issues.append("config_version is TBD")
        return issues

    def is_configured(self) -> bool:
        return self.required_windows is not None


class TemporalConsistencyFilter:
    """
    Buffers window predictions and outputs a stable prediction
    only when temporal consistency criteria are met.

    PRD §36: Configurable strategy for preventing single-window triggering.
    """

    def __init__(self, config: TemporalConsistencyConfig) -> None:
        self.config = config
        issues = config.validate()
        if issues:
            warnings.warn(
                f"TemporalConsistencyConfig has TBD fields: {issues}. "
                "No temporal consistency active until configured.",
                UserWarning,
                stacklevel=2,
            )
        self._buffer: Deque[Prediction] = deque()
        self._smooth_probs: dict = {}

    def update(self, prediction: Prediction) -> Prediction:
        """
        Add a new prediction to the buffer.
        Returns a temporally-consistent prediction or NO_DECISION.
        """
        if prediction.label == IntentOutput.NO_DECISION:
            self._buffer.clear()
            self._smooth_probs = {}
            return prediction

        if not self.config.is_configured():
            # No consistency enforced — pass through
            return prediction

        n = self.config.required_windows
        self._buffer.append(prediction)
        if len(self._buffer) > n:
            self._buffer.popleft()

        if len(self._buffer) < n:
            return Prediction.no_decision(
                f"Accumulating window {len(self._buffer)}/{n}"
            )

        return self._apply_strategy()

    def reset(self) -> None:
        self._buffer.clear()
        self._smooth_probs = {}

    def _apply_strategy(self) -> Prediction:
        strategy = self.config.strategy

        if strategy == ConsistencyStrategy.CONSECUTIVE:
            return self._consecutive()

        if strategy == ConsistencyStrategy.MAJORITY:
            return self._majority()

        if strategy == ConsistencyStrategy.HYSTERESIS:
            return self._hysteresis()

        # Default: majority vote
        return self._majority()

    def _consecutive(self) -> Prediction:
        """All buffered predictions must be the same label."""
        labels = [p.label for p in self._buffer]
        if len(set(labels)) == 1:
            # Return the most recent prediction
            return list(self._buffer)[-1]
        return Prediction.no_decision("Consecutive consistency not met")

    def _majority(self) -> Prediction:
        """Majority of buffered predictions determine output."""
        from collections import Counter
        labels = [p.label for p in self._buffer]
        most_common, count = Counter(labels).most_common(1)[0]
        n = self.config.required_windows
        if count >= (n // 2 + 1):
            # Find a prediction with the majority label for probabilities
            for p in reversed(list(self._buffer)):
                if p.label == most_common:
                    return p
        return Prediction.no_decision("Majority consistency not met")

    def _hysteresis(self) -> Prediction:
        """Change output only if all buffered predictions agree."""
        return self._consecutive()
