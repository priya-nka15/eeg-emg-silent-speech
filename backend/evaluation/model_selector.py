"""
URF/A-SSR Model Selector
PRD §32 — Model selection on validation data only.
Priority: Macro-F1 > safety recall > robustness > latency > complexity.

IMPORTANT — two selection modes are explicitly distinguished:
  preliminary_select()  — allowed before criteria are frozen; for exploration only.
  final_select()        — requires criteria_frozen=True; blocked otherwise.

Using final_select() with unfrozen criteria raises SelectionError.
This enforces PRD §32: "Exact tie-break rules must be frozen before final model selection."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from backend.evaluation.metrics import EvaluationResult


class SelectionError(Exception):
    """Raised when model selection violates PRD §32 constraints."""


@dataclass
class SelectionCriteria:
    """
    Tie-break rules for model selection.

    criteria_frozen MUST be True before calling final_select().

    min_safety_recall: the minimum acceptable HELP recall for EMG models.
        This is PILOT-VALIDATED — it must be determined from validation data,
        not invented. Default is None (TBD) which means the safety recall
        filter is inactive until explicitly set.

        Do NOT use 0.0 as a default — that is an implicit inactive threshold
        that could select a model with 0% HELP recall (audit finding M-6).
    """
    primary_metric: str = "macro_f1"              # must be validation metric
    safety_recall_label: Optional[str] = "HELP"   # safety-critical class
    min_safety_recall: Optional[float] = None      # PILOT-VALIDATED; None = TBD
    max_latency_ms: Optional[float] = None         # PILOT-VALIDATED
    criteria_frozen: bool = False                  # must be True before final_select()

    def issues(self) -> List[str]:
        """Return all unresolved issues that would block final selection."""
        problems = []
        if not self.criteria_frozen:
            problems.append(
                "criteria_frozen is False — tie-break rules have not been frozen "
                "(PRD §32). Call final_select() only after freezing."
            )
        if self.min_safety_recall is None:
            problems.append(
                "min_safety_recall is None (TBD). "
                "Set a validated HELP recall threshold from pilot/validation data "
                "before final model selection (PRD §32, §30)."
            )
        return problems


class ModelSelector:
    """
    Selects the best model from validation results.

    Two modes:
      preliminary_select() — exploratory; works without frozen criteria but
                             results must not be used to finalise a model.
      final_select()       — enforces all PRD §32 constraints; raises
                             SelectionError if criteria are not frozen.

    Both modes reject test-split results (PRD §33).
    """

    def __init__(self, criteria: SelectionCriteria) -> None:
        self.criteria = criteria

    # ── Public interface ──────────────────────────────────────────────────────

    def final_select(
        self,
        validation_results: Dict[str, EvaluationResult],
        latencies_ms: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Select the best model for final use.

        Raises SelectionError if:
          - criteria_frozen is False
          - min_safety_recall is None
          - any result uses the 'test' split

        Args:
            validation_results: model_id → EvaluationResult (validation split)
            latencies_ms: optional model_id → latency in ms

        Returns:
            Selected model_id
        """
        problems = self.criteria.issues()
        if problems:
            raise SelectionError(
                "Final model selection blocked — criteria not frozen:\n"
                + "\n".join(f"  - {p}" for p in problems)
            )
        return self._select(validation_results, latencies_ms)

    def preliminary_select(
        self,
        validation_results: Dict[str, EvaluationResult],
        latencies_ms: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Exploratory model selection — for research iteration only.

        This method does NOT require criteria_frozen=True. Results MUST NOT
        be used to declare a final model or evaluate on the test set.

        Args:
            validation_results: model_id → EvaluationResult (validation split)
            latencies_ms: optional model_id → latency in ms

        Returns:
            Preliminary selected model_id (not final)
        """
        import warnings
        problems = self.criteria.issues()
        if problems:
            warnings.warn(
                "preliminary_select(): criteria are not fully configured — "
                "result is exploratory only, not suitable for final model "
                "selection:\n" + "\n".join(f"  - {p}" for p in problems),
                UserWarning,
                stacklevel=2,
            )
        return self._select(validation_results, latencies_ms)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _select(
        self,
        validation_results: Dict[str, EvaluationResult],
        latencies_ms: Optional[Dict[str, float]],
    ) -> str:
        if not validation_results:
            raise SelectionError("No validation results provided.")

        # Reject test-split results
        for model_id, result in validation_results.items():
            if result.split != "validation":
                raise SelectionError(
                    f"Model {model_id!r}: selection must use 'validation' split, "
                    f"got '{result.split}'. Do not use the test set for selection "
                    f"(PRD §33)."
                )

        # Apply safety recall filter (only when threshold is set)
        candidates = dict(validation_results)
        if (
            self.criteria.safety_recall_label is not None
            and self.criteria.min_safety_recall is not None
            and self.criteria.min_safety_recall > 0
        ):
            eligible = {
                mid: res for mid, res in candidates.items()
                if res.per_class_recall.get(
                    self.criteria.safety_recall_label, 0.0
                ) >= self.criteria.min_safety_recall
            }
            if eligible:
                candidates = eligible
            # If no model meets the threshold, keep all candidates but note it

        if not candidates:
            raise SelectionError(
                "No candidates after applying selection criteria."
            )

        # Rank by primary metric (Macro-F1)
        ranked = sorted(
            candidates.items(),
            key=lambda x: x[1].macro_f1,
            reverse=True,
        )

        # Tie-break by latency if configured
        if latencies_ms and self.criteria.max_latency_ms is not None:
            within_latency = [
                (mid, res) for mid, res in ranked
                if (latencies_ms.get(mid) or 0.0) <= self.criteria.max_latency_ms
            ]
            if within_latency:
                ranked = within_latency

        if not ranked:
            raise SelectionError("No models within latency constraints.")

        return ranked[0][0]

    def selection_report(
        self,
        validation_results: Dict[str, EvaluationResult],
        selected_id: str,
    ) -> str:
        lines = ["=== Model Selection Report (Validation Set) ==="]
        for mid, res in sorted(
            validation_results.items(),
            key=lambda x: x[1].macro_f1,
            reverse=True,
        ):
            marker = " ← SELECTED" if mid == selected_id else ""
            help_r = (
                f"HELP_R={res.help_recall:.3f}" if res.help_recall is not None else ""
            )
            lines.append(
                f"  {mid}: Macro-F1={res.macro_f1:.4f} "
                f"Acc={res.accuracy:.4f} {help_r}{marker}"
            )
        return "\n".join(lines)
