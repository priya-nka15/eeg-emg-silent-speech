"""
URF/A-SSR Evaluation Metrics
PRD §30, §31 — EMG and EEG metrics.
Primary metric: Macro-F1.
HELP recall is safety-critical and must be explicitly reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)


@dataclass
class EvaluationResult:
    """Full metrics report for a model evaluation."""
    modality: str                # "EMG" | "EEG"
    split: str                   # "validation" | "test"
    labels: List[str]            # class labels in order

    macro_f1: float
    accuracy: float
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    confusion_matrix: List[List[int]]

    # Safety-critical for EMG
    help_recall: Optional[float] = None
    help_false_negatives: Optional[int] = None
    false_help_activations: Optional[int] = None

    # EEG-specific
    rejection_rate: Optional[float] = None
    subject_generalization: Optional[dict] = None
    failure_cases: Optional[list] = None

    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "modality": self.modality,
            "split": self.split,
            "labels": self.labels,
            "macro_f1": self.macro_f1,
            "accuracy": self.accuracy,
            "per_class_precision": self.per_class_precision,
            "per_class_recall": self.per_class_recall,
            "per_class_f1": self.per_class_f1,
            "confusion_matrix": self.confusion_matrix,
            "help_recall": self.help_recall,
            "help_false_negatives": self.help_false_negatives,
            "false_help_activations": self.false_help_activations,
            "rejection_rate": self.rejection_rate,
            "subject_generalization": self.subject_generalization,
            "failure_cases": self.failure_cases,
            "notes": self.notes,
        }

    def summary(self) -> str:
        lines = [
            f"=== {self.modality} Evaluation ({self.split}) ===",
            f"Macro-F1:  {self.macro_f1:.4f}",
            f"Accuracy:  {self.accuracy:.4f}",
        ]
        for label in self.labels:
            lines.append(
                f"  {label}: P={self.per_class_precision.get(label, 0):.3f} "
                f"R={self.per_class_recall.get(label, 0):.3f} "
                f"F1={self.per_class_f1.get(label, 0):.3f}"
            )
        if self.help_recall is not None:
            lines.append(f"HELP Recall (safety): {self.help_recall:.4f}")
        if self.false_help_activations is not None:
            lines.append(f"False HELP activations: {self.false_help_activations}")
        return "\n".join(lines)


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    modality: str = "EMG",
    split: str = "validation",
    rejection_mask: Optional[np.ndarray] = None,
) -> EvaluationResult:
    """
    Compute full evaluation metrics.

    Args:
        y_true: True label strings.
        y_pred: Predicted label strings.
        labels: Ordered class labels.
        modality: "EMG" or "EEG".
        split: "validation" or "test".
        rejection_mask: Boolean array; True = rejected (NO_DECISION) for this sample.

    Returns:
        EvaluationResult
    """
    notes = []

    # Rejection rate
    rejection_rate = None
    if rejection_mask is not None:
        rejection_rate = float(np.mean(rejection_mask))
        # Only evaluate on accepted predictions
        accepted = ~rejection_mask
        y_true = y_true[accepted]
        y_pred = y_pred[accepted]
        notes.append(f"Rejection rate: {rejection_rate:.3f} ({np.sum(rejection_mask)} samples)")

    if len(y_true) == 0:
        notes.append("No accepted predictions — all rejected.")
        empty = {l: 0.0 for l in labels}
        return EvaluationResult(
            modality=modality,
            split=split,
            labels=labels,
            macro_f1=0.0,
            accuracy=0.0,
            per_class_precision=empty,
            per_class_recall=empty,
            per_class_f1=empty,
            confusion_matrix=[[0] * len(labels)] * len(labels),
            rejection_rate=rejection_rate,
            notes=notes,
        )

    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    per_class_prec = {
        l: float(v) for l, v in zip(
            labels,
            precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
        )
    }
    per_class_rec = {
        l: float(v) for l, v in zip(
            labels,
            recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
        )
    }
    per_class_f1 = {
        l: float(v) for l, v in zip(
            labels,
            f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
        )
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    # Safety-critical: HELP metrics (EMG only)
    help_recall = None
    help_false_negatives = None
    false_help_activations = None

    if modality == "EMG" and "HELP" in labels:
        help_recall = per_class_rec.get("HELP")
        # False negatives for HELP = HELP predicted as something else
        help_idx = labels.index("HELP")
        cm_arr = np.array(cm)
        if cm_arr.shape[0] > help_idx:
            help_row = cm_arr[help_idx]
            help_false_negatives = int(np.sum(help_row) - help_row[help_idx])
            # False activations = other classes predicted as HELP
            false_help_activations = int(np.sum(cm_arr[:, help_idx]) - cm_arr[help_idx, help_idx])

    return EvaluationResult(
        modality=modality,
        split=split,
        labels=labels,
        macro_f1=macro_f1,
        accuracy=accuracy,
        per_class_precision=per_class_prec,
        per_class_recall=per_class_rec,
        per_class_f1=per_class_f1,
        confusion_matrix=cm,
        help_recall=help_recall,
        help_false_negatives=help_false_negatives,
        false_help_activations=false_help_activations,
        rejection_rate=rejection_rate,
        notes=notes,
    )
