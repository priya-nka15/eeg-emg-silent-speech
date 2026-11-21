"""
Tests for evaluation metrics (PRD §49 Models, §30, §31).
"""
import pytest
import numpy as np

from backend.evaluation.metrics import evaluate


class TestEvaluateMetrics:

    def test_macro_f1_perfect(self):
        y_true = np.array(["YES", "NO", "HELP", "YES", "NO", "HELP"])
        y_pred = np.array(["YES", "NO", "HELP", "YES", "NO", "HELP"])
        result = evaluate(y_true, y_pred, labels=["YES", "NO", "HELP"])
        assert abs(result.macro_f1 - 1.0) < 1e-6

    def test_accuracy_perfect(self):
        y_true = np.array(["YES", "NO", "YES"])
        y_pred = np.array(["YES", "NO", "YES"])
        result = evaluate(y_true, y_pred, labels=["YES", "NO", "HELP"])
        assert result.accuracy == 1.0

    def test_help_recall_reported_for_emg(self):
        y_true = np.array(["HELP", "HELP", "YES", "NO"])
        y_pred = np.array(["HELP", "YES", "YES", "NO"])
        result = evaluate(y_true, y_pred, labels=["YES", "NO", "HELP"], modality="EMG")
        assert result.help_recall is not None
        assert abs(result.help_recall - 0.5) < 1e-6

    def test_help_false_negatives_counted(self):
        y_true = np.array(["HELP", "HELP", "HELP"])
        y_pred = np.array(["YES", "NO", "HELP"])
        result = evaluate(y_true, y_pred, labels=["YES", "NO", "HELP"], modality="EMG")
        assert result.help_false_negatives == 2

    def test_false_help_activations_counted(self):
        y_true = np.array(["YES", "NO", "HELP"])
        y_pred = np.array(["HELP", "HELP", "HELP"])
        result = evaluate(y_true, y_pred, labels=["YES", "NO", "HELP"], modality="EMG")
        assert result.false_help_activations == 2

    def test_confusion_matrix_shape(self):
        y_true = np.array(["YES", "NO", "HELP"] * 3)
        y_pred = np.array(["YES", "NO", "HELP"] * 3)
        result = evaluate(y_true, y_pred, labels=["YES", "NO", "HELP"])
        assert len(result.confusion_matrix) == 3
        assert all(len(row) == 3 for row in result.confusion_matrix)

    def test_split_validation_label(self):
        y_true = np.array(["YES", "NO"])
        y_pred = np.array(["YES", "NO"])
        result = evaluate(y_true, y_pred, labels=["YES", "NO", "HELP"],
                         split="validation")
        assert result.split == "validation"

    def test_rejection_rate_computed(self):
        y_true = np.array(["YES", "NO", "YES", "HELP"])
        y_pred = np.array(["YES", "NO", "YES", "HELP"])
        rejection = np.array([True, False, False, False])
        result = evaluate(y_true, y_pred, labels=["YES", "NO", "HELP"],
                         rejection_mask=rejection)
        assert result.rejection_rate == pytest.approx(0.25)
        # Only 3 non-rejected samples evaluated
        assert result.accuracy == 1.0

    def test_per_class_metrics_in_result(self):
        y_true = np.array(["YES", "NO", "HELP", "YES"])
        y_pred = np.array(["YES", "NO", "HELP", "YES"])
        result = evaluate(y_true, y_pred, labels=["YES", "NO", "HELP"])
        assert "YES" in result.per_class_precision
        assert "NO" in result.per_class_recall
        assert "HELP" in result.per_class_f1

    def test_all_rejected_returns_zero_metrics(self):
        y_true = np.array(["YES", "NO"])
        y_pred = np.array(["YES", "NO"])
        rejection = np.array([True, True])
        result = evaluate(y_true, y_pred, labels=["YES", "NO", "HELP"],
                         rejection_mask=rejection)
        assert result.macro_f1 == 0.0
        assert "No accepted predictions" in " ".join(result.notes)
