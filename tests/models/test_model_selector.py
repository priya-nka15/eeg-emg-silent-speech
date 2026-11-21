"""
Tests for ModelSelector — final vs preliminary selection, validation-only
enforcement, HELP recall filter, unfrozen criteria block final selection.
PRD §32, §49.
"""
import numpy as np
import pytest

from backend.evaluation.metrics import evaluate
from backend.evaluation.model_selector import (
    ModelSelector, SelectionCriteria, SelectionError
)


def make_result(macro_f1, help_recall, split="validation"):
    labels = ["YES", "NO", "HELP"]
    n = 30
    y_true = np.array(labels * (n // 3))
    # Build y_pred that achieves approximately the desired macro_f1
    y_pred = y_true.copy()
    # Scramble some HELP predictions to control HELP recall
    help_positions = np.where(y_true == "HELP")[0]
    n_wrong = int(len(help_positions) * (1.0 - help_recall))
    y_pred[help_positions[:n_wrong]] = "YES"
    return evaluate(y_true, y_pred, labels=labels, split=split)


class TestFinalSelect:
    def test_blocks_when_not_frozen(self):
        criteria = SelectionCriteria(criteria_frozen=False, min_safety_recall=0.7)
        sel = ModelSelector(criteria)
        results = {"m1": make_result(0.9, 0.9)}
        with pytest.raises(SelectionError, match="criteria not frozen"):
            sel.final_select(results)

    def test_blocks_when_safety_recall_tbd(self):
        criteria = SelectionCriteria(criteria_frozen=True, min_safety_recall=None)
        sel = ModelSelector(criteria)
        results = {"m1": make_result(0.9, 0.9)}
        with pytest.raises(SelectionError, match="min_safety_recall is None"):
            sel.final_select(results)

    def test_rejects_test_split(self):
        criteria = SelectionCriteria(criteria_frozen=True, min_safety_recall=0.7)
        sel = ModelSelector(criteria)
        results = {"m1": make_result(0.9, 0.9, split="test")}
        with pytest.raises(SelectionError, match="'test'"):
            sel.final_select(results)

    def test_selects_best_macro_f1(self):
        criteria = SelectionCriteria(criteria_frozen=True, min_safety_recall=0.5)
        sel = ModelSelector(criteria)
        # Build results with explicit known macro_f1 values by using perfect predictions
        labels = ["YES", "NO", "HELP"]
        n = 30
        y_base = np.array(labels * (n // 3))

        # svm: perfect on YES/NO but misses 30% of HELP → lower macro-F1
        y_svm = y_base.copy()
        help_idx = np.where(y_base == "HELP")[0]
        y_svm[help_idx[:3]] = "YES"  # 3 HELP → YES (30% miss)

        # gru: perfect → macro-F1 = 1.0
        y_gru = y_base.copy()

        # cnn_gru: misses 50% HELP → lowest macro-F1
        y_cnn = y_base.copy()
        y_cnn[help_idx[:5]] = "NO"

        results = {
            "svm":     evaluate(y_base, y_svm, labels=labels, split="validation"),
            "gru":     evaluate(y_base, y_gru, labels=labels, split="validation"),
            "cnn_gru": evaluate(y_base, y_cnn, labels=labels, split="validation"),
        }
        # Verify ordering before asserting selection
        assert results["gru"].macro_f1 > results["svm"].macro_f1 > results["cnn_gru"].macro_f1, \
            "Test data does not produce expected F1 ordering"
        selected = sel.final_select(results)
        assert selected == "gru"

    def test_safety_recall_filter_excludes_low_recall(self):
        criteria = SelectionCriteria(criteria_frozen=True, min_safety_recall=0.8)
        sel = ModelSelector(criteria)
        results = {
            "good_recall": make_result(0.70, 0.90),
            "bad_recall":  make_result(0.85, 0.50),  # high F1 but HELP recall too low
        }
        selected = sel.final_select(results)
        # bad_recall excluded by HELP filter, good_recall selected despite lower F1
        assert selected == "good_recall"

    def test_empty_results_raises(self):
        criteria = SelectionCriteria(criteria_frozen=True, min_safety_recall=0.5)
        sel = ModelSelector(criteria)
        with pytest.raises(SelectionError):
            sel.final_select({})


class TestPreliminarySelect:
    def test_proceeds_with_unfrozen_criteria(self):
        """preliminary_select must not block — only warn."""
        criteria = SelectionCriteria(criteria_frozen=False, min_safety_recall=None)
        sel = ModelSelector(criteria)
        results = {"m1": make_result(0.8, 0.8)}
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            selected = sel.preliminary_select(results)
        assert selected == "m1"
        assert any("exploratory" in str(warning.message).lower() or
                   "not fully configured" in str(warning.message).lower()
                   for warning in w)

    def test_still_rejects_test_split(self):
        criteria = SelectionCriteria(criteria_frozen=False)
        sel = ModelSelector(criteria)
        with pytest.raises(SelectionError, match="'test'"):
            sel.preliminary_select({"m1": make_result(0.8, 0.8, split="test")})

    def test_min_safety_recall_zero_was_removed(self):
        """
        Regression: min_safety_recall must default to None, not 0.0.
        A default of 0.0 silently disables the safety filter.
        """
        criteria = SelectionCriteria()
        assert criteria.min_safety_recall is None, (
            "min_safety_recall must default to None (TBD), not 0.0. "
            "0.0 is an implicit inactive threshold that could select a model "
            "with 0% HELP recall."
        )


class TestSelectionReport:
    def test_report_marks_selected(self):
        criteria = SelectionCriteria(criteria_frozen=True, min_safety_recall=0.5)
        sel = ModelSelector(criteria)
        results = {
            "m1": make_result(0.7, 0.8),
            "m2": make_result(0.9, 0.9),
        }
        selected = sel.final_select(results)
        report = sel.selection_report(results, selected)
        assert "SELECTED" in report
        assert "m2" in report
