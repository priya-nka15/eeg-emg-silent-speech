"""
Tests for EEG models — SVM and 1D-CNN (PRD §49 Models).
Covers input/output shapes, probability normalisation, save/load,
untrained-model error, wrong-dimension rejection, and placeholder-label warnings.
"""
import warnings
import tempfile
import pytest
import numpy as np
from pathlib import Path


N_FEATURES = 12
N_SAMPLES = 24
N_SEQ = 80
PLACEHOLDER_LABEL_MAP = {"CONDITION_A": 0, "CONDITION_B": 1}


def make_feature_data(n=N_SAMPLES):
    X = np.random.randn(n, N_FEATURES).astype(np.float32)
    y = np.array(["CONDITION_A", "CONDITION_B"] * (n // 2))
    return X, y


def make_sequence_data(n=N_SAMPLES, seq_len=N_SEQ):
    X = np.random.randn(n, seq_len, 2).astype(np.float32)
    y = np.array(["CONDITION_A", "CONDITION_B"] * (n // 2))
    return X, y


# ── EEG SVM ───────────────────────────────────────────────────────────────────

class TestEEGSVM:

    def test_fit_and_predict_shape(self):
        from backend.models.eeg.svm_model import EEGSVMModel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = EEGSVMModel(label_map=PLACEHOLDER_LABEL_MAP)
        X, y = make_feature_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        preds = model.predict(X[:4])
        assert len(preds) == 4

    def test_predict_proba_shape(self):
        from backend.models.eeg.svm_model import EEGSVMModel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = EEGSVMModel(label_map=PLACEHOLDER_LABEL_MAP)
        X, y = make_feature_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
            proba = model.predict_proba(X[:5])
        assert proba.shape == (5, 2)

    def test_probabilities_sum_to_one(self):
        from backend.models.eeg.svm_model import EEGSVMModel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = EEGSVMModel(label_map=PLACEHOLDER_LABEL_MAP)
            X, y = make_feature_data()
            model.fit(X, y)
            proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_untrained_raises(self):
        from backend.models.eeg.svm_model import EEGSVMModel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = EEGSVMModel(label_map=PLACEHOLDER_LABEL_MAP)
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict_proba(np.random.randn(3, N_FEATURES))

    def test_wrong_input_dim_raises(self):
        from backend.models.eeg.svm_model import EEGSVMModel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = EEGSVMModel(label_map=PLACEHOLDER_LABEL_MAP)
            X, y = make_feature_data()
            model.fit(X, y)
        with pytest.raises((ValueError, Exception)):
            # 3D input must be rejected — exact message varies by sklearn version
            model.predict_proba(np.random.randn(3, 10, 2))

    def test_save_and_load_roundtrip(self):
        from backend.models.eeg.svm_model import EEGSVMModel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = EEGSVMModel(label_map=PLACEHOLDER_LABEL_MAP)
            X, y = make_feature_data()
            model.fit(X, y)
            before = model.predict_proba(X[:3])

        with tempfile.TemporaryDirectory() as tmp:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.save(Path(tmp))
                loaded = EEGSVMModel(label_map=PLACEHOLDER_LABEL_MAP)
                loaded.load(Path(tmp))
                after = loaded.predict_proba(X[:3])

        np.testing.assert_allclose(before, after, atol=1e-5)

    def test_placeholder_label_emits_warning(self):
        """A model with unfrozen placeholder labels must warn (audit finding M-7)."""
        from backend.models.eeg.svm_model import EEGSVMModel
        with pytest.warns(UserWarning, match="placeholder"):
            model = EEGSVMModel()  # no explicit label_map

    def test_explicit_placeholder_map_also_warns(self):
        """Passing the placeholder dict explicitly must also warn."""
        from backend.models.eeg.svm_model import EEGSVMModel
        with pytest.warns(UserWarning, match="placeholder|CONDITION"):
            model = EEGSVMModel(label_map={"CONDITION_A": 0, "CONDITION_B": 1})

    def test_prediction_labels_are_placeholder(self):
        from backend.models.eeg.svm_model import EEGSVMModel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = EEGSVMModel(label_map=PLACEHOLDER_LABEL_MAP)
            X, y = make_feature_data()
            model.fit(X, y)
            preds = model.predict(X)
        for p in preds:
            assert p.label in PLACEHOLDER_LABEL_MAP

    def test_labels_frozen_false_in_saved_config(self):
        from backend.models.eeg.svm_model import EEGSVMModel
        import json
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = EEGSVMModel(label_map=PLACEHOLDER_LABEL_MAP)
            X, y = make_feature_data()
            model.fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.save(Path(tmp))
            with open(Path(tmp) / "config.json") as f:
                cfg = json.load(f)
        assert cfg["labels_frozen"] is False


# ── EEG 1D-CNN ────────────────────────────────────────────────────────────────

class TestEEGCNN1D:

    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        pytest.importorskip("torch", reason="PyTorch not installed — skipping 1D-CNN tests")

    def test_fit_and_predict_shape(self):
        from backend.models.eeg.cnn1d_model import EEGCNN1DModel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = EEGCNN1DModel(
                label_map=PLACEHOLDER_LABEL_MAP,
                cnn_filters=8, cnn_layers=1, n_epochs=2,
            )
        X, y = make_sequence_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
            proba = model.predict_proba(X[:4])
        assert proba.shape == (4, 2)

    def test_probabilities_sum_to_one(self):
        from backend.models.eeg.cnn1d_model import EEGCNN1DModel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = EEGCNN1DModel(
                label_map=PLACEHOLDER_LABEL_MAP,
                cnn_filters=8, cnn_layers=1, n_epochs=2,
            )
            X, y = make_sequence_data()
            model.fit(X, y)
            proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)

    def test_wrong_shape_raises(self):
        from backend.models.eeg.cnn1d_model import EEGCNN1DModel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = EEGCNN1DModel(label_map=PLACEHOLDER_LABEL_MAP, n_epochs=1)
            X, y = make_sequence_data()
            model.fit(X, y)
        with pytest.raises((ValueError, RuntimeError)):
            # 4 input channels instead of 2 — exact error type varies
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.predict_proba(np.random.randn(4, N_SEQ, 4))

    def test_untrained_raises(self):
        from backend.models.eeg.cnn1d_model import EEGCNN1DModel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = EEGCNN1DModel(label_map=PLACEHOLDER_LABEL_MAP)
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict_proba(np.random.randn(3, N_SEQ, 2))

    def test_placeholder_label_emits_warning(self):
        from backend.models.eeg.cnn1d_model import EEGCNN1DModel
        with pytest.warns(UserWarning, match="placeholder|CONDITION|not frozen"):
            model = EEGCNN1DModel()
