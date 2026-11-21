"""
Tests for EMG models — SVM, GRU, CNN-GRU (PRD §49 Models).
Tests input/output shapes, probability normalization, serialization.
"""
import warnings
import tempfile
import pytest
import numpy as np
from pathlib import Path

from backend.models.emg.svm_model import EMGSVMModel
from backend.models.base import Prediction


LABEL_MAP = {"YES": 0, "NO": 1, "HELP": 2}
N_FEATURES = 24
N_SAMPLES = 30
N_SEQ = 100
N_CHANNELS = 4


def make_feature_data(n=N_SAMPLES):
    X = np.random.randn(n, N_FEATURES).astype(np.float32)
    y = np.array(["YES", "NO", "HELP"] * (n // 3) + ["YES"] * (n % 3))
    return X, y


def make_sequence_data(n=N_SAMPLES, seq_len=N_SEQ, n_ch=N_CHANNELS):
    X = np.random.randn(n, seq_len, n_ch).astype(np.float32)
    y = np.array(["YES", "NO", "HELP"] * (n // 3) + ["YES"] * (n % 3))
    return X, y


class TestEMGSVM:

    def test_fit_and_predict_shape(self):
        model = EMGSVMModel(label_map=LABEL_MAP)
        X, y = make_feature_data(30)
        model.fit(X, y)
        preds = model.predict(X[:5])
        assert len(preds) == 5

    def test_predict_proba_shape(self):
        model = EMGSVMModel(label_map=LABEL_MAP)
        X, y = make_feature_data(30)
        model.fit(X, y)
        proba = model.predict_proba(X[:5])
        assert proba.shape == (5, 3)

    def test_probabilities_sum_to_one(self):
        model = EMGSVMModel(label_map=LABEL_MAP)
        X, y = make_feature_data(30)
        model.fit(X, y)
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_prediction_labels_valid(self):
        model = EMGSVMModel(label_map=LABEL_MAP)
        X, y = make_feature_data(30)
        model.fit(X, y)
        preds = model.predict(X)
        for p in preds:
            assert p.label in LABEL_MAP

    def test_untrained_raises(self):
        model = EMGSVMModel(label_map=LABEL_MAP)
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict_proba(np.random.randn(5, N_FEATURES))

    def test_wrong_input_dim_raises(self):
        model = EMGSVMModel(label_map=LABEL_MAP)
        X, y = make_feature_data(30)
        model.fit(X, y)
        with pytest.raises(ValueError, match="2D"):
            model.predict_proba(np.random.randn(5, N_SEQ, N_CHANNELS))

    def test_save_and_load(self):
        model = EMGSVMModel(label_map=LABEL_MAP)
        X, y = make_feature_data(30)
        model.fit(X, y)
        proba_before = model.predict_proba(X[:3])

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(Path(tmpdir))
            loaded = EMGSVMModel(label_map=LABEL_MAP)
            loaded.load(Path(tmpdir))
            proba_after = loaded.predict_proba(X[:3])

        np.testing.assert_allclose(proba_before, proba_after, atol=1e-5)

    def test_confidence_attribute(self):
        model = EMGSVMModel(label_map=LABEL_MAP)
        X, y = make_feature_data(30)
        model.fit(X, y)
        preds = model.predict(X[:5])
        for p in preds:
            assert 0.0 <= p.confidence <= 1.0


class TestEMGGRU:

    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        pytest.importorskip("torch", reason="PyTorch not installed — skipping GRU tests")

    def test_fit_and_predict(self):
        from backend.models.emg.gru_model import EMGGRUModel
        model = EMGGRUModel(
            label_map=LABEL_MAP,
            n_epochs=2,
            hidden_size=16,
            num_layers=1,
        )
        X, y = make_sequence_data(30, seq_len=50, n_ch=4)
        model.fit(X, y)
        proba = model.predict_proba(X[:5])
        assert proba.shape == (5, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)

    def test_gru_requires_3d_input(self):
        from backend.models.emg.gru_model import EMGGRUModel
        model = EMGGRUModel(label_map=LABEL_MAP, n_epochs=1)
        X, y = make_feature_data(30)
        with pytest.raises(ValueError, match="3D"):
            model.fit(X, y)


class TestEMGCNNGRU:

    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        pytest.importorskip("torch", reason="PyTorch not installed — skipping CNN-GRU tests")

    def test_fit_and_predict(self):
        from backend.models.emg.cnn_gru_model import EMGCNNGRUModel
        model = EMGCNNGRUModel(
            label_map=LABEL_MAP,
            n_epochs=2,
            cnn_filters=8,
            gru_hidden_size=16,
            gru_num_layers=1,
        )
        X, y = make_sequence_data(30, seq_len=50, n_ch=4)
        model.fit(X, y)
        proba = model.predict_proba(X[:5])
        assert proba.shape == (5, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)
