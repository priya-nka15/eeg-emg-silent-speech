"""
Tests for EEG preprocessing pipeline (PRD §49 Preprocessing).
"""
import warnings
import pytest
import numpy as np

from backend.preprocessing.eeg.pipeline import (
    EEGPreprocessingConfig, EEGPreprocessingPipeline
)


def make_config(**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EEGPreprocessingConfig(**kwargs)


class TestEEGPipeline:

    def test_output_shape(self):
        cfg = make_config(preprocessing_version="test-v1")
        pipeline = EEGPreprocessingPipeline(cfg)
        raw = np.random.randn(400, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = pipeline.process(raw)
        assert result.data.shape == (400, 2)

    def test_wrong_channel_count_raises(self):
        cfg = make_config(preprocessing_version="test-v1")
        pipeline = EEGPreprocessingPipeline(cfg)
        raw = np.random.randn(400, 4)  # 4 channels instead of 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="2 EEG channels"):
                pipeline.process(raw)

    def test_output_is_finite(self):
        cfg = make_config(preprocessing_version="test-v1")
        pipeline = EEGPreprocessingPipeline(cfg)
        raw = np.random.randn(400, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = pipeline.process(raw)
        assert result.is_finite()

    def test_provenance_recorded(self):
        cfg = make_config(preprocessing_version="eeg-pp-v1")
        pipeline = EEGPreprocessingPipeline(cfg)
        raw = np.random.randn(200, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = pipeline.process(raw, source_id="EEG_test")
        assert result.provenance["source_id"] == "EEG_test"
        assert result.preprocessing_version == "eeg-pp-v1"

    def test_nan_input_raises(self):
        cfg = make_config(preprocessing_version="test-v1")
        pipeline = EEGPreprocessingPipeline(cfg)
        raw = np.random.randn(100, 2)
        raw[5, 0] = np.inf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="NaN or Inf"):
                pipeline.process(raw)

    def test_deterministic(self):
        cfg = make_config(preprocessing_version="test-v1", normalization="zscore")
        pipeline = EEGPreprocessingPipeline(cfg)
        raw = np.random.randn(200, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r1 = pipeline.process(raw.copy())
            r2 = pipeline.process(raw.copy())
        np.testing.assert_array_equal(r1.data, r2.data)

    def test_channel_names(self):
        cfg = make_config(preprocessing_version="test-v1")
        pipeline = EEGPreprocessingPipeline(cfg)
        raw = np.random.randn(100, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = pipeline.process(raw)
        assert result.channel_names == ["EEG_Fp1", "EEG_Fp2"]
