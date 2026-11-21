"""
Tests for EMG preprocessing pipeline (PRD §49 Preprocessing).
"""
import warnings
import pytest
import numpy as np

from backend.preprocessing.emg.pipeline import (
    EMGPreprocessingConfig, EMGPreprocessingPipeline
)


def make_config(**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EMGPreprocessingConfig(**kwargs)


class TestEMGPipeline:

    def test_output_same_shape(self):
        cfg = make_config(preprocessing_version="test-v1")
        pipeline = EMGPreprocessingPipeline(cfg)
        raw = np.random.randn(400, 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = pipeline.process(raw)
        assert result.data.shape == raw.shape

    def test_output_is_finite(self):
        cfg = make_config(preprocessing_version="test-v1")
        pipeline = EMGPreprocessingPipeline(cfg)
        raw = np.random.randn(400, 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = pipeline.process(raw)
        assert result.is_finite()

    def test_provenance_recorded(self):
        cfg = make_config(preprocessing_version="test-v1")
        pipeline = EMGPreprocessingPipeline(cfg)
        raw = np.random.randn(200, 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = pipeline.process(raw, source_id="S001_Sess01")
        assert result.provenance["source_id"] == "S001_Sess01"
        assert len(result.provenance["steps_applied"]) > 0

    def test_nan_input_raises(self):
        cfg = make_config(preprocessing_version="test-v1")
        pipeline = EMGPreprocessingPipeline(cfg)
        raw = np.random.randn(100, 4)
        raw[10, 2] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="NaN or Inf"):
                pipeline.process(raw)

    def test_zscore_normalization_applied(self):
        cfg = make_config(preprocessing_version="test-v1", normalization="zscore")
        pipeline = EMGPreprocessingPipeline(cfg)
        raw = np.random.randn(400, 4) * 1000 + 500
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = pipeline.process(raw)
        # After z-score, mean should be ~0 and std ~1
        assert abs(result.data.mean()) < 0.5
        assert abs(result.data.std() - 1.0) < 0.2

    def test_deterministic_output(self):
        cfg = make_config(preprocessing_version="test-v1", normalization="zscore")
        pipeline = EMGPreprocessingPipeline(cfg)
        raw = np.random.randn(200, 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r1 = pipeline.process(raw.copy())
            r2 = pipeline.process(raw.copy())
        np.testing.assert_array_equal(r1.data, r2.data)

    def test_version_recorded(self):
        cfg = make_config(preprocessing_version="emg-pp-v1")
        pipeline = EMGPreprocessingPipeline(cfg)
        raw = np.random.randn(100, 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = pipeline.process(raw)
        assert result.preprocessing_version == "emg-pp-v1"

    def test_bandpass_applied_when_enabled(self):
        cfg = make_config(
            preprocessing_version="test-v1",
            sw_bandpass_enabled=True,
            sw_bandpass_low_hz=20.0,
            sw_bandpass_high_hz=500.0,
            sampling_rate_hz=4000.0,
        )
        pipeline = EMGPreprocessingPipeline(cfg)
        raw = np.random.randn(1000, 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = pipeline.process(raw)
        assert result.is_finite()
        assert any("bandpass" in s for s in result.provenance["steps_applied"])
