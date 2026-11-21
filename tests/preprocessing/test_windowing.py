"""
Tests for Windower (PRD §49 Preprocessing).
"""
import warnings
import pytest
import numpy as np

from backend.windowing.windower import (
    Windower, WindowingConfig, LabelPolicy, Window
)


def make_config(window_length=100, stride=50, policy=LabelPolicy.STRICT):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WindowingConfig(
            window_length_samples=window_length,
            stride_samples=stride,
            label_policy=policy,
            windowing_version="test-v1",
        )


class TestWindower:

    def test_correct_number_of_windows(self):
        cfg = make_config(window_length=100, stride=50)
        windower = Windower(cfg)
        data = np.random.randn(500, 4)
        windows = windower.extract(data)
        # Expected: (500 - 100) / 50 + 1 = 9
        assert len(windows) == 9

    def test_window_shape(self):
        cfg = make_config(window_length=100, stride=50)
        windower = Windower(cfg)
        data = np.random.randn(300, 4)
        windows = windower.extract(data)
        for w in windows:
            assert w.data.shape == (100, 4)

    def test_output_does_not_modify_input(self):
        cfg = make_config(window_length=50, stride=25)
        windower = Windower(cfg)
        data = np.ones((200, 4))
        original = data.copy()
        windower.extract(data)
        np.testing.assert_array_equal(data, original)

    def test_extract_trial_all_same_label(self):
        cfg = make_config(window_length=100, stride=50)
        windower = Windower(cfg)
        data = np.random.randn(300, 4)
        windows = windower.extract_trial(data, label="YES")
        assert all(w.label == "YES" for w in windows)
        assert all(not w.discarded for w in windows)

    def test_data_shorter_than_window_returns_empty(self):
        cfg = make_config(window_length=100, stride=50)
        windower = Windower(cfg)
        data = np.random.randn(50, 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            windows = windower.extract(data)
        assert windows == []

    def test_tbd_config_raises_on_extract(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = WindowingConfig(
                window_length_samples=0,
                stride_samples=0,
                windowing_version="TBD",
            )
            windower = Windower(cfg)
        data = np.random.randn(200, 4)
        with pytest.raises(RuntimeError, match="TBD"):
            windower.extract(data)

    def test_strict_policy_discards_boundary_windows(self):
        cfg = make_config(window_length=100, stride=50, policy=LabelPolicy.STRICT)
        windower = Windower(cfg)
        data = np.random.randn(300, 4)
        # Labels: first 150 = "YES", rest = "NO" — boundary at 150
        labels_by_sample = {i: "YES" for i in range(150)}
        labels_by_sample.update({i: "NO" for i in range(150, 300)})
        windows = windower.extract(data, labels_by_sample=labels_by_sample)
        discarded = [w for w in windows if w.discarded]
        # The window 100-200 crosses boundary — should be discarded
        assert len(discarded) > 0

    def test_windows_to_arrays_shape(self):
        cfg = make_config(window_length=100, stride=50)
        windower = Windower(cfg)
        data = np.random.randn(300, 4)
        windows = windower.extract_trial(data, label="YES")
        X, y = windower.windows_to_arrays(windows)
        assert X.ndim == 3
        assert X.shape[1] == 100
        assert X.shape[2] == 4
        assert len(y) == len(windows)

    def test_windows_to_arrays_finite(self):
        cfg = make_config(window_length=50, stride=25)
        windower = Windower(cfg)
        data = np.random.randn(200, 4)
        windows = windower.extract_trial(data, label="NO")
        X, y = windower.windows_to_arrays(windows)
        assert np.all(np.isfinite(X))
