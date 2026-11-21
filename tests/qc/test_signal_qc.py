"""
Tests for SignalQC (PRD §49 QC tests).
SignalQC now emits UserWarning on construction (audit fix DL-022).
All tests suppress or explicitly test those warnings.
"""
import warnings
import pytest
import numpy as np

from backend.acquisition.frame_parser import Sample
from backend.qc.signal_qc import SignalQC


def make_samples(
    n: int = 100,
    start_id: int = 0,
    ch_value: int = 0,
    gap_at: int = None,
) -> list:
    samples = []
    for i in range(n):
        sample_id = start_id + i if gap_at is None or i < gap_at else start_id + i + 5
        samples.append(Sample(
            sample_id=sample_id,
            timestamp_us=i * 250,  # 4 kSPS target interval
            EEG_Fp1=ch_value,
            EEG_Fp2=ch_value,
            EMG_Submental=ch_value,
            EMG_Orbicularis=ch_value,
            EMG_Zygomaticus=ch_value,
            EMG_Buccinator=ch_value,
        ))
    return samples


class TestSignalQC:

    def test_clean_signal_passes(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            qc = SignalQC()
        samples = []
        for i in range(200):
            samples.append(Sample(
                sample_id=i,
                timestamp_us=i * 250,
                EEG_Fp1=int(100 * np.sin(2 * np.pi * i / 40)),
                EEG_Fp2=int(100 * np.cos(2 * np.pi * i / 40)),
                EMG_Submental=int(50 * np.sin(2 * np.pi * i / 20)),
                EMG_Orbicularis=int(50 * np.cos(2 * np.pi * i / 20)),
                EMG_Zygomaticus=int(75 * np.sin(2 * np.pi * i / 30)),
                EMG_Buccinator=int(75 * np.cos(2 * np.pi * i / 30)),
            ))
        result = qc.run(samples, session_id="test")
        assert result.total_samples == 200
        assert result.dropped_samples == 0

    def test_dropped_samples_detected(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            qc = SignalQC()
        samples = make_samples(n=10)
        samples[5] = Sample(
            sample_id=samples[4].sample_id + 10,  # gap of 9
            timestamp_us=samples[4].timestamp_us + 2500,
            EEG_Fp1=0, EEG_Fp2=0,
            EMG_Submental=0, EMG_Orbicularis=0,
            EMG_Zygomaticus=0, EMG_Buccinator=0,
        )
        result = qc.run(samples)
        assert result.dropped_samples > 0

    def test_flatline_detected(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            qc = SignalQC(flatline_variance_threshold=10.0, flatline_window_size=50)
        samples = make_samples(n=200, ch_value=0)  # all zeros = flatline
        result = qc.run(samples)
        flatline_total = sum(
            cr.n_flatline_windows for cr in result.per_channel.values()
        )
        assert flatline_total > 0

    def test_saturation_detected(self):
        adc_max = 2**23 - 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            qc = SignalQC(adc_max=adc_max, saturation_consecutive=3)
        saturated_value = adc_max
        samples = []
        for i in range(50):
            v = saturated_value if 10 <= i < 20 else 100
            samples.append(Sample(
                sample_id=i, timestamp_us=i * 250,
                EEG_Fp1=v, EEG_Fp2=100,
                EMG_Submental=100, EMG_Orbicularis=100,
                EMG_Zygomaticus=100, EMG_Buccinator=100,
            ))
        result = qc.run(samples)
        assert result.per_channel["EEG_Fp1"].n_saturation_runs > 0
        assert not result.passed

    def test_empty_samples(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            qc = SignalQC()
        result = qc.run([])
        assert result.total_samples == 0
        assert not result.passed

    def test_nan_detected(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            qc = SignalQC()
        samples = make_samples(n=10)
        result = qc.run(samples)
        for ch_result in result.per_channel.values():
            assert ch_result.n_nan == 0

    def test_saturation_skipped_when_adc_max_none(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            qc = SignalQC(adc_max=None)
        samples = make_samples(n=50)
        result = qc.run(samples)
        for ch in result.per_channel.values():
            assert "Saturation threshold TBD" in " ".join(ch.notes)

    def test_construction_warns_adc_max_tbd(self):
        """
        SignalQC() with adc_max=None must warn that saturation detection
        is disabled (audit finding H-6, fix DL-022).
        """
        with pytest.warns(UserWarning, match="adc_max"):
            SignalQC(adc_max=None)

    def test_construction_warns_unvalidated_thresholds(self):
        """
        SignalQC() must warn that thresholds are unvalidated placeholders.
        """
        with pytest.warns(UserWarning):
            SignalQC()  # triggers both warnings
