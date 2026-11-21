"""
Tests for HardwareValidator gates H1–H8 (PRD §9, §49).
All tests that require physical hardware return NOT_STARTED/BLOCKED — never PASS.
Tests verify the logic of each gate without fabricating hardware results.
"""
import pytest
from backend.acquisition.hardware_validator import HardwareValidator, GateResult
from backend.acquisition.frame_parser import Sample


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_samples(n: int, rate_hz: float = 4000.0, start_id: int = 0) -> list[Sample]:
    """Synthetic sample stream at the given rate."""
    interval_us = int(1_000_000 / rate_hz)
    return [
        Sample(
            sample_id=start_id + i,
            timestamp_us=i * interval_us,
            EEG_Fp1=100, EEG_Fp2=-100,
            EMG_Submental=50, EMG_Orbicularis=-50,
            EMG_Zygomaticus=75, EMG_Buccinator=-75,
        )
        for i in range(n)
    ]


# ── H1: Channel mapping ───────────────────────────────────────────────────────

class TestH1ChannelMapping:
    def test_none_probe_returns_not_started(self):
        hv = HardwareValidator()
        r = hv.validate_channel_mapping(probe_results=None)
        assert r.result == GateResult.NOT_STARTED

    def test_correct_mapping_passes(self):
        hv = HardwareValidator()
        r = hv.validate_channel_mapping({
            "EEG_Fp1": "Fp1",
            "EEG_Fp2": "Fp2",
            "EMG_Submental": "Submental/anterior-digastric",
            "EMG_Orbicularis": "Orbicularis oris",
            "EMG_Zygomaticus": "Zygomaticus major",
            "EMG_Buccinator": "Buccinator",
        })
        assert r.result == GateResult.PASS

    def test_wrong_channel_fails(self):
        hv = HardwareValidator()
        r = hv.validate_channel_mapping({
            "EEG_Fp1": "Fp2",          # swapped
            "EEG_Fp2": "Fp1",
            "EMG_Submental": "Submental/anterior-digastric",
            "EMG_Orbicularis": "Orbicularis oris",
            "EMG_Zygomaticus": "Zygomaticus major",
            "EMG_Buccinator": "Buccinator",
        })
        assert r.result == GateResult.FAIL
        assert "EEG_Fp1" in r.details["failures"]

    def test_missing_channel_fails(self):
        hv = HardwareValidator()
        r = hv.validate_channel_mapping({
            "EEG_Fp1": "Fp1",
            # EEG_Fp2 missing
            "EMG_Submental": "Submental/anterior-digastric",
            "EMG_Orbicularis": "Orbicularis oris",
            "EMG_Zygomaticus": "Zygomaticus major",
            "EMG_Buccinator": "Buccinator",
        })
        assert r.result == GateResult.FAIL


# ── H2: Sampling rate ─────────────────────────────────────────────────────────

class TestH2SamplingRate:
    def test_no_samples_returns_not_started(self):
        hv = HardwareValidator()
        r = hv.validate_sampling_rate(samples=None)
        assert r.result == GateResult.NOT_STARTED

    def test_on_target_rate_passes(self):
        hv = HardwareValidator()
        samples = make_samples(4001, rate_hz=4000.0)
        r = hv.validate_sampling_rate(samples=samples, target_hz=4000.0, tolerance_pct=1.0)
        assert r.result == GateResult.PASS
        assert abs(r.details["deviation_pct"]) <= 1.0

    def test_off_target_rate_fails(self):
        hv = HardwareValidator()
        # 3500 SPS — >10% off
        samples = make_samples(3501, rate_hz=3500.0)
        r = hv.validate_sampling_rate(samples=samples, target_hz=4000.0, tolerance_pct=1.0)
        assert r.result == GateResult.FAIL

    def test_invalid_timestamps_fails(self):
        hv = HardwareValidator()
        # All zero timestamps
        samples = [
            Sample(i, 0, 0, 0, 0, 0, 0, 0) for i in range(10)
        ]
        r = hv.validate_sampling_rate(samples=samples)
        assert r.result == GateResult.FAIL

    def test_single_sample_returns_not_started(self):
        hv = HardwareValidator()
        r = hv.validate_sampling_rate(samples=make_samples(1))
        assert r.result == GateResult.NOT_STARTED


# ── H3: Synchronisation ───────────────────────────────────────────────────────

class TestH3Synchronization:
    def test_no_samples_returns_not_started(self):
        hv = HardwareValidator()
        r = hv.validate_synchronization(samples=None)
        assert r.result == GateResult.NOT_STARTED

    def test_always_not_started_even_with_data(self):
        """
        H3 must always return NOT_STARTED because software cannot verify
        physical simultaneous acquisition — only hardware measurement can.
        """
        hv = HardwareValidator()
        samples = make_samples(100)
        r = hv.validate_synchronization(samples=samples)
        assert r.result == GateResult.NOT_STARTED

    def test_structural_check_reports_monotone_ids(self):
        """Good sample stream: structural_check_passed should be True."""
        hv = HardwareValidator()
        samples = make_samples(50)
        r = hv.validate_synchronization(samples=samples)
        assert r.details["structural_check_passed"] is True

    def test_structural_check_reports_non_monotone_ids(self):
        """Non-monotone sample_ids should be detected in structural_issues."""
        hv = HardwareValidator()
        samples = make_samples(50)
        # Inject a non-monotone id
        samples[25] = Sample(
            sample_id=10,           # goes backwards
            timestamp_us=samples[25].timestamp_us,
            EEG_Fp1=0, EEG_Fp2=0,
            EMG_Submental=0, EMG_Orbicularis=0,
            EMG_Zygomaticus=0, EMG_Buccinator=0,
        )
        r = hv.validate_synchronization(samples=samples)
        # Still NOT_STARTED (physical hw required) but structural issue recorded
        assert r.result == GateResult.NOT_STARTED
        assert r.details["structural_check_passed"] is False
        assert r.details["non_monotone_sample_ids"] > 0

    def test_note_says_hardware_required(self):
        hv = HardwareValidator()
        samples = make_samples(50)
        r = hv.validate_synchronization(samples=samples)
        assert "hardware" in r.details["note"].lower() or "hardware" in r.description.lower()


# ── H4: Dropped samples ───────────────────────────────────────────────────────

class TestH4DroppedSamples:
    def test_no_samples_returns_not_started(self):
        hv = HardwareValidator()
        r = hv.validate_dropped_samples(samples=None)
        assert r.result == GateResult.NOT_STARTED

    def test_continuous_stream_passes(self):
        hv = HardwareValidator()
        r = hv.validate_dropped_samples(samples=make_samples(100))
        assert r.result == GateResult.PASS
        assert r.details["total_dropped"] == 0

    def test_gap_detected_fails(self):
        hv = HardwareValidator()
        # Build a stream with a real gap: IDs 0-49 then 55-104 (5 samples dropped)
        samples = []
        for i in range(50):
            samples.append(Sample(i, i * 250, 0, 0, 0, 0, 0, 0))
        for i in range(55, 105):
            samples.append(Sample(i, i * 250, 0, 0, 0, 0, 0, 0))
        r = hv.validate_dropped_samples(samples=samples, max_allowed=0)
        assert r.result == GateResult.FAIL
        assert r.details["total_dropped"] == 5

    def test_max_allowed_respected(self):
        hv = HardwareValidator()
        # 2 dropped samples: IDs 0-49, then 52-101
        samples = []
        for i in range(50):
            samples.append(Sample(i, i * 250, 0, 0, 0, 0, 0, 0))
        for i in range(52, 102):
            samples.append(Sample(i, i * 250, 0, 0, 0, 0, 0, 0))
        r = hv.validate_dropped_samples(samples=samples, max_allowed=5)
        assert r.result == GateResult.PASS


# ── H5: Saturation ────────────────────────────────────────────────────────────

class TestH5Saturation:
    def test_no_samples_returns_not_started(self):
        hv = HardwareValidator()
        r = hv.validate_saturation(samples=None)
        assert r.result == GateResult.NOT_STARTED

    def test_adc_max_none_returns_blocked(self):
        """adc_max is HARDWARE-TBD — must return BLOCKED, not PASS."""
        hv = HardwareValidator()
        r = hv.validate_saturation(samples=make_samples(50), adc_max=None)
        assert r.result == GateResult.BLOCKED

    def test_no_saturation_passes(self):
        hv = HardwareValidator()
        adc_max = 2**23 - 1
        samples = make_samples(50)  # values are 100, well below adc_max
        r = hv.validate_saturation(samples=samples, adc_max=adc_max)
        assert r.result == GateResult.PASS

    def test_saturation_detected_fails(self):
        hv = HardwareValidator()
        adc_max = 200  # low threshold for test purposes
        samples = []
        for i in range(50):
            val = 300 if 10 <= i < 15 else 50  # saturated for 5 consecutive
            samples.append(Sample(i, i * 250, val, 0, 0, 0, 0, 0))
        r = hv.validate_saturation(samples=samples, adc_max=adc_max, consecutive_threshold=3)
        assert r.result == GateResult.FAIL
        assert r.details["saturation_events_per_channel"]["EEG_Fp1"] > 0


# ── H6: Flatline ──────────────────────────────────────────────────────────────

class TestH6Flatline:
    def test_no_samples_returns_not_started(self):
        hv = HardwareValidator()
        r = hv.validate_flatline(samples=None)
        assert r.result == GateResult.NOT_STARTED

    def test_active_signal_passes(self):
        import numpy as np
        hv = HardwareValidator()
        rng = np.random.default_rng(42)
        samples = [
            Sample(i, i * 250,
                   int(rng.integers(-500, 500)),
                   int(rng.integers(-500, 500)),
                   int(rng.integers(-200, 200)),
                   int(rng.integers(-200, 200)),
                   int(rng.integers(-200, 200)),
                   int(rng.integers(-200, 200)))
            for i in range(200)
        ]
        r = hv.validate_flatline(samples=samples, variance_threshold=1.0)
        assert r.result == GateResult.PASS

    def test_flatline_channel_fails(self):
        hv = HardwareValidator()
        # All zeros = flatline
        samples = [Sample(i, i * 250, 0, 0, 0, 0, 0, 0) for i in range(200)]
        r = hv.validate_flatline(samples=samples, variance_threshold=1.0, window_size=100)
        assert r.result == GateResult.FAIL


# ── H7: Noise ─────────────────────────────────────────────────────────────────

class TestH7Noise:
    def test_all_none_returns_not_started(self):
        hv = HardwareValidator()
        r = hv.validate_noise()
        assert r.result == GateResult.NOT_STARTED

    def test_informational_with_data(self):
        """H7 is always informational — no pass/fail threshold."""
        hv = HardwareValidator()
        samples = make_samples(50)
        r = hv.validate_noise(electronics_only_samples=samples)
        # Result is always PASS (informational)
        assert r.result == GateResult.PASS
        assert "electronics_only" in r.details
        assert "EEG_Fp1" in r.details["electronics_only"]


# ── H8: Sustained streaming ───────────────────────────────────────────────────

class TestH8SustainedStreaming:
    def test_no_samples_returns_not_started(self):
        hv = HardwareValidator()
        r = hv.validate_sustained_streaming(samples=None)
        assert r.result == GateResult.NOT_STARTED

    def test_no_expected_rate_returns_not_started(self):
        """
        Without expected_sampling_rate_hz, H8 must refuse to compute
        expected sample count — not fall back to a hard-coded assumption.
        """
        hv = HardwareValidator()
        samples = make_samples(200)
        r = hv.validate_sustained_streaming(
            samples=samples,
            expected_sampling_rate_hz=None,  # TBD
        )
        assert r.result == GateResult.NOT_STARTED
        assert "expected_sampling_rate_hz" in r.description.lower() or "not supplied" in r.description.lower()

    def test_short_duration_fails(self):
        hv = HardwareValidator()
        samples = make_samples(200, rate_hz=4000.0)
        # Duration = 200/4000 = 0.05s, required = 300s
        r = hv.validate_sustained_streaming(
            samples=samples,
            target_duration_s=300.0,
            expected_sampling_rate_hz=4000.0,
        )
        assert r.result == GateResult.FAIL

    def test_sufficient_duration_no_drops_passes(self):
        """Simulate a stream long enough to pass H8."""
        import numpy as np
        hv = HardwareValidator()
        # 310 seconds at 4000 SPS = 1,240,000 samples
        # Using larger interval to avoid generating millions of objects
        n = 1240
        # Space the samples out so duration is ≥300s
        interval_us = int(310 * 1_000_000 / n)  # ~250,000 µs per sample
        samples = [
            Sample(i, i * interval_us, 0, 0, 0, 0, 0, 0) for i in range(n)
        ]
        r = hv.validate_sustained_streaming(
            samples=samples,
            target_duration_s=300.0,
            expected_sampling_rate_hz=4.0,  # matched to our synthetic rate
            max_drop_rate_ppm=1000.0,
        )
        assert r.result == GateResult.PASS
        assert r.details["expected_sampling_rate_hz"] == 4.0  # confirms no hard-code

    def test_hard_coded_4000_not_used(self):
        """
        Regression: validate that the hard-coded 4000 has been removed.
        If we pass expected_sampling_rate_hz=2000, the details must reflect 2000,
        not 4000.
        """
        import numpy as np
        hv = HardwareValidator()
        n = 700
        interval_us = int(350 * 1_000_000 / n)
        samples = [Sample(i, i * interval_us, 0, 0, 0, 0, 0, 0) for i in range(n)]
        r = hv.validate_sustained_streaming(
            samples=samples,
            target_duration_s=300.0,
            expected_sampling_rate_hz=2000.0,  # deliberate non-4000 value
            max_drop_rate_ppm=1000.0,
        )
        # Whatever the result, the stored rate must be 2000, not 4000
        assert r.details.get("expected_sampling_rate_hz") == 2000.0
