"""
URF/A-SSR Mock Data Source
PRD §40 — Mock mode: no hardware required.
All mock data is explicitly labelled MOCK_DATA and must never enter official research results.
"""
from __future__ import annotations

import time
from typing import Iterator

import numpy as np

from backend.acquisition.frame_parser import Sample

MOCK_DATA_LABEL = "MOCK_DATA"
"""All data produced by MockSource must be tagged with this label."""


class MockSource:
    """
    Generates synthetic signal samples for software testing.

    Generates physiologically-plausible shaped noise — NOT real biosignals.
    Must never be used in research results.
    """

    def __init__(
        self,
        sampling_rate_hz: float = 4000.0,
        seed: int = 42,
        label: str = MOCK_DATA_LABEL,
    ) -> None:
        self.sampling_rate_hz = sampling_rate_hz
        self.seed = seed
        self.label = label
        self._rng = np.random.default_rng(seed)
        self._sample_id: int = 0
        self._start_ns = time.monotonic_ns()
        self._is_mock = True  # Permanent flag — cannot be cleared

    def generate_sample(self, event_marker: int = 0) -> Sample:
        """Generate one synthetic sample."""
        timestamp_us = int((time.monotonic_ns() - self._start_ns) / 1000)
        # Low-amplitude noise shaped to look vaguely like biosignals
        eeg_noise = self._rng.integers(-500, 500, size=2)
        emg_noise = self._rng.integers(-200, 200, size=4)

        sample = Sample(
            sample_id=self._sample_id,
            timestamp_us=timestamp_us,
            EEG_Fp1=int(eeg_noise[0]),
            EEG_Fp2=int(eeg_noise[1]),
            EMG_Submental=int(emg_noise[0]),
            EMG_Orbicularis=int(emg_noise[1]),
            EMG_Zygomaticus=int(emg_noise[2]),
            EMG_Buccinator=int(emg_noise[3]),
            event_marker=event_marker,
            status=0,
        )
        self._sample_id += 1
        return sample

    def stream(self, n_samples: int = None) -> Iterator[Sample]:
        """
        Yield samples at the configured rate.
        If n_samples is None, streams indefinitely.
        """
        interval = 1.0 / self.sampling_rate_hz
        count = 0
        while n_samples is None or count < n_samples:
            yield self.generate_sample()
            count += 1
            time.sleep(interval)

    def generate_batch(self, n: int) -> list:
        """Generate n samples without rate-limiting (for testing)."""
        return [self.generate_sample() for _ in range(n)]

    @property
    def is_mock(self) -> bool:
        return True
