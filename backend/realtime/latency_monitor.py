"""
URF/A-SSR Latency Monitor
PRD §41 — Measure, do not assume. Do not invent a latency target before
the complete pipeline has been measured.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Dict, List, Optional


@dataclass
class LatencyStats:
    stage: str
    n_measurements: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "n": self.n_measurements,
            "mean_ms": round(self.mean_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
        }


class LatencyMonitor:
    """
    Measures per-stage processing latency.

    Stages (PRD §41):
        - preprocessing
        - feature_extraction
        - emg_inference
        - eeg_inference
        - fusion
        - end_to_end

    Usage:
        with monitor.measure("preprocessing"):
            ... do preprocessing ...
    """

    def __init__(self) -> None:
        self._measurements: Dict[str, List[float]] = defaultdict(list)
        self._dropped_samples: int = 0
        self._packet_loss: int = 0
        self._buffer_overflows: int = 0

    def record(self, stage: str, duration_ms: float) -> None:
        self._measurements[stage].append(duration_ms)

    def measure(self, stage: str) -> "_LatencyContext":
        return _LatencyContext(self, stage)

    def record_dropped_sample(self) -> None:
        self._dropped_samples += 1

    def record_packet_loss(self) -> None:
        self._packet_loss += 1

    def record_buffer_overflow(self) -> None:
        self._buffer_overflows += 1

    def stats(self, stage: str) -> Optional[LatencyStats]:
        measurements = self._measurements.get(stage, [])
        if len(measurements) < 2:
            return None
        sorted_m = sorted(measurements)
        p95_idx = int(0.95 * len(sorted_m))
        return LatencyStats(
            stage=stage,
            n_measurements=len(measurements),
            mean_ms=mean(measurements),
            std_ms=stdev(measurements),
            min_ms=min(measurements),
            max_ms=max(measurements),
            p95_ms=sorted_m[p95_idx],
        )

    def all_stats(self) -> Dict[str, Optional[LatencyStats]]:
        return {stage: self.stats(stage) for stage in self._measurements}

    def report(self) -> str:
        lines = ["=== Latency Report (measured) ==="]
        for stage, s in sorted(self.all_stats().items()):
            if s:
                lines.append(
                    f"  {stage}: mean={s.mean_ms:.2f}ms "
                    f"p95={s.p95_ms:.2f}ms "
                    f"min={s.min_ms:.2f}ms max={s.max_ms:.2f}ms "
                    f"(n={s.n_measurements})"
                )
        lines.append(f"Dropped samples: {self._dropped_samples}")
        lines.append(f"Packet loss: {self._packet_loss}")
        lines.append(f"Buffer overflows: {self._buffer_overflows}")
        return "\n".join(lines)


class _LatencyContext:
    """Context manager for measuring a single stage."""

    def __init__(self, monitor: LatencyMonitor, stage: str) -> None:
        self._monitor = monitor
        self._stage = stage
        self._start: float = 0.0

    def __enter__(self) -> "_LatencyContext":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        self._monitor.record(self._stage, elapsed_ms)
