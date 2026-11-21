"""
URF/A-SSR Windowing Module
PRD §25 — Configurable windowing, pilot-validated.
Window size/stride/alignment/label_policy all TBD from pilot.
A window crossing incompatible event boundaries follows a documented policy.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class LabelPolicy(str, Enum):
    MAJORITY    = "majority_label"  # label = class occupying most of window
    STRICT      = "strict"          # discard if window crosses boundary
    ONSET_ALIGNED = "onset_aligned" # windows aligned to trial onset only


@dataclass
class WindowingConfig:
    """
    All parameters TBD until pilot analysis (PRD §25).
    Using defaults here only for unit testing; production values
    must be determined from pilot and frozen before main data collection.
    """
    window_length_samples: int = 0        # PILOT-VALIDATED — 0 means TBD
    stride_samples: int = 0               # PILOT-VALIDATED — 0 means TBD
    alignment: str = "left"               # PILOT-VALIDATED
    label_policy: LabelPolicy = LabelPolicy.STRICT  # PILOT-VALIDATED
    windowing_version: str = "TBD"        # PILOT-VALIDATED

    def validate(self) -> List[str]:
        issues = []
        if self.window_length_samples <= 0:
            issues.append(
                "window_length_samples is 0 (TBD) — must be set from pilot analysis"
            )
        if self.stride_samples <= 0:
            issues.append("stride_samples is 0 (TBD) — must be set from pilot analysis")
        if self.windowing_version == "TBD":
            issues.append("windowing_version is TBD — not suitable for final mode")
        return issues

    def is_tbd(self) -> bool:
        return self.window_length_samples <= 0 or self.stride_samples <= 0


@dataclass
class Window:
    """A single windowed segment."""
    data: np.ndarray          # shape (window_length, n_channels)
    label: Optional[str]
    start_sample: int
    end_sample: int
    source_id: str = ""
    discarded: bool = False
    discard_reason: str = ""


class Windower:
    """
    Extracts overlapping windows from preprocessed signal data.

    PRD §25: Window parameters must be pilot-validated.
    This class warns if TBD parameters are used.
    """

    def __init__(self, config: WindowingConfig) -> None:
        self.config = config
        issues = config.validate()
        if issues:
            warnings.warn(
                f"WindowingConfig has TBD fields: {issues}. "
                "Do not use for final evaluation without pilot validation.",
                UserWarning,
                stacklevel=2,
            )

    def extract(
        self,
        data: np.ndarray,
        labels_by_sample: Optional[dict] = None,
        source_id: str = "",
    ) -> List[Window]:
        """
        Extract windows from a (n_samples, n_channels) array.

        Args:
            data: Preprocessed signal array.
            labels_by_sample: dict mapping sample_index → label string.
                If None, all windows get label=None.
            source_id: Recording identifier.

        Returns:
            List of Window objects.
        """
        if self.config.is_tbd():
            raise RuntimeError(
                "Cannot window: window_length_samples or stride_samples is TBD. "
                "Set from pilot analysis before use."
            )

        n_samples, n_channels = data.shape
        wl = self.config.window_length_samples
        stride = self.config.stride_samples

        if wl > n_samples:
            warnings.warn(
                f"Window length ({wl}) > data length ({n_samples}). "
                "No windows extracted.",
                UserWarning,
            )
            return []

        windows = []
        start = 0
        while start + wl <= n_samples:
            end = start + wl
            segment = data[start:end]
            label = self._assign_label(start, end, labels_by_sample)
            w = Window(
                data=segment.copy(),
                label=label,
                start_sample=start,
                end_sample=end - 1,
                source_id=source_id,
            )

            # Apply label policy for boundary crossings
            if label is None and labels_by_sample is not None:
                if self.config.label_policy == LabelPolicy.STRICT:
                    w.discarded = True
                    w.discard_reason = "boundary_crossing_strict"

            windows.append(w)
            start += stride

        return windows

    def extract_trial(
        self,
        data: np.ndarray,
        label: str,
        source_id: str = "",
    ) -> List[Window]:
        """
        Extract windows from a single trial (all samples from same label).
        """
        n_samples = data.shape[0]
        wl = self.config.window_length_samples
        stride = self.config.stride_samples

        if self.config.is_tbd():
            raise RuntimeError("Window parameters are TBD — set from pilot.")

        windows = []
        start = 0
        while start + wl <= n_samples:
            end = start + wl
            windows.append(Window(
                data=data[start:end].copy(),
                label=label,
                start_sample=start,
                end_sample=end - 1,
                source_id=source_id,
            ))
            start += stride
        return windows

    def _assign_label(
        self,
        start: int,
        end: int,
        labels_by_sample: Optional[dict],
    ) -> Optional[str]:
        """Assign a label to a window based on the configured label policy."""
        if labels_by_sample is None:
            return None

        window_labels = [
            labels_by_sample.get(i)
            for i in range(start, end)
            if labels_by_sample.get(i) is not None
        ]

        if not window_labels:
            return None

        if self.config.label_policy == LabelPolicy.MAJORITY:
            from collections import Counter
            most_common = Counter(window_labels).most_common(1)
            return most_common[0][0] if most_common else None

        if self.config.label_policy == LabelPolicy.STRICT:
            unique = set(window_labels)
            return window_labels[0] if len(unique) == 1 else None

        if self.config.label_policy == LabelPolicy.ONSET_ALIGNED:
            return window_labels[0] if window_labels else None

        return None

    def windows_to_arrays(
        self,
        windows: List[Window],
        include_discarded: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert list of windows to (X, y) arrays for model input.

        Returns:
            X: shape (n_windows, window_length, n_channels)
            y: shape (n_windows,) of label strings
        """
        filtered = windows if include_discarded else [w for w in windows if not w.discarded]
        if not filtered:
            raise ValueError("No valid windows to convert.")

        X = np.stack([w.data for w in filtered], axis=0)
        y = np.array([w.label for w in filtered])
        return X, y
