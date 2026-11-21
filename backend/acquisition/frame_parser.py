"""
URF/A-SSR Frame Parser
PRD §7 — Implements the hardware-software contract frame format.
Transport/wire encoding is TBD from firmware. This module defines the
software-side contract and validates incoming frames.

The frame format version and exact wire encoding must come from firmware
documentation. All TBD fields are checked at parse time.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional


# Channel names in canonical order (PRD §12 Sample schema)
CHANNEL_NAMES = [
    "EEG_Fp1",
    "EEG_Fp2",
    "EMG_Submental",
    "EMG_Orbicularis",
    "EMG_Zygomaticus",
    "EMG_Buccinator",
]

N_CHANNELS = len(CHANNEL_NAMES)  # 6

# Status bitfield positions (TBD from firmware — defined structurally here)
STATUS_LEAD_OFF_BITS = list(range(6))  # bits 0-5 = CH1–CH6 lead-off
STATUS_SATURATION_BIT = 6
STATUS_RESERVED_BIT = 7


@dataclass
class Sample:
    """
    A single synchronized sample from all 6 channels.
    PRD §12 Sample data model.
    """
    sample_id: int          # rolling uint32 counter
    timestamp_us: int       # microseconds since session start
    EEG_Fp1: int            # raw 24-bit signed ADC value
    EEG_Fp2: int
    EMG_Submental: int
    EMG_Orbicularis: int
    EMG_Zygomaticus: int
    EMG_Buccinator: int
    event_marker: int = 0   # 0-255
    status: int = 0         # status bitfield

    @property
    def channels(self) -> dict[str, int]:
        return {
            "EEG_Fp1":         self.EEG_Fp1,
            "EEG_Fp2":         self.EEG_Fp2,
            "EMG_Submental":   self.EMG_Submental,
            "EMG_Orbicularis": self.EMG_Orbicularis,
            "EMG_Zygomaticus": self.EMG_Zygomaticus,
            "EMG_Buccinator":  self.EMG_Buccinator,
        }

    def is_lead_off(self, channel_idx: int) -> bool:
        """True if the lead-off bit for channel_idx (0-5) is set."""
        return bool(self.status & (1 << STATUS_LEAD_OFF_BITS[channel_idx]))

    def is_saturated(self) -> bool:
        return bool(self.status & (1 << STATUS_SATURATION_BIT))

    def to_dict(self) -> dict:
        d = {
            "sample_id": self.sample_id,
            "timestamp_us": self.timestamp_us,
            "event_marker": self.event_marker,
            "status": self.status,
        }
        d.update(self.channels)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        # Support both sample_id and sample_index
        sid = d.get("sample_id", d.get("sample_index", 0))
        # Support timestamp_us (int) or host_timestamp (seconds float)
        if "timestamp_us" in d and d["timestamp_us"] not in (None, ""):
            ts = d["timestamp_us"]
        elif "host_timestamp" in d and d["host_timestamp"] not in (None, ""):
            ts = float(d["host_timestamp"]) * 1_000_000
        else:
            ts = 0

        def _get_ch(keys):
            for k in keys:
                if k in d and d[k] not in (None, ""):
                    return d[k]
            return 0

        return cls(
            sample_id=int(float(sid)),
            timestamp_us=int(float(ts)),
            EEG_Fp1=int(float(_get_ch(["EEG_Fp1", "eeg_fp1"]))),
            EEG_Fp2=int(float(_get_ch(["EEG_Fp2", "eeg_fp2"]))),
            EMG_Submental=int(float(_get_ch(["EMG_Submental", "EMG_submental", "emg_submental"]))),
            EMG_Orbicularis=int(float(_get_ch(["EMG_Orbicularis", "EMG_orbicularis", "emg_orbicularis"]))),
            EMG_Zygomaticus=int(float(_get_ch(["EMG_Zygomaticus", "EMG_zygomaticus", "emg_zygomaticus"]))),
            EMG_Buccinator=int(float(_get_ch(["EMG_Buccinator", "EMG_buccinator", "emg_buccinator"]))),
            event_marker=int(float(d.get("event_marker", 0) or 0)),
            status=int(float(d.get("status", 0) or 0)),
        )


# ── Frame parsing ──────────────────────────────────────────────────────────────
#
# IMPORTANT: The actual wire format (endianness, packing, framing bytes) is TBD
# from the firmware team. The structures below define the software contract and
# will be updated when firmware documentation is provided (DL-002, DL-003).
#
# Placeholder binary frame layout (subject to change):
#   Header:       2 bytes  0xAA 0xBB
#   sample_id:    4 bytes  uint32 LE
#   timestamp_us: 8 bytes  uint64 LE
#   CH1..CH6:     6×4=24 bytes  int32 LE each (24-bit value in 32-bit container)
#   event_marker: 1 byte
#   status:       1 byte
#   Footer:       2 bytes  0xCC 0xDD
#   Total:        42 bytes

FRAME_HEADER = b'\xAA\xBB'
FRAME_FOOTER = b'\xCC\xDD'
FRAME_SIZE = 42  # bytes — TBD from firmware
FRAME_FORMAT = "<IQ6iBB"  # little-endian: uint32, uint64, 6×int32, 2×uint8


class FrameParseError(Exception):
    """Raised when a frame cannot be parsed."""


class FrameParser:
    """
    Parses binary frames from the hardware stream into Sample objects.

    The actual wire format is TBD from firmware documentation.
    This implementation uses the placeholder format defined above and will
    be updated when the real format is confirmed.
    """

    FRAME_FORMAT_VERSION = "TBD"  # HARDWARE-TBD

    def __init__(self, warn_on_tbd: bool = True) -> None:
        if warn_on_tbd:
            import warnings
            warnings.warn(
                "FrameParser: FRAME_FORMAT_VERSION is TBD. "
                "Update from firmware documentation before live use.",
                UserWarning,
                stacklevel=2,
            )
        self._last_sample_id: Optional[int] = None
        self.dropped_samples: int = 0
        self.frames_parsed: int = 0
        self.parse_errors: int = 0

    def parse_frame(self, data: bytes) -> Sample:
        """
        Parse a single binary frame into a Sample.

        Raises FrameParseError on malformed data.
        """
        if len(data) != FRAME_SIZE:
            self.parse_errors += 1
            raise FrameParseError(
                f"Expected {FRAME_SIZE} bytes, got {len(data)}"
            )

        if data[:2] != FRAME_HEADER:
            self.parse_errors += 1
            raise FrameParseError(
                f"Bad header: {data[:2].hex()!r} (expected {FRAME_HEADER.hex()!r})"
            )

        if data[-2:] != FRAME_FOOTER:
            self.parse_errors += 1
            raise FrameParseError(
                f"Bad footer: {data[-2:].hex()!r} (expected {FRAME_FOOTER.hex()!r})"
            )

        payload = data[2:-2]  # strip header and footer
        try:
            fields = struct.unpack(FRAME_FORMAT, payload)
        except struct.error as e:
            self.parse_errors += 1
            raise FrameParseError(f"struct.unpack failed: {e}") from e

        sample_id, timestamp_us = fields[0], fields[1]
        ch_values = fields[2:8]   # 6 channel values
        event_marker, status = fields[8], fields[9]

        sample = Sample(
            sample_id=sample_id,
            timestamp_us=timestamp_us,
            EEG_Fp1=ch_values[0],
            EEG_Fp2=ch_values[1],
            EMG_Submental=ch_values[2],
            EMG_Orbicularis=ch_values[3],
            EMG_Zygomaticus=ch_values[4],
            EMG_Buccinator=ch_values[5],
            event_marker=event_marker,
            status=status,
        )

        self._check_counter(sample.sample_id)
        self.frames_parsed += 1
        return sample

    def _check_counter(self, sample_id: int) -> None:
        """Detect gaps in the sample counter (dropped samples, PRD §H4)."""
        if self._last_sample_id is not None:
            expected = (self._last_sample_id + 1) & 0xFFFFFFFF  # uint32 wrap
            if sample_id != expected:
                gap = (sample_id - self._last_sample_id) & 0xFFFFFFFF
                self.dropped_samples += gap - 1
        self._last_sample_id = sample_id

    def build_frame(self, sample: Sample) -> bytes:
        """
        Serialize a Sample to the binary frame format.
        Used for testing and mock sources.
        """
        payload = struct.pack(
            FRAME_FORMAT,
            sample.sample_id,
            sample.timestamp_us,
            sample.EEG_Fp1,
            sample.EEG_Fp2,
            sample.EMG_Submental,
            sample.EMG_Orbicularis,
            sample.EMG_Zygomaticus,
            sample.EMG_Buccinator,
            sample.event_marker,
            sample.status,
        )
        return FRAME_HEADER + payload + FRAME_FOOTER

    def reset_counters(self) -> None:
        self._last_sample_id = None
        self.dropped_samples = 0
        self.frames_parsed = 0
        self.parse_errors = 0

    def stats(self) -> dict:
        return {
            "frames_parsed": self.frames_parsed,
            "parse_errors": self.parse_errors,
            "dropped_samples": self.dropped_samples,
        }
