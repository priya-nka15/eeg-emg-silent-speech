"""
Tests for FrameParser (PRD §49 Acquisition tests).
"""
import struct
import warnings
import pytest

from backend.acquisition.frame_parser import (
    FrameParser,
    FrameParseError,
    Sample,
    FRAME_HEADER,
    FRAME_FOOTER,
    FRAME_FORMAT,
    FRAME_SIZE,
    CHANNEL_NAMES,
    N_CHANNELS,
)


def make_valid_frame(
    sample_id=0,
    timestamp_us=0,
    ch_values=None,
    event_marker=0,
    status=0,
) -> bytes:
    ch_values = ch_values or [0] * 6
    payload = struct.pack(
        FRAME_FORMAT,
        sample_id,
        timestamp_us,
        *ch_values,
        event_marker,
        status,
    )
    return FRAME_HEADER + payload + FRAME_FOOTER


@pytest.fixture
def parser():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return FrameParser(warn_on_tbd=False)


class TestFrameParser:

    def test_parse_valid_frame(self, parser):
        frame = make_valid_frame(sample_id=1, timestamp_us=250)
        sample = parser.parse_frame(frame)
        assert sample.sample_id == 1
        assert sample.timestamp_us == 250

    def test_channel_values(self, parser):
        ch_values = [100, -200, 300, -400, 500, -600]
        frame = make_valid_frame(ch_values=ch_values)
        sample = parser.parse_frame(frame)
        assert sample.EEG_Fp1 == 100
        assert sample.EEG_Fp2 == -200
        assert sample.EMG_Submental == 300
        assert sample.EMG_Orbicularis == -400
        assert sample.EMG_Zygomaticus == 500
        assert sample.EMG_Buccinator == -600

    def test_event_marker(self, parser):
        frame = make_valid_frame(event_marker=0x05)
        sample = parser.parse_frame(frame)
        assert sample.event_marker == 0x05

    def test_status_field(self, parser):
        frame = make_valid_frame(status=0b00000011)
        sample = parser.parse_frame(frame)
        assert sample.status == 0b00000011
        assert sample.is_lead_off(0) is True
        assert sample.is_lead_off(1) is True
        assert sample.is_lead_off(2) is False

    def test_bad_header(self, parser):
        frame = b'\x00\x00' + make_valid_frame()[2:]
        with pytest.raises(FrameParseError, match="Bad header"):
            parser.parse_frame(frame)

    def test_bad_footer(self, parser):
        frame = make_valid_frame()[:-2] + b'\x00\x00'
        with pytest.raises(FrameParseError, match="Bad footer"):
            parser.parse_frame(frame)

    def test_wrong_length(self, parser):
        with pytest.raises(FrameParseError, match="Expected"):
            parser.parse_frame(b'\x00' * 10)

    def test_empty_frame(self, parser):
        with pytest.raises(FrameParseError):
            parser.parse_frame(b'')

    def test_sample_counter_gap_detection(self, parser):
        frame0 = make_valid_frame(sample_id=0)
        frame2 = make_valid_frame(sample_id=2)  # gap: sample 1 missing
        parser.parse_frame(frame0)
        parser.parse_frame(frame2)
        assert parser.dropped_samples == 1

    def test_no_gap_when_consecutive(self, parser):
        for i in range(5):
            parser.parse_frame(make_valid_frame(sample_id=i))
        assert parser.dropped_samples == 0

    def test_frames_parsed_counter(self, parser):
        for i in range(3):
            parser.parse_frame(make_valid_frame(sample_id=i))
        assert parser.frames_parsed == 3

    def test_parse_errors_counter(self, parser):
        try:
            parser.parse_frame(b'bad data here!!')
        except FrameParseError:
            pass
        assert parser.parse_errors >= 1

    def test_reset_counters(self, parser):
        parser.parse_frame(make_valid_frame(sample_id=0))
        parser.parse_frame(make_valid_frame(sample_id=5))
        parser.reset_counters()
        assert parser.dropped_samples == 0
        assert parser.frames_parsed == 0

    def test_roundtrip(self, parser):
        """build_frame → parse_frame must reproduce the original sample."""
        original = Sample(
            sample_id=42,
            timestamp_us=10500,
            EEG_Fp1=1000,
            EEG_Fp2=-500,
            EMG_Submental=200,
            EMG_Orbicularis=-100,
            EMG_Zygomaticus=300,
            EMG_Buccinator=150,
            event_marker=3,
            status=1,
        )
        frame = parser.build_frame(original)
        recovered = parser.parse_frame(frame)
        assert recovered.sample_id == original.sample_id
        assert recovered.EEG_Fp1 == original.EEG_Fp1
        assert recovered.EMG_Buccinator == original.EMG_Buccinator
        assert recovered.event_marker == original.event_marker

    def test_six_channels_in_sample(self, parser):
        frame = make_valid_frame()
        sample = parser.parse_frame(frame)
        assert all(hasattr(sample, ch) for ch in CHANNEL_NAMES)

    def test_sample_to_dict(self, parser):
        frame = make_valid_frame(sample_id=7, ch_values=[1, 2, 3, 4, 5, 6])
        sample = parser.parse_frame(frame)
        d = sample.to_dict()
        assert d["sample_id"] == 7
        assert "EEG_Fp1" in d
        assert "EMG_Buccinator" in d

    def test_sample_from_dict_roundtrip(self, parser):
        frame = make_valid_frame(sample_id=99, ch_values=[10, 20, 30, 40, 50, 60])
        sample = parser.parse_frame(frame)
        d = sample.to_dict()
        restored = Sample.from_dict(d)
        assert restored.sample_id == sample.sample_id
        assert restored.EEG_Fp1 == sample.EEG_Fp1
