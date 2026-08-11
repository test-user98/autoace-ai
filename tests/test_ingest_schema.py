"""Ingest and schema invariants, exercised against the three real clips."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from voicetrial.ingest import (
    SAMPLE_RATE,
    DecodeError,
    channels_are_identical,
    load,
)
from voicetrial.schema import NoiseSeverity, Prediction, coerce

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
CLIPS = ["call_001.ogg", "call_002.ogg", "call_003.ogg"]
EXPECTED_DURATIONS = {"call_001.ogg": 30.9, "call_002.ogg": 35.0, "call_003.ogg": 171.9}

needs_clips = pytest.mark.skipif(
    not all((RAW / c).is_file() for c in CLIPS),
    reason="provided clips not present in data/raw",
)


@needs_clips
@pytest.mark.parametrize("clip", CLIPS)
def test_decodes_to_16k_mono(clip: str):
    audio = load(RAW / clip)

    assert audio.samples.dtype == np.float32
    assert audio.samples.ndim == 1
    assert audio.duration_s == pytest.approx(EXPECTED_DURATIONS[clip], abs=0.5)
    assert audio.samples.size == pytest.approx(
        EXPECTED_DURATIONS[clip] * SAMPLE_RATE, rel=0.02
    )


@needs_clips
@pytest.mark.parametrize("clip", CLIPS)
def test_provided_clips_are_duplicated_mono_not_channel_separated(clip: str):
    """Guards the finding that killed the free speaker-separation shortcut.

    If a future batch IS channel-separated this flips to False, and role
    assignment can take the cheap path instead of diarization.
    """
    assert channels_are_identical(RAW / clip) is True


def test_corrupt_file_raises_decode_error_not_crash(tmp_path: Path):
    bad = tmp_path / "truncated.wav"
    bad.write_bytes(b"RIFF____WAVEfmt not-actually-audio")

    with pytest.raises(DecodeError):
        load(bad)


def test_missing_file_raises_decode_error(tmp_path: Path):
    with pytest.raises(DecodeError):
        load(tmp_path / "nope.wav")


def test_empty_file_raises_decode_error(tmp_path: Path):
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")

    with pytest.raises(DecodeError):
        load(empty)


def test_zero_duration_audio_raises_decode_error(tmp_path: Path):
    silent = tmp_path / "zero.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "anullsrc=r=16000:cl=mono", "-t", "0", str(silent)],
        check=True, capture_output=True,
    )
    with pytest.raises(DecodeError):
        load(silent)


# --- schema invariants -------------------------------------------------------

BASE = {
    "emotional_tone": "neutral",
    "emotional_intensity": "low",
    "background_noise_present": False,
    "background_noise_type": "",
    "background_noise_severity": "none",
    "audio_quality": "clear",
    "speaker_overlap_present": False,
    "long_silence_present": False,
    "confidence": 0.5,
}


def test_valid_row_accepted():
    assert Prediction(**BASE).background_noise_severity is NoiseSeverity.NONE


def test_no_noise_but_severity_set_is_rejected():
    with pytest.raises(ValidationError, match="severity='none'"):
        Prediction(**BASE | {"background_noise_severity": "medium"})


def test_no_noise_but_type_set_is_rejected():
    with pytest.raises(ValidationError, match="type=''"):
        Prediction(**BASE | {"background_noise_type": "TV"})


def test_noise_present_but_severity_none_is_rejected():
    with pytest.raises(ValidationError, match="severity != 'none'"):
        Prediction(**BASE | {"background_noise_present": True})


def test_confidence_out_of_range_is_rejected():
    with pytest.raises(ValidationError):
        Prediction(**BASE | {"confidence": 1.5})


def test_invalid_enum_value_is_rejected():
    with pytest.raises(ValidationError):
        Prediction(**BASE | {"emotional_tone": "angry"})  # not in the brief's enum


def test_coerce_repairs_rather_than_dropping_the_row():
    row = coerce(BASE | {"background_noise_severity": "high", "background_noise_type": "TV"})
    assert row.background_noise_severity is NoiseSeverity.NONE
    assert row.background_noise_type == ""


def test_all_three_provided_labels_satisfy_the_schema():
    """The brief's own labels must validate, or our invariant is wrong."""
    provided = [
        BASE | {"emotional_tone": "upset", "emotional_intensity": "high",
                "confidence": 0.82},
        BASE | {"emotional_tone": "neutral", "emotional_intensity": "medium",
                "background_noise_present": True, "background_noise_type": "TV",
                "background_noise_severity": "medium",
                "speaker_overlap_present": True, "confidence": 0.82},
        BASE | {"emotional_tone": "satisfied", "emotional_intensity": "medium",
                "background_noise_present": True,
                "background_noise_type": "sharp static",
                "background_noise_severity": "medium",
                "speaker_overlap_present": True, "confidence": 0.82},
    ]
    for row in provided:
        Prediction(**row)
