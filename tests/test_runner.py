"""Batch runner: per-file error isolation and output shape.

The brief's requirement is that one bad file never kills the batch, so every
test here asserts the *other* files still produced predictions.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from voicetrial.ingest import Audio
from voicetrial.predict import Evidence, StubPredictor
from voicetrial.runner import run_batch
from voicetrial.schema import Prediction

from .test_manifest import make_audio

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.fixture
def batch(tmp_path: Path) -> Path:
    folder = tmp_path / "evaluation_batch"
    folder.mkdir()
    return folder


def test_stub_predictions_are_schema_valid(batch: Path):
    make_audio(batch / "call_001.wav")
    (batch / "labels.csv").write_text("name,result_json\ncall_001.wav,\n")

    result = run_batch(batch)

    assert len(result.succeeded) == 1
    Prediction(**result.succeeded[0].prediction)  # revalidates the emitted row


def test_corrupt_file_fails_alone(batch: Path):
    make_audio(batch / "good.wav")
    (batch / "bad.wav").write_bytes(b"RIFF____WAVEnot-audio")
    (batch / "labels.csv").write_text("name,result_json\ngood.wav,\nbad.wav,\n")

    result = run_batch(batch)

    assert len(result.succeeded) == 1
    assert len(result.failed) == 1
    assert result.failed[0].name == "bad.wav"
    assert "decode failed" in result.failed[0].error


def test_predictor_exception_fails_alone(batch: Path):
    make_audio(batch / "a.wav")
    make_audio(batch / "b.wav")
    (batch / "labels.csv").write_text("name,result_json\na.wav,\nb.wav,\n")

    class Exploding(StubPredictor):
        def predict(self, audio: Audio) -> tuple[Prediction, Evidence]:
            if audio.source_path.name == "b.wav":
                raise RuntimeError("model went bang")
            return super().predict(audio)

    result = run_batch(batch, predictor=Exploding())

    assert len(result.succeeded) == 1
    assert result.failed[0].error == "RuntimeError: model went bang"


def test_missing_audio_appears_as_a_failed_row_not_a_silent_gap(batch: Path):
    make_audio(batch / "present.wav")
    (batch / "labels.csv").write_text("name,result_json\npresent.wav,\nabsent.wav,\n")

    result = run_batch(batch)

    assert len(result.rows) == 2
    failed = result.failed[0]
    assert failed.name == "absent.wav"
    assert "no matching audio" in failed.error


def test_csv_export_preserves_original_filenames(batch: Path):
    make_audio(batch / "Call_001.WAV")
    (batch / "labels.csv").write_text("name,result_json\ncall_001.wav,\n")

    rows = list(csv.DictReader(io.StringIO(run_batch(batch).to_csv())))

    assert rows[0]["name"] == "Call_001.WAV"
    assert json.loads(rows[0]["result_json"])["emotional_tone"] == "neutral"


def test_json_export_carries_version_and_measured_throughput(batch: Path):
    make_audio(batch / "call_001.wav", seconds=1.0)
    (batch / "labels.csv").write_text("name,result_json\ncall_001.wav,\n")

    payload = json.loads(run_batch(batch).to_json())

    assert payload["system_version"] == StubPredictor.version
    assert payload["summary"]["succeeded"] == 1
    assert payload["summary"]["realtime_factor"] > 0


def test_expected_labels_are_carried_through_for_scoring(batch: Path):
    make_audio(batch / "call_001.wav")
    (batch / "labels.csv").write_text(
        'name,result_json\ncall_001.wav,"{""emotional_tone"":""upset""}"\n'
    )

    result = run_batch(batch)

    assert result.succeeded[0].expected == {"emotional_tone": "upset"}


def test_empty_batch_does_not_crash(batch: Path):
    (batch / "labels.csv").write_text("name,result_json\n")

    result = run_batch(batch)

    assert result.rows == []
    assert result.realtime_factor == 0.0


@pytest.mark.skipif(
    not (RAW / "labels.csv").is_file() or not (RAW / "call_001.ogg").is_file(),
    reason="provided clips not present",
)
def test_end_to_end_on_the_three_provided_clips():
    """The real batch, exactly as AutoAce supplied it."""
    result = run_batch(RAW)

    assert len(result.succeeded) == 3
    assert result.failed == []
    assert result.total_audio_s == pytest.approx(237.8, abs=1.0)
    for row in result.succeeded:
        Prediction(**row.prediction)
        assert row.expected is not None  # labels.csv parsed for all three


def test_json_export_carries_the_validation_report(batch: Path):
    """Every rejection category must be answerable from the downloaded JSON
    alone — it is the artifact the evaluator keeps."""
    make_audio(batch / "good.wav")
    make_audio(batch / "orphan.wav")
    (batch / "notes.txt").write_text("x")
    (batch / "labels.csv").write_text(
        'name,result_json\ngood.wav,\ngood.wav,\nabsent.wav,\nbad.wav,"{oops"\n'
    )

    payload = json.loads(run_batch(batch).to_json())
    validation = payload["validation"]

    assert validation["unmatched_audio"] == ["orphan.wav"]
    assert validation["unsupported"] == ["notes.txt"]
    assert validation["missing_audio"] == ["absent.wav", "bad.wav"]
    assert validation["duplicate_rows"] == ["good.wav"]
    assert validation["ready"] == ["good.wav"]


def test_duplicate_rows_surface_as_failed_rows_too(batch: Path):
    make_audio(batch / "call_001.wav")
    (batch / "labels.csv").write_text(
        "name,result_json\ncall_001.wav,\ncall_001.wav,\n"
    )

    result = run_batch(batch)

    assert len(result.succeeded) == 1
    assert any("duplicate manifest row" in r.error for r in result.failed)
