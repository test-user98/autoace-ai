"""The adversarial batch matrix from PLAN.md §10.

Every case must produce a clear per-file report and must NOT fail the batch.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from voicetrial.manifest import parse_batch

LABELED_ROW = (
    '"{""emotional_tone"":""upset"",""emotional_intensity"":""high"",'
    '""background_noise_present"":false,""background_noise_type"":"""",'
    '""background_noise_severity"":""none"",""audio_quality"":""clear"",'
    '""speaker_overlap_present"":false,""long_silence_present"":false,'
    '""confidence"":0.82}"'
)


def make_audio(path: Path, seconds: float = 0.5, fmt: str | None = None) -> None:
    """Generate a tiny valid audio file in whatever format the suffix implies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
    ]
    if fmt:
        cmd += ["-f", fmt]
    cmd.append(str(path))
    subprocess.run(cmd, check=True, capture_output=True)


@pytest.fixture
def batch(tmp_path: Path) -> Path:
    folder = tmp_path / "evaluation_batch"
    folder.mkdir()
    return folder


def test_happy_path_labeled(batch: Path):
    make_audio(batch / "call_001.wav")
    (batch / "labels.csv").write_text(f"name,result_json\ncall_001.wav,{LABELED_ROW}\n")

    report = parse_batch(batch)

    assert report.ok
    assert len(report.items) == 1
    assert report.labeled
    assert report.items[0].expected["emotional_tone"] == "upset"


def test_result_json_column_absent_is_valid_hidden_test_set(batch: Path):
    """The brief says result_json 'may be empty or omitted' — not an error."""
    make_audio(batch / "call_001.wav")
    (batch / "labels.csv").write_text("name\ncall_001.wav\n")

    report = parse_batch(batch)

    assert report.ok
    assert len(report.items) == 1
    assert not report.labeled
    assert report.items[0].expected is None


def test_result_json_empty_is_valid(batch: Path):
    make_audio(batch / "call_001.wav")
    (batch / "labels.csv").write_text("name,result_json\ncall_001.wav,\n")

    report = parse_batch(batch)

    assert report.ok
    assert not report.labeled
    assert report.items[0].expected is None


def test_csv_row_with_no_audio_is_reported_not_fatal(batch: Path):
    make_audio(batch / "call_001.wav")
    (batch / "labels.csv").write_text(
        "name,result_json\ncall_001.wav,\ncall_999.wav,\n"
    )

    report = parse_batch(batch)

    assert report.ok  # the batch still processes
    assert report.missing_audio == ["call_999.wav"]
    assert len(report.items) == 1


def test_audio_with_no_csv_row_is_reported_not_fatal(batch: Path):
    make_audio(batch / "call_001.wav")
    make_audio(batch / "orphan.wav")
    (batch / "labels.csv").write_text("name,result_json\ncall_001.wav,\n")

    report = parse_batch(batch)

    assert report.ok
    assert report.unmatched_audio == ["orphan.wav"]
    assert len(report.items) == 1


def test_mixed_formats_all_accepted(batch: Path):
    make_audio(batch / "a.wav")
    make_audio(batch / "b.mp3")
    make_audio(batch / "c.ogg")
    make_audio(batch / "d.flac")
    (batch / "labels.csv").write_text(
        "name,result_json\na.wav,\nb.mp3,\nc.ogg,\nd.flac,\n"
    )

    report = parse_batch(batch)

    assert report.ok
    assert len(report.items) == 4


def test_unsupported_extension_is_reported_not_fatal(batch: Path):
    make_audio(batch / "call_001.wav")
    (batch / "notes.txt").write_text("hello")
    (batch / "labels.csv").write_text("name,result_json\ncall_001.wav,\n")

    report = parse_batch(batch)

    assert report.ok
    assert report.unsupported == ["notes.txt"]
    assert len(report.items) == 1


def test_duplicate_rows_reported_once(batch: Path):
    make_audio(batch / "call_001.wav")
    (batch / "labels.csv").write_text(
        "name,result_json\ncall_001.wav,\ncall_001.wav,\n"
    )

    report = parse_batch(batch)

    assert report.ok
    assert report.duplicate_rows == ["call_001.wav"]
    assert len(report.items) == 1


def test_filename_case_mismatch_still_matches(batch: Path):
    make_audio(batch / "Call_001.WAV")
    (batch / "labels.csv").write_text("name,result_json\ncall_001.wav,\n")

    report = parse_batch(batch)

    assert report.ok
    assert len(report.items) == 1
    assert report.items[0].name == "Call_001.WAV"  # real filename preserved


def test_unparseable_label_reported_row_still_processed(batch: Path):
    make_audio(batch / "call_001.wav")
    (batch / "labels.csv").write_text('name,result_json\ncall_001.wav,"{not json"\n')

    report = parse_batch(batch)

    assert report.ok
    assert len(report.bad_labels) == 1
    assert len(report.items) == 1  # still predicted, just not scored


def test_bom_and_whitespace_headers_tolerated(batch: Path):
    make_audio(batch / "call_001.wav")
    (batch / "labels.csv").write_text(
        "﻿ name , result_json \ncall_001.wav,\n", encoding="utf-8"
    )

    report = parse_batch(batch)

    assert report.ok
    assert len(report.items) == 1


def test_zipped_folder_shape_is_resolved(batch: Path):
    """`evaluation_batch.zip` extracts to a wrapper dir — the most likely real
    input shape. Must not hard-fail with 'no CSV manifest found'."""
    inner = batch / "evaluation_batch"
    inner.mkdir()
    make_audio(inner / "call_001.wav")
    (inner / "labels.csv").write_text("name,result_json\ncall_001.wav,\n")

    report = parse_batch(batch)

    assert report.ok, report.errors
    assert len(report.items) == 1
    assert report.root == inner


def test_macosx_wrapper_dir_ignored_when_descending(batch: Path):
    inner = batch / "evaluation_batch"
    inner.mkdir()
    (batch / "__MACOSX").mkdir()
    make_audio(inner / "call_001.wav")
    (inner / "labels.csv").write_text("name,result_json\ncall_001.wav,\n")

    report = parse_batch(batch)

    assert report.ok, report.errors
    assert len(report.items) == 1


def _fs_is_case_sensitive(folder: Path) -> bool:
    probe = folder / "_CaseProbe"
    probe.write_text("x")
    try:
        return not (folder / "_caseprobe").exists()
    finally:
        probe.unlink()


def test_case_only_filename_collision_is_reported_not_silent(batch: Path):
    """On a case-sensitive filesystem both files are real. Dropping one silently
    would contradict 'clearly report missing or unmatched files'.

    Skips on macOS (case-insensitive), which is precisely why this defect is
    invisible locally — the eval container is Linux, where it bites.
    """
    if not _fs_is_case_sensitive(batch):
        pytest.skip("case-insensitive filesystem cannot hold both names")

    make_audio(batch / "call_001.wav")
    make_audio(batch / "Call_001.wav")
    (batch / "labels.csv").write_text("name,result_json\ncall_001.wav,\n")

    report = parse_batch(batch)

    assert report.ok
    # Exactly one wins; the other is reported rather than vanishing.
    assert len(report.items) == 1
    assert len(report.name_collisions) == 1
    reported = set(report.name_collisions) | {report.items[0].name}
    assert reported == {"call_001.wav", "Call_001.wav"}


def test_collision_detection_logic_is_reachable(batch: Path, monkeypatch):
    """Filesystem-independent proof that the collision branch works, since the
    test above cannot run on macOS."""
    make_audio(batch / "call_001.wav")
    make_audio(batch / "other.wav")
    (batch / "labels.csv").write_text("name,result_json\ncall_001.wav,\n")

    # Force both real files to normalize to the same key.
    monkeypatch.setattr("voicetrial.manifest._normalize", lambda name: "same-key")

    report = parse_batch(batch)

    assert len(report.name_collisions) == 1


def test_missing_manifest_is_a_batch_level_error(batch: Path):
    make_audio(batch / "call_001.wav")

    report = parse_batch(batch)

    assert not report.ok
    assert "no CSV manifest" in report.errors[0]


def test_manifest_without_name_column_is_a_batch_level_error(batch: Path):
    make_audio(batch / "call_001.wav")
    (batch / "labels.csv").write_text("filename,result_json\ncall_001.wav,\n")

    report = parse_batch(batch)

    assert not report.ok
    assert "no 'name' column" in report.errors[0]
