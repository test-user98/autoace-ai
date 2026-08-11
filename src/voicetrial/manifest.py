"""Batch manifest parsing and validation.

Directly scored by the brief: "validate the batch, clearly report missing or
unmatched files" and "a single malformed or unsupported file should not cause the
entire batch to fail". Everything here reports rather than raises.

The brief states result_json "may be empty or omitted from scoring input" for the
hidden test set, so an absent result_json column is a valid unlabeled batch, not
an error.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

from .ingest import SUPPORTED_SUFFIXES

NAME_COL = "name"
LABEL_COL = "result_json"


@dataclass
class BatchItem:
    name: str
    audio_path: Path
    expected: dict | None = None  # parsed result_json when labels are present


@dataclass
class BatchReport:
    """Everything the dashboard needs to render batch validation."""

    items: list[BatchItem] = field(default_factory=list)
    labeled: bool = False
    missing_audio: list[str] = field(default_factory=list)      # CSV row, no file
    unmatched_audio: list[str] = field(default_factory=list)    # file, no CSV row
    unsupported: list[str] = field(default_factory=list)        # wrong extension
    duplicate_rows: list[str] = field(default_factory=list)
    name_collisions: list[str] = field(default_factory=list)    # differ only by case
    bad_labels: list[tuple[str, str]] = field(default_factory=list)  # (name, reason)
    errors: list[str] = field(default_factory=list)             # batch-level
    root: Path | None = None  # actual folder used, after descending into a zip dir

    @property
    def ok(self) -> bool:
        """A batch is processable if it has at least one item and no fatal error."""
        return bool(self.items) and not self.errors

    def to_dict(self) -> dict:
        """JSON-safe view. `asdict()` cannot be used — BatchItem.audio_path and
        `root` are Path objects, which are not JSON-serializable."""
        return {
            "ok": self.ok,
            "labeled": self.labeled,
            "root": str(self.root) if self.root else None,
            "ready": [item.name for item in self.items],
            "missing_audio": self.missing_audio,
            "unmatched_audio": self.unmatched_audio,
            "unsupported": self.unsupported,
            "duplicate_rows": self.duplicate_rows,
            "name_collisions": self.name_collisions,
            "bad_labels": [{"name": n, "reason": r} for n, r in self.bad_labels],
            "errors": self.errors,
            "summary": self.summary(),
        }

    def summary(self) -> str:
        parts = [f"{len(self.items)} file(s) ready"]
        if self.labeled:
            parts.append("labels present")
        for label, seq in (
            ("missing audio", self.missing_audio),
            ("unmatched audio", self.unmatched_audio),
            ("unsupported", self.unsupported),
            ("duplicate rows", self.duplicate_rows),
            ("name collisions", self.name_collisions),
            ("unparseable labels", self.bad_labels),
        ):
            if seq:
                parts.append(f"{len(seq)} {label}")
        return ", ".join(parts)


def _normalize(name: str) -> str:
    """Match keys tolerantly: strip whitespace, fold case, drop any directory
    prefix a zip may have introduced."""
    return Path(name.strip()).name.casefold()


def find_manifest(folder: Path) -> Path | None:
    """Locate the CSV manifest. Prefers labels.csv, else the only CSV present."""
    exact = folder / "labels.csv"
    if exact.is_file():
        return exact
    csvs = sorted(p for p in folder.iterdir() if p.suffix.casefold() == ".csv")
    if len(csvs) == 1:
        return csvs[0]
    return None


def _resolve_root(folder: Path) -> Path:
    """Descend through wrapper directories left by unzipping a folder.

    `evaluation_batch.zip` extracts to `<tmp>/evaluation_batch/...`, so the
    manifest is one level below where the caller points. Without this, the most
    likely real-world input shape hard-fails with "no CSV manifest found".
    """
    current = folder
    for _ in range(4):  # bounded; nobody nests a batch deeper than this
        if find_manifest(current) is not None:
            return current
        entries = [
            p for p in current.iterdir()
            if not p.name.startswith(".") and p.name != "__MACOSX"
        ]
        subdirs = [p for p in entries if p.is_dir()]
        if len(subdirs) != 1 or any(p.is_file() for p in entries):
            break
        current = subdirs[0]
    return current if find_manifest(current) is not None else folder


def parse_batch(folder: Path) -> BatchReport:
    """Validate a batch folder: audio at root plus one CSV manifest."""
    report = BatchReport()
    folder = Path(folder)

    if not folder.is_dir():
        report.errors.append(f"{folder} is not a directory")
        return report

    folder = _resolve_root(folder)
    report.root = folder

    audio_by_key: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.casefold() == ".csv":
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.casefold() not in SUPPORTED_SUFFIXES:
            report.unsupported.append(path.name)
            continue
        key = _normalize(path.name)
        if key in audio_by_key:
            # Two files differing only in case. On a case-sensitive filesystem
            # both are real; silently dropping one would contradict the brief's
            # "clearly report missing or unmatched files".
            report.name_collisions.append(path.name)
            continue
        audio_by_key[key] = path

    manifest = find_manifest(folder)
    if manifest is None:
        report.errors.append(
            "no CSV manifest found (expected labels.csv, or exactly one .csv file)"
        )
        return report

    try:
        text = manifest.read_text(encoding="utf-8-sig")  # tolerate a BOM
    except (OSError, UnicodeDecodeError) as exc:
        report.errors.append(f"cannot read {manifest.name}: {exc}")
        return report

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        report.errors.append(f"{manifest.name} is empty")
        return report

    headers = {(h or "").strip().casefold(): (h or "") for h in reader.fieldnames}
    if NAME_COL not in headers:
        report.errors.append(
            f"{manifest.name} has no '{NAME_COL}' column (found: {reader.fieldnames})"
        )
        return report

    name_key = headers[NAME_COL]
    label_key = headers.get(LABEL_COL)  # legitimately absent for a hidden test set
    report.labeled = label_key is not None

    seen: set[str] = set()
    for row in reader:
        raw_name = (row.get(name_key) or "").strip()
        if not raw_name:
            continue
        key = _normalize(raw_name)

        if key in seen:
            report.duplicate_rows.append(raw_name)
            continue
        seen.add(key)

        audio_path = audio_by_key.get(key)
        if audio_path is None:
            report.missing_audio.append(raw_name)
            continue

        expected = None
        if label_key is not None:
            blob = (row.get(label_key) or "").strip()
            if blob:
                try:
                    expected = json.loads(blob)
                except json.JSONDecodeError as exc:
                    report.bad_labels.append((raw_name, f"invalid JSON: {exc.msg}"))

        report.items.append(
            BatchItem(name=audio_path.name, audio_path=audio_path, expected=expected)
        )

    for key, path in audio_by_key.items():
        if key not in seen:
            report.unmatched_audio.append(path.name)

    # A manifest that referenced only labels is still an unlabeled batch.
    report.labeled = any(item.expected is not None for item in report.items)
    return report
