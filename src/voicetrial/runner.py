"""Batch execution with per-file error isolation.

The brief is explicit: "A single malformed or unsupported file should not cause
the entire batch to fail. The dashboard should identify which file failed and
why." So every failure mode below is caught per item and recorded — the runner
has no path that aborts a batch because one file was bad.
"""

from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .ingest import DecodeError, load
from .manifest import BatchReport, parse_batch
from .predict import Predictor, StubPredictor  # noqa: F401
from .schema import Prediction

SCHEMA_FIELDS = list(Prediction.model_fields)

# Populated when the real predictor cannot load; surfaced in the result.
_FALLBACK_REASON: list[str] = []


@dataclass
class RowResult:
    name: str
    prediction: dict | None = None
    evidence: dict = field(default_factory=dict)
    expected: dict | None = None
    error: str | None = None
    elapsed_s: float = 0.0
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class RunResult:
    rows: list[RowResult] = field(default_factory=list)
    report: BatchReport | None = None
    system_version: str = ""
    predictor_warning: str | None = None
    total_audio_s: float = 0.0
    total_elapsed_s: float = 0.0

    @property
    def succeeded(self) -> list[RowResult]:
        return [r for r in self.rows if r.ok]

    @property
    def failed(self) -> list[RowResult]:
        return [r for r in self.rows if not r.ok]

    @property
    def realtime_factor(self) -> float:
        """Audio seconds processed per wall-clock second. The §1 cost model is
        derived from this, so it is measured rather than assumed."""
        if self.total_elapsed_s <= 0:
            return 0.0
        return self.total_audio_s / self.total_elapsed_s

    def to_json(self) -> str:
        return json.dumps(
            {
                "system_version": self.system_version,
                "predictor_warning": self.predictor_warning,
                # The validation report travels with the results. Without it the
                # downloaded JSON cannot answer "which files were rejected and
                # why", which is exactly what the brief asks the batch flow to
                # report.
                "validation": self.report.to_dict() if self.report else None,
                "summary": {
                    "total": len(self.rows),
                    "succeeded": len(self.succeeded),
                    "failed": len(self.failed),
                    "total_audio_s": round(self.total_audio_s, 3),
                    "total_elapsed_s": round(self.total_elapsed_s, 3),
                    "realtime_factor": round(self.realtime_factor, 2),
                },
                "results": [asdict(r) for r in self.rows],
            },
            indent=2,
        )

    def to_csv(self) -> str:
        """Round-trips the input manifest shape: `name` plus `result_json`,
        preserving the original filename as the brief requires."""
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(["name", "result_json", "error"])
        for row in self.rows:
            writer.writerow([
                row.name,
                json.dumps(row.prediction) if row.prediction else "",
                row.error or "",
            ])
        return buf.getvalue()


def run_batch(folder: Path, predictor: Predictor | None = None) -> RunResult:
    """Validate a batch folder, then predict each valid item."""
    if predictor is None:
        # Real predictor when the models are present; the stub only as a
        # fallback so a missing artifact degrades rather than fails the batch.
        try:
            from .acoustic import AcousticPredictor

            predictor = AcousticPredictor()
        except Exception as exc:
            # Never swallow this. A silent fall back to the stub is how a
            # deployment ends up serving constants while reporting success —
            # the failure must be visible in the result, not just in a log.
            predictor = StubPredictor()
            _FALLBACK_REASON.append(f"{type(exc).__name__}: {exc}")
    report = parse_batch(folder)
    result = RunResult(report=report, system_version=predictor.version)
    if _FALLBACK_REASON:
        result.predictor_warning = _FALLBACK_REASON[-1]

    # CSV rows whose audio is missing are still surfaced as failed rows, so the
    # evaluator sees one line per manifest entry rather than a silent omission.
    for name in report.missing_audio:
        result.rows.append(RowResult(name=name, error="no matching audio file in batch"))
    for name in report.unsupported:
        result.rows.append(RowResult(name=name, error="unsupported file type"))
    for name in report.name_collisions:
        result.rows.append(
            RowResult(name=name, error="filename collides with another file (case-only difference)")
        )
    for name in report.duplicate_rows:
        result.rows.append(
            RowResult(name=name, error="duplicate manifest row — only the first was processed")
        )

    batch_start = time.perf_counter()
    for item in report.items:
        started = time.perf_counter()
        row = RowResult(name=item.name, expected=item.expected)
        try:
            audio = load(item.audio_path)
            row.duration_s = audio.duration_s
            prediction, evidence = predictor.predict(audio)
            row.prediction = prediction.model_dump(mode="json")
            row.evidence = dict(evidence)
            result.total_audio_s += audio.duration_s
        except DecodeError as exc:
            row.error = f"decode failed: {exc}"
        except Exception as exc:  # a model blowing up must not kill the batch
            row.error = f"{type(exc).__name__}: {exc}"
        row.elapsed_s = time.perf_counter() - started
        result.rows.append(row)

    result.total_elapsed_s = time.perf_counter() - batch_start
    return result
