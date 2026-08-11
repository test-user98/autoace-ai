"""The prediction interface every stage plugs into.

Phase 0.5 ships `StubPredictor` so the whole system — upload, validate, process,
review, download — is deployable and testable before any model exists. Later
phases replace the body of `predict()`; nothing downstream changes.
"""

from __future__ import annotations

from typing import Protocol

from .ingest import Audio
from .schema import Prediction, coerce

# Version stamped onto every row so a result can always be traced back to the
# code that produced it. Bump when prediction behaviour changes.
SYSTEM_VERSION = "0.1.0-stub"


class Evidence(dict):
    """Per-field evidence: {field_name: {...}}.

    Deliberately a plain dict — each stage owns the shape of its own entry, and
    the dashboard renders whatever is there. What matters is that evidence is
    recorded alongside every value rather than narrated after the fact.
    """


class Predictor(Protocol):
    """Implemented by the stub, then by the real pipeline."""

    version: str

    def predict(self, audio: Audio) -> tuple[Prediction, Evidence]: ...


class StubPredictor:
    """Emits a schema-valid constant row.

    Not a placeholder to be embarrassed about: it makes the deployment testable
    end to end, and it is the fixture the batch-runner tests assert against.
    """

    version = SYSTEM_VERSION

    def predict(self, audio: Audio) -> tuple[Prediction, Evidence]:
        row = coerce(
            {
                "emotional_tone": "neutral",
                "emotional_intensity": "low",
                "background_noise_present": False,
                "background_noise_type": "",
                "background_noise_severity": "none",
                "audio_quality": "clear",
                "speaker_overlap_present": False,
                "long_silence_present": False,
                "confidence": 0.0,  # honest: a stub knows nothing
            }
        )
        evidence = Evidence(
            _stub=True,
            duration_s=round(audio.duration_s, 3),
            note="stub predictor — no analysis performed",
        )
        return row, evidence
