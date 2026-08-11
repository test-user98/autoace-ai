"""Required output schema, verbatim from the AutoAce brief, plus the hard
consistency invariants the brief's own labels imply."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class EmotionalTone(StrEnum):
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"
    FRUSTRATED = "frustrated"
    UPSET = "upset"
    DISTRESSED = "distressed"


class EmotionalIntensity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class NoiseSeverity(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AudioQuality(StrEnum):
    CLEAR = "clear"
    SLIGHTLY_IMPAIRED = "slightly_impaired"
    SEVERELY_IMPAIRED = "severely_impaired"


class Prediction(BaseModel):
    """One row of output. Field order matches the brief."""

    emotional_tone: EmotionalTone
    emotional_intensity: EmotionalIntensity
    background_noise_present: bool
    background_noise_type: str = ""
    background_noise_severity: NoiseSeverity
    audio_quality: AudioQuality
    speaker_overlap_present: bool
    long_silence_present: bool
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _noise_fields_agree(self) -> Prediction:
        """`background_noise_present` gates the other two noise fields.

        Confirmed by the provided labels.csv: call_001 has present=false with
        severity="none" and type="". Enforcing it here prevents the model from
        emitting self-contradictory rows, which is free accuracy on three of the
        nine scored fields.
        """
        if not self.background_noise_present:
            if self.background_noise_severity is not NoiseSeverity.NONE:
                raise ValueError(
                    "background_noise_present=false requires severity='none', "
                    f"got '{self.background_noise_severity}'"
                )
            if self.background_noise_type != "":
                raise ValueError(
                    "background_noise_present=false requires type='', "
                    f"got '{self.background_noise_type}'"
                )
        elif self.background_noise_severity is NoiseSeverity.NONE:
            raise ValueError(
                "background_noise_present=true requires severity != 'none'"
            )
        return self


def coerce(raw: dict) -> Prediction:
    """Repair the consistency invariant rather than dropping a row.

    A downstream stage that disagrees with itself should still emit a valid row;
    `background_noise_present` is the authority because it is measured directly
    from the residual energy rather than inferred.
    """
    row = dict(raw)
    if not row.get("background_noise_present", False):
        row["background_noise_severity"] = NoiseSeverity.NONE
        row["background_noise_type"] = ""
    elif row.get("background_noise_severity") == NoiseSeverity.NONE:
        row["background_noise_severity"] = NoiseSeverity.LOW
    return Prediction(**row)
