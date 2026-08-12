"""The real predictor for the six acoustic fields.

Two engines, chosen per field by what actually generalises:

* **Gradient-boosted models over numpy features** for the noise fields. These
  survive a root-carrier holdout because they key on the *character of the noise*
  rather than the character of the voice: `background_noise_type` shows +0.79
  lift over its majority baseline, `_severity` +0.54.

* **pyannote/segmentation-3.0** for overlap and silence. Those two collapsed
  under root-carrier holdout (overlap 0.670, and 0.485 on one fold — worse than a
  constant predictor) because three real voices cannot teach speaker
  independence. pyannote was trained on thousands of speakers and carries it
  natively. This is the one place where a pretrained model is not a shortcut but
  the only correct answer.

`emotional_tone` and `_intensity` remain unimplemented and are reported as
`neutral`/`low` with confidence 0.0. That is deliberate: a guess dressed as a
prediction is worse than a visible gap.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .dsp import measure
from .ingest import Audio
from .predict import Evidence
from .schema import Prediction, coerce

def _model_dir() -> Path:
    """Locate the trained models across local and container layouts.

    Locally the package sits at <repo>/src/voicetrial; in the Modal image the
    source is mounted at /root/src and the models at /root/models, so a single
    relative path cannot serve both. Searching explicit candidates is better than
    a silent miss — a missing model directory previously degraded to the stub
    without saying so, and the deployment served constants while reporting
    success.
    """
    here = Path(__file__).resolve()
    for candidate in (
        Path("/root/models"),
        here.parents[2] / "models",
        here.parents[3] / "models",
        Path.cwd() / "models",
    ):
        if candidate.is_dir() and any(candidate.glob("*.joblib")):
            return candidate
    raise FileNotFoundError(
        "no *.joblib models found; run `uv run python scripts/train_eval.py`"
    )


MODEL_DIR = None
SYSTEM_VERSION = "0.2.0-acoustic"

# Fraction of frames with 2+ concurrent speakers before a clip counts as
# overlapping. Set from the three real clips (0.7% / 1.2% / 1.7% against ground
# truth NO / YES / YES) and therefore PROVISIONAL — n=3 cannot support a
# threshold, and this is the first thing to re-fit given more labelled audio.
OVERLAP_FRAME_PCT = 1.0

# Contiguous non-speech, per pyannote's speaker map rather than an energy floor.
#
# 3.0 s was wrong for conversational audio. pyannote measures gaps of 7.2 / 4.4 /
# 8.5 s in the three real clips, and all three are labelled
# `long_silence_present: false` — so ordinary turn-taking in these calls already
# reaches 8.5 s. The brief asks for "an UNUSUALLY long period of silence or dead
# air that may indicate a call-flow or audio problem", which is a different and
# rarer thing. Set above the observed conversational range.
#
# Calibrated on n=3 and therefore provisional: this is the single threshold most
# in need of re-fitting once more labelled audio exists.
LONG_SILENCE_S = 10.0

_MODELS: dict | None = None
_SEG = None


def _load_models() -> dict:
    global _MODELS
    if _MODELS is None:
        import joblib

        _MODELS = {}
        for path in _model_dir().glob("*.joblib"):
            _MODELS[path.stem] = joblib.load(path)
    return _MODELS


def _load_segmentation():
    """Lazily load pyannote. Returns None if unavailable so the acoustic fields
    still work — a missing optional model must not fail the batch."""
    global _SEG
    if _SEG is None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            return None
        try:
            from pyannote.audio import Inference, Model

            model = Model.from_pretrained(
                "pyannote/segmentation-3.0", use_auth_token=token
            )
            _SEG = Inference(model, step=10.0, duration=10.0)
        except Exception:
            _SEG = False  # tried and failed; do not retry per clip
    return _SEG or None


def _speaker_map(audio: Audio) -> tuple[float, float, str] | None:
    """(overlap %, longest silence s, source) from pyannote, or None."""
    inference = _load_segmentation()
    if inference is None:
        return None
    try:
        import torch

        wav = torch.from_numpy(audio.samples).unsqueeze(0)
        out = inference({"waveform": wav, "sample_rate": 16_000})
        data = np.asarray(out.data)
        active = (data.reshape(-1, data.shape[-1]) > 0.5).sum(axis=1)
        if active.size == 0:
            return None
        frame_s = audio.duration_s / active.size
        longest = run = 0
        for quiet in (active < 1).tolist():
            run = run + 1 if quiet else 0
            longest = max(longest, run)
        return (
            float((active >= 2).mean() * 100.0),
            float(longest * frame_s),
            "pyannote/segmentation-3.0",
        )
    except Exception:
        return None


class AcousticPredictor:
    """Six real fields; tone deliberately left unimplemented."""

    version = SYSTEM_VERSION

    def predict(self, audio: Audio) -> tuple[Prediction, Evidence]:
        analysis = measure(audio.samples)
        feats = asdict(analysis)
        models = _load_models()

        row: dict = {}
        evidence = Evidence(
            engine="dsp+gbm" + ("+pyannote" if _load_segmentation() else ""),
            measurements={
                k: round(v, 4)
                for k, v in feats.items()
                if isinstance(v, (int, float))
            },
            fields={},
        )

        for field, bundle in models.items():
            x = np.nan_to_num(
                np.array([[feats[f] for f in bundle["features"]]], dtype=float)
            )
            value = bundle["model"].predict(x)[0]
            proba = float(bundle["model"].predict_proba(x).max())
            value = {"True": True, "False": False}.get(str(value), value)
            # The generator's vocabulary uses "none" for the absent class; the
            # schema requires the empty string. Without this the consistency
            # validator has to repair every no-noise row.
            if field == "background_noise_type" and value == "none":
                value = ""
            row[field] = value
            evidence["fields"][field] = {
                "value": row[field],
                "probability": round(proba, 3),
                # Every field carries the accuracy of the protocol it was
                # measured under, so a consumer can weight it correctly.
                "holdout_accuracy": round(bundle["holdout_accuracy"], 3),
                "majority_baseline": round(bundle["baseline"], 3),
                "source": "gbm over acoustic features",
            }

        # pyannote overrides the two fields the feature models cannot generalise.
        speaker = _speaker_map(audio)
        if speaker is not None:
            overlap_pct, silence_s, source = speaker
            row["speaker_overlap_present"] = overlap_pct >= OVERLAP_FRAME_PCT
            row["long_silence_present"] = silence_s >= LONG_SILENCE_S
            evidence["fields"]["speaker_overlap_present"] = {
                "value": row["speaker_overlap_present"],
                "overlap_frames_pct": round(overlap_pct, 2),
                "threshold_pct": OVERLAP_FRAME_PCT,
                "source": source,
                "note": "threshold set on n=3 real clips — provisional",
            }
            evidence["fields"]["long_silence_present"] = {
                "value": row["long_silence_present"],
                "longest_silence_s": round(silence_s, 2),
                "threshold_s": LONG_SILENCE_S,
                "source": source,
            }

        # Tone is not implemented. Reported honestly rather than guessed.
        row["emotional_tone"] = "neutral"
        row["emotional_intensity"] = "low"
        evidence["fields"]["emotional_tone"] = {
            "value": "neutral",
            "source": "NOT IMPLEMENTED — placeholder, not a prediction",
        }

        # Confidence reflects only what was actually measured. The two tone
        # fields are unimplemented, so the ceiling is 6/8 of the schema.
        probs = [
            e["probability"]
            for e in evidence["fields"].values()
            if "probability" in e
        ]
        row["confidence"] = round(float(np.mean(probs)) * 0.75, 3) if probs else 0.0

        return coerce(row), evidence
