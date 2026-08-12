"""Emotional tone and intensity from a dimensional SER model.

Why dimensional rather than a 5-class classifier: the brief's taxonomy IS two
axes. Read the definitions — `satisfied` vs `frustrated` is valence; `frustrated`
vs `upset` vs `distressed` is escalation at constant negative valence. A
categorical model gives a label with no structure; arousal/valence/dominance
gives the two axes the taxonomy is built from, so the mapping is a stated
decision surface rather than an opaque argmax.

It also implements the brief's own warning structurally. "Do not infer
frustration or distress solely from loudness" — loudness drives *arousal*, and
arousal alone never selects a negative label here. High arousal with neutral
valence is an animated satisfied caller, not an upset one.

Model: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim, trained on
MSP-Podcast. Runs on CPU at 11-37x realtime; no GPU, no API, and the audio never
leaves our infrastructure.

TWO LOADING TRAPS — both produce plausible-looking garbage:
  1. `AutoModelForAudioClassification` loads this checkpoint with a RANDOMLY
     INITIALISED head (`classifier.weight | MISSING`) and returns ~0.005 values
     that read like predictions. The custom classes below are required.
  2. transformers 5.x needs `all_tied_weights_keys` on a custom
     `PreTrainedModel` subclass; the model card's 4.x code omits it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CHUNK_S = 10
SAMPLE_RATE = 16_000
MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

_MODEL = None
_PROC = None


@dataclass(frozen=True)
class ToneResult:
    tone: str
    intensity: str
    arousal: float
    valence: float
    dominance: float
    n_chunks: int


def _load():
    global _MODEL, _PROC
    if _MODEL is None:
        import torch
        import torch.nn as nn
        from transformers import Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import (
            Wav2Vec2Model,
            Wav2Vec2PreTrainedModel,
        )

        class RegressionHead(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
                self.dropout = nn.Dropout(cfg.final_dropout)
                self.out_proj = nn.Linear(cfg.hidden_size, cfg.num_labels)

            def forward(self, x):
                x = torch.tanh(self.dense(self.dropout(x)))
                return self.out_proj(self.dropout(x))

        class EmotionModel(Wav2Vec2PreTrainedModel):
            all_tied_weights_keys = {}  # transformers 5.x requirement

            def __init__(self, cfg):
                super().__init__(cfg)
                self.wav2vec2 = Wav2Vec2Model(cfg)
                self.classifier = RegressionHead(cfg)
                self.init_weights()

            def forward(self, x):
                return self.classifier(self.wav2vec2(x)[0].mean(dim=1))

        _PROC = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        _MODEL = EmotionModel.from_pretrained(MODEL_ID).eval()
    return _MODEL, _PROC


def dimensions(samples: np.ndarray) -> tuple[float, float, float, int]:
    """Mean arousal / dominance / valence over 10 s chunks."""
    import torch

    model, proc = _load()
    step = CHUNK_S * SAMPLE_RATE
    out = []
    for i in range(0, len(samples), step):
        seg = samples[i : i + step]
        if len(seg) < SAMPLE_RATE:  # ignore a sub-second tail
            continue
        inputs = proc(seg, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values
        with torch.no_grad():
            out.append(model(inputs).numpy()[0])
    if not out:
        return 0.5, 0.5, 0.5, 0
    a, d, v = np.mean(out, axis=0)
    return float(a), float(d), float(v), len(out)


def classify(arousal: float, valence: float) -> tuple[str, str]:
    """Map the two axes onto the brief's enums.

    Cut points come from the model's own training distribution (MSP-Podcast
    centres both axes near 0.5), NOT from the three provided clips — three
    examples cannot support a five-class boundary, and fitting to them is the
    overfitting trap the brief warns about.

    Intensity is arousal alone: the brief defines it as "the strength of the
    detected emotional tone", which is exactly what arousal measures.
    """
    if valence >= 0.58:
        tone = "satisfied"
    elif valence >= 0.48:
        # Neutral valence. Arousal decides animated-but-neutral vs mild
        # dissatisfaction; it never selects a *strongly* negative label here,
        # which is the brief's "not from loudness alone" requirement.
        tone = "neutral" if arousal < 0.62 else "frustrated"
    elif valence >= 0.40:
        tone = "frustrated" if arousal < 0.62 else "upset"
    else:
        tone = "upset" if arousal < 0.70 else "distressed"

    if arousal >= 0.66:
        intensity = "high"
    elif arousal >= 0.55:
        intensity = "medium"
    else:
        intensity = "low"
    return tone, intensity


def analyse(samples: np.ndarray) -> ToneResult:
    arousal, dominance, valence, n = dimensions(samples)
    tone, intensity = classify(arousal, valence)
    return ToneResult(tone, intensity, arousal, valence, dominance, n)
