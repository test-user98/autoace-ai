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
    source: str = "whole clip"


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


def customer_audio(samples: np.ndarray, token: str | None = None) -> tuple[np.ndarray, str]:
    """Isolate the customer's speech, falling back to the whole clip.

    The brief asks for "the primary emotional tone expressed by THE CUSTOMER".
    Averaging SER over the entire clip mixes a calm, professional agent with an
    escalated customer, and the longer the call the harder it regresses toward
    the middle — `call_003` has 18 chunks and lands almost exactly on the model's
    prior. That is the most likely reason valence inverts on the provided clips
    while ordering correctly on RAVDESS (24 actors, single speaker per clip,
    AUC 0.747).

    Role assignment: **the speaker who does NOT start the call is the customer.**
    Call-centre convention is that the agent answers and greets first. This is an
    assumption, not a measurement, and it is the first thing to revisit given
    labelled multi-speaker audio.
    """
    import os

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        return samples, "whole clip (no HF token)"
    try:
        import torch
        from pyannote.audio import Pipeline

        # pyannote 4.x renamed this on Pipeline (`token`) while Model still
        # accepts `use_auth_token` — inconsistent between the two APIs, and the
        # wrong name raises TypeError that a broad except turns into a silent
        # fallback. Try both.
        try:
            pipe = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", token=token
            )
        except TypeError:
            pipe = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=token
            )
        if pipe is None:
            return samples, "whole clip (diarization repo not accessible)"
        wav = torch.from_numpy(samples).unsqueeze(0)
        ann = pipe({"waveform": wav, "sample_rate": SAMPLE_RATE})

        spans: dict[str, list[tuple[float, float]]] = {}
        for turn, _, spk in ann.itertracks(yield_label=True):
            spans.setdefault(spk, []).append((turn.start, turn.end))
        if len(spans) < 2:
            return samples, f"whole clip ({len(spans)} speaker detected)"

        first = min(spans, key=lambda s: min(t[0] for t in spans[s]))
        customer = max(
            (s for s in spans if s != first),
            key=lambda s: sum(e - b for b, e in spans[s]),
        )
        keep = np.concatenate([
            samples[int(b * SAMPLE_RATE) : int(e * SAMPLE_RATE)]
            for b, e in sorted(spans[customer])
        ])
        if keep.size < SAMPLE_RATE:
            return samples, "whole clip (customer turns too short)"
        return keep, f"customer={customer} (agent={first}), {keep.size/SAMPLE_RATE:.1f}s"
    except Exception as exc:
        # pyannote 4.x pulls `speaker-diarization-community-1`, a THIRD gated
        # repo beyond segmentation-3.0 and speaker-diarization-3.1. Rather than
        # give up on customer isolation, fall back to pitch clustering — two
        # speakers on a call almost always differ in F0, which needs no gated
        # model at all.
        try:
            return _customer_by_pitch(samples)
        except Exception:
            return samples, f"whole clip ({type(exc).__name__})"


def _customer_by_pitch(samples: np.ndarray) -> tuple[np.ndarray, str]:
    """Split two talkers by fundamental frequency, no pretrained model needed.

    Crude next to real diarization — it cannot survive two speakers with similar
    pitch, and it has no notion of turn continuity. But the alternative is
    averaging emotion across the agent and the customer, which is what makes
    valence invert on these clips, so a rough split is strictly better than none.
    Reported honestly in `source` so a consumer can weight it.
    """
    from .dsp import FRAME, HOP, _frames, _voicing

    fr = _frames(samples)
    voiced = _voicing(fr) > 0.35
    if voiced.sum() < 20:
        return samples, "whole clip (too little voiced speech to split)"

    # Per-frame F0 via autocorrelation peak position.
    x = fr - fr.mean(axis=1, keepdims=True)
    spec = np.fft.rfft(x, n=2 * FRAME, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), axis=1)[:, :FRAME]
    lo, hi = int(SAMPLE_RATE / 350), min(int(SAMPLE_RATE / 70), FRAME - 1)
    f0 = SAMPLE_RATE / (lo + np.argmax(ac[:, lo:hi], axis=1) + 1e-9)

    vf = f0[voiced]
    split = float(np.median(vf))
    low, high = vf[vf <= split], vf[vf > split]
    if len(low) < 10 or len(high) < 10 or (high.mean() - low.mean()) < 15.0:
        return samples, "whole clip (speakers not separable by pitch)"

    # The agent answers first, so whoever opens the call is the agent.
    first_group_is_low = f0[np.argmax(voiced)] <= split
    keep_high = first_group_is_low          # customer is the OTHER group
    sel = voiced & ((f0 > split) if keep_high else (f0 <= split))

    idx = np.where(sel)[0]
    keep = np.concatenate([samples[i * HOP : i * HOP + FRAME] for i in idx])
    if keep.size < SAMPLE_RATE:
        return samples, "whole clip (customer turns too short)"
    mean_f0 = (high if keep_high else low).mean()
    return keep, f"pitch-split customer (F0~{mean_f0:.0f}Hz, {keep.size/SAMPLE_RATE:.1f}s)"


def analyse(samples: np.ndarray, diarize: bool = False) -> ToneResult:
    """Tone and intensity.

    `diarize` defaults to **False**. Customer isolation via the pitch-split
    fallback was measured and made accuracy WORSE (tone 1/3 -> 0/3): valence
    moved the wrong way at both ends, and call_001's "customer" came back at
    F0 ~287 Hz, implausible for conversational speech, so the split grabbed the
    wrong frames. Shipping a change that measurably degrades results would be
    reasoning from a hypothesis instead of from evidence.

    The hypothesis is NOT disproven — the instrument was too crude to test it.
    Re-enable once pyannote/speaker-diarization-community-1 is accepted and real
    turn boundaries are available.
    """
    speech, source = (customer_audio(samples) if diarize else (samples, "whole clip"))
    arousal, dominance, valence, n = dimensions(speech)
    tone, intensity = classify(arousal, valence)
    return ToneResult(tone, intensity, arousal, valence, dominance, n, source)
