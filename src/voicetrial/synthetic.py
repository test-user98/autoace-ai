"""Synthetic evaluation set with exact ground truth.

Three labelled clips cannot support a threshold, let alone a confusion matrix.
But six of the nine fields are acoustic, and acoustic conditions can be
*constructed*: mix a known noise at a known SNR, clip at a known rate, insert a
silence of known length, overlay a second talker at a known offset. The label is
then not an annotator's opinion — it is what was synthesised.

That gives per-class metrics for `background_noise_present`, `_severity`,
`_type`, `audio_quality`, `speaker_overlap_present` and `long_silence_present`
at any sample size, and it is what the thresholds are fitted and reported
against. It cannot stand in for emotional tone, which is genuinely subjective
and is handled separately in Phase 2.

Speech base material is taken from the provided clips only as a *carrier* — the
conditions layered on top are what is being measured, and the tone labels of the
source are never used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ingest import SAMPLE_RATE

RNG_SEED = 20260812  # fixed: the eval set must be byte-reproducible


@dataclass(frozen=True)
class Condition:
    """Exactly what was synthesised — this IS the ground truth."""

    noise_type: str            # "" when none
    snr_db: float | None       # None when no noise
    clipping: float            # fraction of samples driven into clipping
    lowpass_hz: float | None   # muffling; None = untouched
    dropout_rate: float        # fraction of frames zeroed (packet loss)
    silence_s: float           # inserted contiguous silence
    overlap: bool              # second talker mixed in

    @property
    def noise_present(self) -> bool:
        return self.noise_type != ""

    @property
    def noise_severity(self) -> str:
        """Severity is defined by SNR, which is what "how much it interferes"
        means physically. Bands chosen from intelligibility literature, not from
        the provided clips."""
        if not self.noise_present or self.snr_db is None:
            return "none"
        if self.snr_db >= 20:
            return "low"
        if self.snr_db >= 10:
            return "medium"
        return "high"

    @property
    def audio_quality(self) -> str:
        """Quality is degradation of the SPEECH SIGNAL — clipping, band loss,
        dropouts. Additive noise is deliberately excluded: the brief states
        quality is judged independently of background noise, and `call_003`
        (medium static, yet labelled `clear`) confirms it."""
        severe = self.clipping >= 0.02 or self.dropout_rate >= 0.15 or (
            self.lowpass_hz is not None and self.lowpass_hz <= 1500
        )
        slight = self.clipping >= 0.002 or self.dropout_rate >= 0.03 or (
            self.lowpass_hz is not None and self.lowpass_hz <= 3400
        )
        if severe:
            return "severely_impaired"
        if slight:
            return "slightly_impaired"
        return "clear"

    @property
    def long_silence(self) -> bool:
        return self.silence_s >= 3.0

    def to_labels(self) -> dict:
        return {
            "background_noise_present": self.noise_present,
            "background_noise_type": self.noise_type,
            "background_noise_severity": self.noise_severity,
            "audio_quality": self.audio_quality,
            "speaker_overlap_present": self.overlap,
            "long_silence_present": self.long_silence,
        }


# --- noise generators -------------------------------------------------------
# Each mimics the spectral and temporal character of a real source, because the
# classifier is asked for a *type*, not just a level.

def _white(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n)


def _hiss(n: int, rng: np.random.Generator) -> np.ndarray:
    """Static: broadband, tilted toward high frequencies."""
    x = rng.standard_normal(n)
    return np.diff(np.concatenate([[0.0], x]))  # +6 dB/oct


def _road(n: int, rng: np.random.Generator) -> np.ndarray:
    """Road/wind: broadband but strongly low-frequency."""
    x = rng.standard_normal(n)
    out = np.zeros(n)
    acc = 0.0
    for i in range(n):  # one-pole lowpass ≈ 120 Hz
        acc = 0.995 * acc + 0.005 * x[i]
        out[i] = acc
    return out


def _hum(n: int, rng: np.random.Generator) -> np.ndarray:
    """Mains hum: 50 Hz plus harmonics, highly tonal."""
    t = np.arange(n) / SAMPLE_RATE
    sig = sum(np.sin(2 * np.pi * 50 * k * t) / k for k in (1, 2, 3, 4))
    return sig + 0.02 * rng.standard_normal(n)


def _music(n: int, rng: np.random.Generator) -> np.ndarray:
    """Music: sustained harmonic tones over a slow chord change."""
    t = np.arange(n) / SAMPLE_RATE
    out = np.zeros(n)
    for root in (220.0, 277.2, 330.0):
        for k in (1, 2, 3):
            out += np.sin(2 * np.pi * root * k * t) / (k * 3)
    return out * (0.7 + 0.3 * np.sin(2 * np.pi * 0.25 * t))


def _babble(n: int, rng: np.random.Generator, carrier: np.ndarray) -> np.ndarray:
    """Office chatter / TV: speech-shaped and syllable-modulated (~4 Hz).

    Built from reversed, offset speech so the spectrum is genuinely speech-like —
    this is the source that should be hardest to tell from a second talker, and
    it is the one `call_002` is expected to confuse (PLAN.md §9).
    """
    if carrier.size == 0:
        return _white(n, rng)
    reps = int(np.ceil(n / carrier.size)) + 1
    tiled = np.tile(carrier[::-1], reps)[:n]
    t = np.arange(n) / SAMPLE_RATE
    return tiled * (0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t))


NOISE_MAKERS = {
    "static": _hiss,
    "road noise": _road,
    "hum": _hum,
    "music": _music,
    "office chatter": None,   # needs the carrier; handled in `render`
    "television": None,
}


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt((x ** 2).mean()) + 1e-12)


def render(
    speech: np.ndarray,
    cond: Condition,
    rng: np.random.Generator,
    distractor: np.ndarray | None = None,
) -> np.ndarray:
    """Apply `cond` to `speech` and return the degraded signal."""
    x = speech.astype(np.float64).copy()

    if cond.overlap and distractor is not None and distractor.size > 0:
        d = np.resize(distractor, x.size)
        x = x + 0.7 * (_rms(x) / _rms(d)) * d

    if cond.silence_s > 0:
        gap = np.zeros(int(cond.silence_s * SAMPLE_RATE))
        cut = x.size // 2
        x = np.concatenate([x[:cut], gap, x[cut:]])

    if cond.lowpass_hz is not None:
        spec = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(x.size, 1.0 / SAMPLE_RATE)
        spec[freqs > cond.lowpass_hz] *= 0.02
        x = np.fft.irfft(spec, n=x.size)

    if cond.dropout_rate > 0:
        fr = 160
        n_fr = x.size // fr
        kill = rng.random(n_fr) < cond.dropout_rate
        for i in np.where(kill)[0]:
            x[i * fr:(i + 1) * fr] = 0.0

    if cond.noise_present and cond.snr_db is not None:
        maker = NOISE_MAKERS.get(cond.noise_type)
        if maker is None:
            noise = _babble(x.size, rng, speech)
        else:
            noise = maker(x.size, rng)
        target = _rms(x) / (10 ** (cond.snr_db / 20.0))
        x = x + noise * (target / _rms(noise))

    if cond.clipping > 0:
        level = float(np.quantile(np.abs(x), max(0.0, 1.0 - cond.clipping)))
        if level > 0:
            x = np.clip(x, -level, level)

    peak = np.abs(x).max()
    if peak > 0.99:
        x = x * (0.99 / peak)
    return x.astype(np.float32)


def build_conditions(n: int, rng: np.random.Generator) -> list[Condition]:
    """Sample a stratified set of conditions.

    Stratified rather than uniform so every class of every field is populated —
    a random sweep would leave `severely_impaired` and `high` severity almost
    empty, and a confusion matrix with empty rows says nothing.
    """
    types = ["", "static", "road noise", "hum", "music", "office chatter", "television"]
    snrs = [5.0, 8.0, 12.0, 16.0, 22.0, 28.0]

    # Each condition axis is drawn INDEPENDENTLY. An earlier version stepped the
    # axes with modular patterns (i%3, i%4, i%5), which made them correlate: the
    # separability report then credited `dropout_pct` with AUC 0.93 for
    # *noise_present*, a field it has no physical relationship to. Confounded
    # sampling produces confident, wrong conclusions about which feature works.
    conds: list[Condition] = []
    for _ in range(n):
        ntype = types[rng.integers(len(types))]
        conds.append(
            Condition(
                noise_type=ntype,
                snr_db=None if ntype == "" else float(snrs[rng.integers(len(snrs))]),
                clipping=float(rng.choice([0.0, 0.0, 0.005, 0.03])),
                lowpass_hz=None if rng.random() < 0.6 else float(rng.choice([3000.0, 1200.0])),
                dropout_rate=float(rng.choice([0.0, 0.0, 0.05, 0.2])),
                silence_s=float(rng.choice([0.0, 0.0, 4.0, 6.0])),
                overlap=bool(rng.random() < 0.4),
            )
        )
    return conds
