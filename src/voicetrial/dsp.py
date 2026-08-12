"""Signal analysis for the six acoustic fields.

Deliberately numpy/scipy only — no learned model, no weights to download, no GPU.
Six of the nine required fields are properties of the signal, not of its meaning,
and solving them deterministically is what buys the cost headroom for the one
field that genuinely needs a model (emotional tone).

The architecture enforces the brief's two warnings rather than hoping a threshold
respects them:

* **"do not infer background noise solely from poor audio quality"** — the signal
  is split into estimated speech and estimated noise residual first. Quality is
  measured on the speech estimate; noise is measured on the residual. They read
  different arrays, so one cannot leak into the other.
* **"do not infer frustration solely from loudness"** — no level feature reaches
  the tone decision at all; tone is Phase 2 and consumes prosody z-scored against
  the speaker's own baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ingest import SAMPLE_RATE

FRAME = 400          # 25 ms analysis window
HOP = 160            # 10 ms hop
NFFT = 512
EPS = 1e-12

# A contiguous non-speech run at or beyond this is "dead air". Absolute rather
# than a fraction of the clip: the provided clips run 31 s to 172 s, so any
# proportional rule would mean wildly different things across them.
LONG_SILENCE_S = 3.0

# Below this a frame is digital silence, not a quiet passage. Absolute by
# design: dead air and packet loss are level-independent facts.
DIGITAL_SILENCE_DBFS = -70.0


@dataclass
class Analysis:
    """Everything the field decisions are derived from, kept for the evidence
    trail so a value can always be traced to a measurement."""

    speech_ratio: float
    longest_silence_s: float
    snr_db: float
    noise_floor_dbfs: float
    speech_level_dbfs: float
    clipping_pct: float
    bandwidth_hz: float
    dropout_pct: float
    residual_centroid_hz: float
    residual_flatness: float
    residual_band_ratios: dict = field(default_factory=dict)
    residual_modulation_4hz: float = 0.0
    residual_harmonicity: float = 0.0
    overlap_frames_pct: float = 0.0


def _frames(x: np.ndarray) -> np.ndarray:
    """Split into overlapping analysis frames (n_frames, FRAME)."""
    if x.size < FRAME:
        x = np.pad(x, (0, FRAME - x.size))
    n = 1 + (x.size - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx]


def _db(v: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.asarray(v) + EPS)


def voice_activity(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Adaptive energy VAD.

    A fixed dBFS threshold fails across recordings with different gain, so the
    noise floor is estimated from the clip's own quiet frames and the threshold
    floats above it. Returns (per-frame speech mask, speech ratio, longest
    contiguous silence in seconds).
    """
    fr = _frames(x)
    energy_db = _db((fr ** 2).mean(axis=1))

    # Digital silence (dead air, packet loss, a muted leg) is absolute, not
    # relative. Excluding it before estimating the floor matters: a long zero
    # gap otherwise *becomes* the 10th percentile, dragging the threshold down
    # until the whole clip reads as speech. That alone cost the long-silence
    # detector its accuracy (AUC 0.76 on gaps of known length).
    digital_silence = energy_db < DIGITAL_SILENCE_DBFS
    audible = energy_db[~digital_silence]
    if audible.size < 8:
        return ~digital_silence, float((~digital_silence).mean()), _longest_run(digital_silence)

    floor = np.percentile(audible, 10.0)
    peak = np.percentile(audible, 95.0)
    # Halfway between floor and peak, but never less than 6 dB above the floor:
    # on a clip that is nearly all speech the percentile spread collapses.
    thresh = max(floor + 6.0, floor + 0.35 * (peak - floor))
    mask = (energy_db > thresh) & ~digital_silence

    # Hangover: speech tails fall below threshold before the talker stops, and
    # single-frame dropouts inside a word are not silence.
    mask = _smooth(mask, width=8)

    return mask, float(mask.mean()), _longest_run(~mask)


def _longest_run(flags: np.ndarray) -> float:
    """Longest contiguous True run, in seconds."""
    longest = run = 0
    for f in flags:
        run = run + 1 if f else 0
        longest = max(longest, run)
    return longest * HOP / SAMPLE_RATE


def _smooth(mask: np.ndarray, width: int) -> np.ndarray:
    """Close gaps shorter than `width` frames, then drop islands shorter than
    half that — removes both spurious gaps and spurious detections."""
    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        if not out[i]:
            j = i
            while j < n and not out[j]:
                j += 1
            if 0 < i and j < n and (j - i) < width:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def separate(x: np.ndarray, speech_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the signal into an estimated speech component and a noise residual.

    Spectral subtraction with an over-subtraction factor and a spectral floor.
    The noise magnitude spectrum is estimated from frames the VAD marked as
    non-speech, so this is a genuine estimate from *this* recording rather than
    an assumed profile.

    Returns (clean_mag, residual_mag, noise_profile), all magnitude spectrograms —
    the downstream measurements are spectral, so no inverse transform is needed.
    """
    fr = _frames(x) * np.hanning(FRAME)[None, :]
    spec = np.abs(np.fft.rfft(fr, n=NFFT, axis=1))

    n = min(len(speech_mask), len(spec))
    spec, mask = spec[:n], speech_mask[:n]

    noise_frames = spec[~mask]
    if len(noise_frames) < 3:
        # Almost no silence to learn from: fall back to the per-bin 10th
        # percentile across the whole clip, which approximates a floor.
        noise = np.percentile(spec, 10.0, axis=0)
    else:
        noise = noise_frames.mean(axis=0)

    alpha = 2.0    # over-subtraction; biases toward removing noise
    beta = 0.05    # spectral floor; prevents musical-noise artefacts
    clean = np.maximum(spec - alpha * noise[None, :], beta * spec)
    residual = np.maximum(spec - clean, 0.0)

    return clean, residual, noise


def measure(x: np.ndarray) -> Analysis:
    """Run the full acoustic analysis once; every field decision reads from it."""
    mask, speech_ratio, longest_silence = voice_activity(x)
    clean, residual, noise = separate(x, mask)
    n = min(len(mask), len(clean))
    mask = mask[:n]

    freqs = np.fft.rfftfreq(NFFT, 1.0 / SAMPLE_RATE)

    speech_pow = (clean[mask] ** 2).mean() if mask.any() else EPS
    noise_pow = float((noise ** 2).mean())
    snr_db = float(_db(speech_pow) - _db(noise_pow))

    fr = _frames(x)
    frame_db = _db((fr ** 2).mean(axis=1))[:n]
    noise_floor_dbfs = float(np.percentile(frame_db, 10.0))
    speech_level_dbfs = float(frame_db[mask].mean()) if mask.any() else noise_floor_dbfs

    # --- quality, measured on the SPEECH estimate only ----------------------
    clipping_pct = float((np.abs(x) > 0.985).mean() * 100.0)

    speech_spec = clean[mask].mean(axis=0) if mask.any() else clean.mean(axis=0)
    cumulative = np.cumsum(speech_spec ** 2)
    total = cumulative[-1] + EPS
    bandwidth_hz = float(freqs[np.searchsorted(cumulative, 0.95 * total)])

    # Dropouts: frames inside speech regions that collapse toward the floor —
    # packet loss and gating artefacts, not pauses.
    # Measured against an ABSOLUTE floor. The previous version compared to the
    # clip's own noise floor, which additive noise raises — so it scored AUC
    # 0.84 for *noise_present*, a field it is unrelated to. It was measuring
    # noise, not dropouts.
    speech_span = np.where(mask)[0]
    if speech_span.size:
        lo, hi = speech_span[0], speech_span[-1] + 1
        interior = frame_db[lo:hi]
        dropout_pct = float((interior < DIGITAL_SILENCE_DBFS).mean() * 100.0)
    else:
        dropout_pct = 0.0

    # --- noise character, measured on the RESIDUAL only ---------------------
    res_mean = residual.mean(axis=0)
    res_pow = res_mean ** 2
    res_total = res_pow.sum() + EPS
    centroid = float((freqs * res_pow).sum() / res_total)
    flatness = float(np.exp(np.log(res_mean + EPS).mean()) / (res_mean.mean() + EPS))

    def band(lo: float, hi: float) -> float:
        sel = (freqs >= lo) & (freqs < hi)
        return float(res_pow[sel].sum() / res_total)

    bands = {
        "low_0_300": band(0, 300),
        "speech_300_3400": band(300, 3400),
        "high_3400_8000": band(3400, 8000),
    }

    # Speech-like sources (TV, background conversation) modulate their envelope
    # around 4 Hz — the syllable rate. Steady sources (hiss, hum, road) do not.
    res_env = residual.sum(axis=1)
    modulation = _band_energy_ratio(res_env, centre_hz=4.0, width_hz=3.0)

    # Harmonicity separates tonal sources (music, hum) from broadband ones.
    harmonicity = _harmonicity(res_mean)

    overlap_pct = _overlap_ratio(clean, mask)

    return Analysis(
        speech_ratio=speech_ratio,
        longest_silence_s=longest_silence,
        snr_db=snr_db,
        noise_floor_dbfs=noise_floor_dbfs,
        speech_level_dbfs=speech_level_dbfs,
        clipping_pct=clipping_pct,
        bandwidth_hz=bandwidth_hz,
        dropout_pct=dropout_pct,
        residual_centroid_hz=centroid,
        residual_flatness=flatness,
        residual_band_ratios=bands,
        residual_modulation_4hz=modulation,
        residual_harmonicity=harmonicity,
        overlap_frames_pct=overlap_pct,
    )


def _band_energy_ratio(env: np.ndarray, centre_hz: float, width_hz: float) -> float:
    """Fraction of envelope-modulation energy near `centre_hz`."""
    if env.size < 16:
        return 0.0
    env = env - env.mean()
    spec = np.abs(np.fft.rfft(env))
    rate = SAMPLE_RATE / HOP           # envelope sample rate (100 Hz)
    f = np.fft.rfftfreq(env.size, 1.0 / rate)
    sel = (f >= centre_hz - width_hz) & (f <= centre_hz + width_hz)
    return float(spec[sel].sum() / (spec.sum() + EPS))


def _harmonicity(mag: np.ndarray) -> float:
    """Peakiness of the spectrum: tonal sources concentrate energy in few bins."""
    if mag.size == 0:
        return 0.0
    top = np.sort(mag)[-8:].sum()
    return float(top / (mag.sum() + EPS))


def _overlap_ratio(clean: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of speech frames showing two concurrent harmonic sources.

    Heuristic, and honestly a weak one: a single talker produces one harmonic
    comb, so a second strong, non-multiple periodicity in the low-band cepstrum
    suggests a second voice. It will over-trigger on speech-like background
    (a television), which is exactly the failure mode `call_002` is expected to
    expose — see PLAN.md §9.
    """
    if not mask.any():
        return 0.0
    frames = clean[mask]
    if frames.shape[0] == 0:
        return 0.0

    log_spec = np.log(frames + EPS)
    ceps = np.fft.irfft(log_spec, axis=1)
    # Quefrency range covering 70–350 Hz F0.
    lo, hi = int(SAMPLE_RATE / 350), int(SAMPLE_RATE / 70)
    hi = min(hi, ceps.shape[1] - 1)
    if hi <= lo:
        return 0.0
    band = ceps[:, lo:hi]

    counts = 0
    for row in band:
        if row.size < 5:
            continue
        peak = row.max()
        if peak <= 0:
            continue
        # Peaks within 65% of the strongest, separated enough not to be the
        # same peak or its octave.
        strong = np.where(row > 0.65 * peak)[0]
        if strong.size < 2:
            continue
        if (strong.max() - strong.min()) > 12:
            counts += 1
    return float(counts / band.shape[0] * 100.0)
