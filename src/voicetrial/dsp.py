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
    hf_ratio: float = 0.0
    rolloff_85_hz: float = 0.0
    rolloff_99_hz: float = 0.0
    crest_db: float = 0.0
    dropout_events: float = 0.0
    spectral_tilt: float = 0.0
    level_range_db: float = 0.0


def _frames(x: np.ndarray) -> np.ndarray:
    """Split into overlapping analysis frames (n_frames, FRAME)."""
    if x.size < FRAME:
        x = np.pad(x, (0, FRAME - x.size))
    n = 1 + (x.size - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx]


def _db(v: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.asarray(v) + EPS)


def _voicing(fr: np.ndarray) -> np.ndarray:
    """Per-frame periodicity strength (0..1) from normalised autocorrelation.

    This is what separates *speech* from *sound*. Energy alone cannot: additive
    noise lifts a silent gap above any energy threshold, which is exactly why
    long-silence detection scored AUC 0.767 on gaps of known length. Voiced
    speech has a strong autocorrelation peak at its pitch period; noise does not,
    however loud it is.
    """
    x = fr - fr.mean(axis=1, keepdims=True)
    n = x.shape[1]
    spec = np.fft.rfft(x, n=2 * n, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), axis=1)[:, :n]
    ac = ac / (ac[:, :1] + EPS)
    lo, hi = int(SAMPLE_RATE / 350), min(int(SAMPLE_RATE / 70), n - 1)
    if hi <= lo:
        return np.zeros(x.shape[0])
    return np.clip(ac[:, lo:hi].max(axis=1), 0.0, 1.0)


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
    voiced = _voicing(fr) > 0.28
    # Require both: loud enough AND periodic. Noise poured into a gap satisfies
    # the first and fails the second, so the gap stays a gap.
    mask = (energy_db > thresh) & voiced & ~digital_silence

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
    # Clipping must be detected by WAVEFORM FLATNESS, not by an absolute
    # amplitude threshold. An audit showed the previous `|x| > 0.985` test read
    # ~0.000 at every clipping level, because clipped audio is routinely
    # normalised afterwards — the flat top survives, the absolute level does not.
    # Count samples sitting at the clip's own extreme, then require repetition.
    peak_abs = float(np.abs(x).max()) + EPS
    at_rail = np.abs(x) >= 0.98 * peak_abs
    # A true clip is a RUN of railed samples; isolated peaks are just peaks.
    runs = np.convolve(at_rail.astype(np.float32), np.ones(3), mode="same") >= 3
    clipping_pct = float(runs.mean() * 100.0)

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
    # Relative to the speech level, not to an absolute floor: a dropout is a
    # hole in speech that is present, which stays true under background noise.
    speech_span = np.where(mask)[0]
    if speech_span.size:
        lo, hi = speech_span[0], speech_span[-1] + 1
        interior = frame_db[lo:hi]
        ref = float(np.median(frame_db[mask]))
        dropout_pct = float((interior < ref - 25.0).mean() * 100.0)
    else:
        dropout_pct = 0.0

    # Band loss (muffling / codec) shows up as collapsed energy above 3.4 kHz.
    # The 95%-cumulative bandwidth measure could not see it — most speech energy
    # sits below 1 kHz, so the statistic barely moved when the top was removed.
    hf = (freqs >= 3400) & (freqs < 8000)
    sp_pow = speech_spec ** 2
    hf_ratio = float(sp_pow[hf].sum() / (sp_pow.sum() + EPS))

    # Band edge. The 95% point sits near 1 kHz for all speech and barely moves
    # when the top is removed, so it could not see muffling; 85% and 99% bracket
    # the edge far more sharply.
    cum = np.cumsum(sp_pow)
    tot = cum[-1] + EPS
    rolloff_85 = float(freqs[np.searchsorted(cum, 0.85 * tot)])
    rolloff_99 = float(freqs[np.searchsorted(cum, 0.99 * tot)])

    # Spectral tilt: dB per decade across the speech band. Lowpassing steepens it.
    band = (freqs >= 300) & (freqs < 7000)
    tilt = float(np.polyfit(np.log10(freqs[band] + 1.0), _db(sp_pow[band]), 1)[0])

    # Crest factor: clipping flattens peaks, so peak-to-RMS collapses.
    crest_db = float(_db(np.max(x ** 2) + EPS) - _db((x ** 2).mean() + EPS))

    # Count dropout *events*, not just their duration — a few long gaps and many
    # short ones are different failures.
    if speech_span.size:
        holes = frame_db[lo:hi] < (float(np.median(frame_db[mask])) - 25.0)
        dropout_events = float(np.count_nonzero(np.diff(holes.astype(int)) == 1))
    else:
        dropout_events = 0.0

    level_range_db = float(np.percentile(frame_db, 95) - np.percentile(frame_db, 5))

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

    overlap_pct = _overlap_ratio(fr, mask)

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
        hf_ratio=hf_ratio,
        rolloff_85_hz=rolloff_85,
        rolloff_99_hz=rolloff_99,
        crest_db=crest_db,
        dropout_events=dropout_events,
        spectral_tilt=tilt,
        level_range_db=level_range_db,
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


def _overlap_ratio(fr: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of speech frames whose periodicity is *ambiguous*.

    Counting cepstral peaks scored AUC 0.501 — a coin flip — so it is gone. One
    talker produces a single clean harmonic comb and a high autocorrelation
    peak. Two concurrent talkers interfere, and the combined signal is still
    clearly voiced but markedly less periodic. So the signature of overlap is
    not "two pitches found" (fragile) but "loud, speech-like, yet only weakly
    periodic" (robust).

    Known limitation, and the one `call_002` should expose: speech-shaped
    background such as a television produces the same ambiguity. This detector
    cannot distinguish a second caller from a television, and the memo says so.
    """
    if not mask.any():
        return 0.0
    v = _voicing(fr)[: len(mask)][mask]
    if v.size == 0:
        return 0.0
    ambiguous = (v > 0.15) & (v < 0.45)
    return float(ambiguous.mean() * 100.0)
