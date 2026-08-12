# AutoAce Voice Trial — tone & background-noise analysis

Analyses call audio for background noise, audio quality, speaker overlap and dead
air, and serves it through a hosted dashboard that accepts a batch ZIP.

## Status — read this first

**Six of nine fields are implemented. `emotional_tone` and `emotional_intensity`
are NOT.** They return `neutral`/`low` as a visible placeholder rather than a
guess. Nothing in this repo predicts emotion today.

Accuracy on the three provided clips: **13/18 field-predictions correct (72%)**.

| field | engine | real-clip | synthetic holdout | baseline |
|---|---|---|---|---|
| `background_noise_type` | GBM over acoustic features | 1/3 | 0.956 | 0.166 |
| `background_noise_severity` | GBM | 1/3 | 0.858 | 0.317 |
| `background_noise_present` | GBM | 2/3 | 0.977 | 0.874 |
| `audio_quality` | GBM | 3/3 | 0.759 | 0.560 |
| `speaker_overlap_present` | pyannote/segmentation-3.0 | 3/3 | — | 0.586 |
| `long_silence_present` | pyannote/segmentation-3.0 | 3/3 | — | 0.517 |
| `emotional_tone` / `_intensity` | **not implemented** | — | — | — |

Synthetic figures use **root-carrier holdout** — each fold trains without one
entire real recording. Random splits and variant-level splits both leak and
inflate every score; see `STATE.md`. Read every accuracy against its baseline.

## Quick start

```bash
uv sync                                   # Python 3.12 is pinned
uv run pytest -q                          # 48 passed, 1 skipped

export HF_TOKEN=hf_...                    # needed for pyannote (gated repo)
uv run python scripts/train_eval.py 900   # trains + writes models/*.joblib
```

Analyse a batch folder (audio + `labels.csv` at its root):

```bash
uv run python -c "
from voicetrial.runner import run_batch
r = run_batch('data/raw'); print(r.to_csv())"
```

## Hosted dashboard

| | |
|---|---|
| URL | see `STATE.md` §6b |
| Login | `admin` / `admin` — **change before sharing** |
| Compute | Modal, scale-to-zero, CPU only |
| Storage | Vercel Blob, **private**; the batch is deleted after the run |

Upload a ZIP (audio at the root plus one CSV manifest with `name` and optional
`result_json`), watch progress, review results, download CSV or JSON.

## How it works

```
ffmpeg → 16 kHz mono
  ├── DSP features (numpy): VAD, spectral subtraction into speech + noise
  │      residual, 20 features → gradient-boosted model per noise/quality field
  └── pyannote/segmentation-3.0 → speaker map → overlap %, longest silence
```

Two design decisions worth knowing:

**Quality is measured on the speech estimate, noise on the residual.** They read
different arrays, so "do not infer background noise from poor audio quality"
holds structurally rather than by a threshold respecting it.

**pyannote handles overlap and silence because the feature models could not.**
Under root-carrier holdout they scored 0.670 and 0.730 — one fold hit 0.485,
worse than a constant predictor — because three real voices cannot teach speaker
independence. pyannote was trained on thousands.

## Cost and latency

Measured, not estimated: **11× realtime** end-to-end with pyannote (21.3 s for
237.8 s of audio), **160× realtime** for the DSP path alone.

At a $0.15/hr CPU instance that is **~$0.0004 per audio-minute — about 13% of the
$0.003 ceiling**, with no GPU anywhere. Tone is unbuilt, so this figure does not
yet include it.

## Reproducing the evaluation

```bash
uv run python scripts/calibrate.py 160    # per-feature separability (AUC)
uv run python scripts/train_eval.py 900   # per-field accuracy + confusion matrix
cd app && node scripts/e2e_check.mjs <batch.zip>   # drives the DEPLOYED system
```

`scripts/train_eval.py` prints a confusion matrix and a majority baseline per
field. Seeds are fixed; runs are reproducible.

## Known limitations

- **Tone is not implemented.** Two of nine fields are placeholders.
- **3.96 minutes of real audio exists**, and it is both the carrier for the
  synthetic set and the only real test data. Synthetic numbers measure whether
  the detectors work in principle; they do not predict hidden-set accuracy.
- **Label vocabulary differs from the real labels** — the model says
  `office chatter` where `labels.csv` says `TV`, and `static` vs `sharp static`.
- **Two thresholds were set on n=3** (`OVERLAP_FRAME_PCT`, `LONG_SILENCE_S`) and
  are the first things to re-fit given more labelled audio.
- Natural conversational pauses are absent from the synthetic generator, so it
  cannot see silence failures at any sample size.

`STATE.md` carries the full timestamped history, including three evaluation
protocols that were wrong and how each was caught.
