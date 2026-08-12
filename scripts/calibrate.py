"""Build the synthetic eval set, measure it, and report feature separability.

This is the tool that decides thresholds. Fitting them on the three provided
clips would be overfitting to n=3; here every label is exactly what was
synthesised, so separability is a fact rather than an impression.

    uv run python scripts/calibrate.py [n_clips]
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voicetrial.dsp import measure  # noqa: E402
from voicetrial.ingest import load  # noqa: E402
from voicetrial.synthetic import RNG_SEED, build_conditions, render  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FEATURES = [
    "speech_ratio", "longest_silence_s", "snr_db", "noise_floor_dbfs",
    "speech_level_dbfs", "clipping_pct", "bandwidth_hz", "dropout_pct",
    "residual_centroid_hz", "residual_flatness", "residual_harmonicity",
    "residual_modulation_4hz", "overlap_frames_pct",
]


def separability(values: np.ndarray, labels: np.ndarray) -> float:
    """AUC of a single feature against a binary label — 0.5 is useless, 1.0 is
    perfect. Rank-based, so it is threshold-free and scale-free."""
    pos, neg = values[labels], values[~labels]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, values.size + 1)
    auc = (ranks[labels].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)
    return float(max(auc, 1 - auc))  # direction-agnostic


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 140
    rng = np.random.default_rng(RNG_SEED)

    carriers = [load(RAW / f).samples for f in sorted(RAW.glob("*.ogg"))]
    if not carriers:
        raise SystemExit("no carrier audio in data/raw")

    conds = build_conditions(n, rng)
    rows, truth = [], []
    for i, cond in enumerate(conds):
        speech = carriers[i % len(carriers)]
        # Cap length so 140 clips stay quick; conditions are unaffected.
        speech = speech[: 20 * 16000]
        distractor = carriers[(i + 1) % len(carriers)][: 20 * 16000]
        audio = render(speech, cond, rng, distractor)
        rows.append(asdict(measure(audio)))
        truth.append(cond)

    print(f"\nsynthetic clips: {len(rows)}  (seed {RNG_SEED})\n")

    tasks = {
        "noise_present": np.array([c.noise_present for c in truth]),
        "noise_high_sev": np.array([c.noise_severity == "high" for c in truth]),
        "quality_impaired": np.array([c.audio_quality != "clear" for c in truth]),
        "quality_severe": np.array([c.audio_quality == "severely_impaired" for c in truth]),
        "overlap": np.array([c.overlap for c in truth]),
        "long_silence": np.array([c.long_silence for c in truth]),
    }

    print(f"{'feature':<26}" + "".join(f"{t:>18}" for t in tasks))
    print("-" * (26 + 18 * len(tasks)))
    for feat in FEATURES:
        vals = np.array([r[feat] for r in rows], dtype=float)
        if not np.isfinite(vals).all():
            vals = np.nan_to_num(vals)
        line = f"{feat:<26}"
        for label in tasks.values():
            line += f"{separability(vals, label):>18.3f}"
        print(line)

    print("\nclass balance:")
    for name, label in tasks.items():
        print(f"  {name:<18} {int(label.sum()):>4} / {label.size}")

    print("\nAUC >= 0.80 is a usable feature; ~0.50 is noise.")


if __name__ == "__main__":
    main()
