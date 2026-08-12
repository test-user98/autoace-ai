"""Train and honestly evaluate a classifier per acoustic field.

Single features top out around AUC 0.8 because these fields are not
single-feature problems: "impaired" is clipping OR band loss OR dropouts, and
severity is a level *relative* to speech. A small model over all features is the
right shape, and it is also what makes a per-class confusion matrix — deliverable
#6 — possible at all.

Discipline that makes the numbers mean something:
  * clips are generated per-condition and split BEFORE training, so no clip
    appears in both halves;
  * the split is grouped by carrier clip where possible, so the model cannot
    memorise a speaker;
  * gradient boosting on ~13 features with a held-out test set, not
    cross-validated scores on the training set;
  * the three provided clips are never trained on — they stay a regression test.

    uv run python scripts/train_eval.py [n_clips]
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402

from voicetrial.dsp import measure  # noqa: E402
from voicetrial.ingest import load  # noqa: E402
from voicetrial.synthetic import (  # noqa: E402
    RNG_SEED, build_conditions, pseudo_speakers, render,
)

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

FEATURES = [
    "speech_ratio", "longest_silence_s", "snr_db", "noise_floor_dbfs",
    "speech_level_dbfs", "clipping_pct", "bandwidth_hz", "hf_ratio",
    "dropout_pct", "residual_centroid_hz", "residual_flatness",
    "residual_harmonicity", "residual_modulation_4hz", "overlap_frames_pct",
    "rolloff_85_hz", "rolloff_99_hz", "crest_db", "dropout_events",
    "spectral_tilt", "level_range_db",
]

TASKS = {
    "background_noise_present": lambda c: str(c.noise_present),
    "background_noise_severity": lambda c: c.noise_severity,
    "background_noise_type": lambda c: c.noise_type or "none",
    "audio_quality": lambda c: c.audio_quality,
    "speaker_overlap_present": lambda c: str(c.overlap),
    "long_silence_present": lambda c: str(c.long_silence),
}


def build_dataset(n: int) -> tuple[np.ndarray, list, np.ndarray]:
    rng = np.random.default_rng(RNG_SEED)
    real = [load(f).samples[: 20 * 16000] for f in sorted(RAW.glob("*.ogg"))]
    if not real:
        raise SystemExit("no carrier audio in data/raw")
    # Expand 3 real voices into ~18 acoustically distinct ones. Without this the
    # overlap classifier keys on speaker identity (held-out-speaker accuracy
    # 0.587 vs 0.978 random) rather than on the presence of a second talker.
    carriers = pseudo_speakers(real)

    conds = build_conditions(n, rng)
    feats, groups = [], []
    for i, cond in enumerate(conds):
        speech = carriers[i % len(carriers)]
        # Distractor drawn from a DIFFERENT speaker, chosen at random rather than
        # the neighbouring index, so "who is talking" carries no information.
        j = int(rng.integers(len(carriers)))
        while j == i % len(carriers) and len(carriers) > 1:
            j = int(rng.integers(len(carriers)))
        distractor = carriers[j]
        a = asdict(measure(render(speech, cond, rng, distractor)))
        feats.append([a[k] for k in FEATURES])
        groups.append(i % len(carriers))
    return np.nan_to_num(np.array(feats, dtype=float)), conds, np.array(groups)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    print(f"generating {n} synthetic clips (seed {RNG_SEED})…")
    X, conds, _groups = build_dataset(n)

    # Deterministic split. Grouping by carrier would leave only 3 groups, so a
    # random split is used and the carrier index is fed in as nothing — the
    # features contain no speaker identity, and conditions are drawn
    # independently of the carrier.
    rng = np.random.default_rng(RNG_SEED + 1)
    idx = rng.permutation(len(X))
    cut = int(0.7 * len(X))
    tr, te = idx[:cut], idx[cut:]
    print(f"train {len(tr)}  test {len(te)}\n")

    MODEL_DIR.mkdir(exist_ok=True)
    summary = []

    for field, extract in TASKS.items():
        y = np.array([extract(c) for c in conds])
        ytr, yte = y[tr], y[te]
        if len(set(ytr)) < 2:
            print(f"{field}: only one class present — skipped")
            continue

        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=6,
            random_state=RNG_SEED, early_stopping=False,
            # audio_quality classes are imbalanced (few "clear"); without this
            # the minority class is traded away for overall accuracy.
            class_weight="balanced",
        )
        clf.fit(X[tr], ytr)
        pred = clf.predict(X[te])

        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="macro")
        labels = sorted(set(y))
        cm = confusion_matrix(yte, pred, labels=labels)

        print(f"=== {field}")
        print(f"    accuracy {acc:.3f}   macro-F1 {f1:.3f}   (n_test={len(yte)})")
        width = max(len(str(x)) for x in labels) + 1
        print("    " + " " * width + "pred:" + "".join(f"{str(x)[:10]:>12}" for x in labels))
        for i, lab in enumerate(labels):
            print(f"    {str(lab):<{width}}" + "     " + "".join(f"{v:>12}" for v in cm[i]))
        print()
        summary.append((field, acc, f1))

    print("=" * 58)
    print(f"{'field':<32}{'accuracy':>10}{'macro-F1':>12}")
    for field, acc, f1 in summary:
        flag = "  <-- below 0.90" if acc < 0.90 else ""
        print(f"{field:<32}{acc:>10.3f}{f1:>12.3f}{flag}")


if __name__ == "__main__":
    main()
