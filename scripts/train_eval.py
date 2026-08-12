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
import joblib  # noqa: E402
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
    # CARRIER CONTAMINATION: call_002 already contains TV noise and call_003
    # contains static, so a synthetic clip labelled "static at 12 dB" built on
    # call_002 actually contains TV + static. That is label noise across the
    # whole training set, and it is measurable: training on all three carriers
    # predicts "office chatter" for call_003's static (1/3), while training on
    # the one carrier labelled no-noise recovers it (2/3).
    #
    # Only call_001 is labelled background_noise_present=false, so it is the
    # only clean carrier available. Set VOICETRIAL_ALL_CARRIERS=1 to compare.
    import os

    from voicetrial.dsp import denoise

    paths = sorted(RAW.glob("*.ogg"))
    real = []
    for f in paths:
        samples = load(f).samples[: 20 * 16000]
        # Denoise every carrier. call_001 is already clean so this is close to a
        # no-op there; on call_002 and call_003 it removes the baked-in
        # television and static that would otherwise mislabel the training set.
        if not os.environ.get("VOICETRIAL_RAW_CARRIERS"):
            samples = denoise(samples)
        real.append(samples)
    print(f"carriers: {[p.stem for p in paths]} "
          f"({'denoised' if not os.environ.get('VOICETRIAL_RAW_CARRIERS') else 'RAW'})")
    globals()["real"] = real
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
        # ROOT carrier, not the resampled variant. pseudo_speakers is
        # carrier-major, so variant // per_carrier is the real recording. An
        # audit proved that holding out variants leaves every real voice on both
        # sides of the split: a probe identified the source recording of a
        # "held-out" sample at 0.922 (chance 0.333).
        groups.append((i % len(carriers)) // max(1, len(carriers) // len(real)))
    return np.nan_to_num(np.array(feats, dtype=float)), conds, np.array(groups)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    print(f"generating {n} synthetic clips (seed {RNG_SEED})…")
    X, conds, groups = build_dataset(n)

    # ROOT-CARRIER HOLDOUT is the only valid protocol here. A random split and a
    # variant-level "speaker" split both leave all three real recordings on both
    # sides, and both inflate every score. Reported numbers are the mean over
    # folds, each holding out one entire real recording.
    print(f"root-carrier folds: {sorted(set(groups.tolist()))}\n")

    MODEL_DIR.mkdir(exist_ok=True)
    summary = []

    for field, extract in TASKS.items():
        y = np.array([extract(c) for c in conds])
        # Majority-class baseline: the number every accuracy must be read
        # against. `noise_present` is 87% "yes" by construction, so 0.99 there is
        # a far smaller claim than it appears.
        vals, cnts = np.unique(y, return_counts=True)
        baseline = float(cnts.max() / cnts.sum())

        fold_acc, fold_f1, cms = [], [], []
        for g in sorted(set(groups.tolist())):
            tr, te = groups != g, groups == g
            if len(set(y[tr])) < 2 or te.sum() == 0:
                continue
            ytr, yte = y[tr], y[te]

            clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=6,
            random_state=RNG_SEED, early_stopping=False,
            # audio_quality classes are imbalanced (few "clear"); without this
            # the minority class is traded away for overall accuracy.
            class_weight="balanced",
        )
            clf.fit(X[tr], ytr)
            pred = clf.predict(X[te])
            fold_acc.append(accuracy_score(yte, pred))
            fold_f1.append(f1_score(yte, pred, average="macro"))
            cms.append(confusion_matrix(yte, pred, labels=sorted(set(y))))

        if not fold_acc:
            print(f"{field}: insufficient class coverage — skipped")
            continue
        acc, f1 = float(np.mean(fold_acc)), float(np.mean(fold_f1))
        labels = sorted(set(y))
        cm = sum(cms)

        print(f"=== {field}")
        print(f"    accuracy {acc:.3f}   macro-F1 {f1:.3f}   "
              f"baseline {baseline:.3f}   folds {[round(a, 3) for a in fold_acc]}")
        width = max(len(str(x)) for x in labels) + 1
        print("    " + " " * width + "pred:" + "".join(f"{str(x)[:10]:>12}" for x in labels))
        for i, lab in enumerate(labels):
            print(f"    {str(lab):<{width}}" + "     " + "".join(f"{v:>12}" for v in cm[i]))
        print()
        # Persist a model trained on ALL folds — the shipped artifact. The
        # reported metrics come from the held-out folds above, never from this.
        final = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=6,
            random_state=RNG_SEED, early_stopping=False, class_weight="balanced",
        )
        final.fit(X, y)
        joblib.dump({"model": final, "features": FEATURES,
                     "holdout_accuracy": acc, "baseline": baseline},
                    MODEL_DIR / f"{field}.joblib")
        summary.append((field, acc, f1, baseline))

    print("=" * 58)
    print(f"{'field':<30}{'accuracy':>10}{'macro-F1':>10}{'baseline':>10}{'lift':>8}")
    for field, acc, f1, base in summary:
        flag = "  <-- below 0.90" if acc < 0.90 else ""
        print(f"{field:<30}{acc:>10.3f}{f1:>10.3f}{base:>10.3f}{acc - base:>+8.3f}{flag}")
    print("\nRoot-carrier holdout: each fold trains without one entire real")
    print("recording. Accuracy must be read against the majority baseline.")


if __name__ == "__main__":
    main()
