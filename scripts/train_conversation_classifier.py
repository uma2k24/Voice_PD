"""Train a CPU acoustic baseline from an explicit speaker/age/task manifest.

Example: python -m scripts.train_conversation_classifier --manifest data/manifest.csv
Paths are relative to the manifest, labels are HC=0/PD=1, speaker IDs must be
globally unique. No inferred diagnosis, age, task or speaker identity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import parselmouth
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.conversation_acoustics import (extract_acoustics, SAMPLE_RATE,
                                       EXTRACTOR_VERSION, FEATURE_NAMES, SETTINGS)
from scipy.signal import resample_poly
import math

REQUIRED = ["path", "speaker_id", "label", "age", "sex", "task"]


def read_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"speaker_id": str})
    if not set(REQUIRED).issubset(frame.columns) or frame[REQUIRED].isna().any().any():
        raise ValueError(f"Manifest requires nonempty columns: {REQUIRED}")
    if any(frame[key].astype(str).str.strip().eq("").any() for key in ("path", "speaker_id", "sex", "task")):
        raise ValueError("Manifest contains empty identifiers or tasks.")
    if not frame.label.isin([0, 1]).all():
        raise ValueError("Labels must be HC=0, PD=1.")
    if not frame.task.isin(["conversation", "reading", "sustained_vowel"]).all():
        raise ValueError("Task must be conversation, reading, or sustained_vowel.")
    ages = pd.to_numeric(frame.age, errors="raise")
    if not ages.between(0, 120).all() or not (ages % 1 == 0).all():
        raise ValueError("Age must be an integer in 0-120.")
    for column in ("label", "age", "sex"):
        if (frame.groupby("speaker_id")[column].nunique() > 1).any():
            raise ValueError(f"Inconsistent {column} for the same speaker.")
    frame["age"] = ages
    frame["path"] = frame.path.map(lambda p: str((path.parent / p).resolve()))
    if frame.path.duplicated().any():
        raise ValueError("Duplicate recording paths.")
    frame = frame[frame.age >= 50].copy()
    if frame.empty:
        raise ValueError("No recordings in the age-50 cohort.")
    return frame


def extract_recording(args):
    recording, max_windows = args
    rows, rejected = [], []
    try:
        y, rate = sf.read(recording["path"], always_2d=True)
        y = y.mean(axis=1)
        divisor = math.gcd(rate, SAMPLE_RATE)
        y = resample_poly(y, SAMPLE_RATE // divisor, rate // divisor)
        window = 4 * SAMPLE_RATE
        # Same four-second windows as serving; all remain in the speaker's fold.
        offsets = list(range(0, max(1, y.size - window + 1), SAMPLE_RATE))
        if len(offsets) > max_windows:
            offsets = [offsets[i] for i in np.linspace(0, len(offsets) - 1, max_windows, dtype=int)]
        for start in offsets:
            measurement = extract_acoustics(y[start:start + window])
            if not measurement["quality"]["usable"] or y[start:start + window].size < window:
                rejected.append({**recording, "offset_seconds": start / SAMPLE_RATE,
                                 "reason": ",".join(measurement["quality"]["reasons"]) or "under_four_seconds"})
                continue
            rows.append({**recording, "offset_seconds": start / SAMPLE_RATE,
                         **measurement["features"]})
    except (OSError, RuntimeError, ValueError) as exc:
        rejected.append({**recording, "reason": f"{type(exc).__name__}: {exc}"})
    return rows, rejected


def extract_manifest(frame: pd.DataFrame, workers=1, max_windows=3) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, rejected = [], []
    arguments = [(recording, max_windows) for recording in frame.to_dict("records")]
    def collect(results):
        for index, (accepted, failed) in enumerate(results, 1):
            rows.extend(accepted)
            rejected.extend(failed)
            if index % 10 == 0 or index == len(arguments):
                print(f"Extracted {index}/{len(arguments)} recordings; {len(rows)} usable windows.", flush=True)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            collect(pool.map(extract_recording, arguments))
    else:
        collect(map(extract_recording, arguments))
    if not rows:
        raise ValueError("No usable four-second speech windows.")
    return pd.DataFrame(rows), pd.DataFrame(rejected)


def make_classifier(name: str):
    estimator = (LogisticRegression(C=1, class_weight="balanced", max_iter=2000, random_state=42)
                 if name == "logreg" else RandomForestClassifier(
                     n_estimators=200, max_depth=5, min_samples_leaf=5,
                     class_weight="balanced", random_state=42, n_jobs=1))
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), estimator)


def speaker_metrics(frame, scores):
    scored = frame[["speaker_id", "label"]].copy()
    scored["score_pd"] = scores
    speakers = scored.groupby("speaker_id", as_index=False).agg(label=("label", "first"), score_pd=("score_pd", "mean"))
    metrics = {"speaker_auc": float(roc_auc_score(speakers.label, speakers.score_pd)),
               "speaker_balanced_accuracy_at_0_5": float(balanced_accuracy_score(speakers.label, speakers.score_pd >= .5)),
               "speaker_brier": float(brier_score_loss(speakers.label, speakers.score_pd)),
               "n_speakers": len(speakers)}
    return metrics, speakers


def fit_baseline(frame, classifier="logreg", folds=5, loso=False):
    X, y, groups = frame[FEATURE_NAMES].astype(float), frame.label.astype(int), frame.speaker_id
    subjects = frame.groupby("speaker_id").label.first()
    if subjects.nunique() != 2 or subjects.value_counts().min() < (2 if loso else folds):
        raise ValueError("Too few speakers per class for the requested speaker-separated CV.")
    splitter = LeaveOneGroupOut() if loso else StratifiedGroupKFold(folds, shuffle=True, random_state=42)
    pipeline = make_classifier(classifier)
    scores = np.full(len(frame), np.nan)
    fold_speakers = []
    # All imputation/scaling/model fitting happens inside the training fold.
    # No hyperparameter or threshold tuning on these held-out scores.
    for train, test in splitter.split(X, y, groups):
        if y.iloc[train].nunique() != 2:
            raise ValueError("A training fold has only one class.")
        model = clone(pipeline).fit(X.iloc[train], y.iloc[train])
        scores[test] = model.predict_proba(X.iloc[test])[:, 1]
        fold_speakers.append({"train": sorted(set(groups.iloc[train])), "test": sorted(set(groups.iloc[test]))})
    metrics, speakers = speaker_metrics(frame, scores)
    pipeline.fit(X, y)
    return pipeline, metrics, speakers, fold_speakers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models_conversation"))
    parser.add_argument("--classifier", choices=["logreg", "random_forest"], default="logreg")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--loso", action="store_true")
    parser.add_argument("--external-manifest", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-windows-per-recording", type=int, default=3,
                        help="Uniformly sample at most this many four-second windows per recording.")
    args = parser.parse_args()
    if args.workers < 1 or args.max_windows_per_recording < 1 or args.folds < 2:
        parser.error("workers/windows must be positive and folds must be at least 2")
    manifest = read_manifest(args.manifest)
    external = read_manifest(args.external_manifest) if args.external_manifest else None
    if external is not None:
        if set(manifest.speaker_id) & set(external.speaker_id):
            raise ValueError("External speakers overlap training speakers.")
        if not set(external.task).issubset(set(manifest.task)):
            raise ValueError("External recording tasks differ from training tasks.")
        if set(manifest.path) & set(external.path):
            raise ValueError("External recordings overlap training recordings.")
    frame, rejected = extract_manifest(manifest, args.workers, args.max_windows_per_recording)
    pipeline, metrics, speakers, splits = fit_baseline(frame, args.classifier, args.folds, args.loso)
    report = {"extractor_version": EXTRACTOR_VERSION, "features": FEATURE_NAMES,
              "settings": SETTINGS, "minimum_age": 50, "tasks": sorted(frame.task.unique()),
              "classifier": args.classifier, "cv": "LOSO" if args.loso else "StratifiedGroupKFold",
              "threshold": .5, "speaker_metrics": metrics,
              "n_windows": len(frame), "n_rejected_windows": len(rejected),
              "max_windows_per_recording": args.max_windows_per_recording,
              "versions": {"scikit_learn": sklearn.__version__, "praat_parselmouth": parselmouth.__version__},
              "n_eligible_recordings": len(manifest),
              "n_usable_recordings": int(frame.path.nunique()),
              "limitations": ["Research prototype; score is not a personal disease probability.",
                              "Task coverage is limited to the supplied training recordings.",
                              "No external validation unless external_metrics is present."]}
    cohort = frame.drop_duplicates("speaker_id")
    report["cohort"] = {str(label): {"n_speakers": int(len(group)),
        "age_min": int(group.age.min()), "age_max": int(group.age.max()),
        "sex_counts": {str(k): int(v) for k, v in group.sex.value_counts().items()}}
        for label, group in cohort.groupby("label")}
    if external is not None:
        external_frame, _ = extract_manifest(external, args.workers, args.max_windows_per_recording)
        if not set(external_frame.task).issubset(set(frame.task)):
            raise ValueError("No usable training coverage for an external task.")
        report["external_metrics"], external_speakers = speaker_metrics(
            external_frame, pipeline.predict_proba(external_frame[FEATURE_NAMES].astype(float))[:, 1])
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.output / "classifier.joblib")
    (args.output / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "fold_speakers.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    speakers.to_csv(args.output / "oof_speaker_scores.csv", index=False)
    frame.to_csv(args.output / "training_windows.csv", index=False)
    rejected.to_csv(args.output / "rejected_windows.csv", index=False)
    if external is not None:
        external_speakers.to_csv(args.output / "external_speaker_scores.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
