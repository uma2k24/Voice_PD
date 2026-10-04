"""Versioned acoustic classifier contract; never manufacture a PD probability."""
from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .conversation_acoustics import EXTRACTOR_VERSION, FEATURE_NAMES, SETTINGS


class AcousticClassifier:
    def __init__(self, model_dir: str | Path):
        self.pipeline = None
        self.metadata = {}
        self.error = None
        directory = Path(model_dir)
        if not (directory / "classifier.joblib").exists():
            self.error = "Train the conversation acoustic baseline first."
            return
        try:
            self.metadata = json.loads((directory / "metadata.json").read_text())
            if self.metadata.get("extractor_version") != EXTRACTOR_VERSION:
                raise ValueError("Extractor version mismatch.")
            if self.metadata.get("features") != FEATURE_NAMES:
                raise ValueError("Feature schema mismatch.")
            if self.metadata.get("settings") != SETTINGS:
                raise ValueError("Acoustic settings mismatch.")
            if self.metadata.get("minimum_age") != 50:
                raise ValueError("Model must use the age-50 cohort.")
            if not self.metadata.get("tasks"):
                raise ValueError("Model has no recording task coverage.")
            # Load only local, trusted training artifacts (joblib is executable).
            self.pipeline = joblib.load(directory / "classifier.joblib")
            if list(self.pipeline.classes_) != [0, 1]:
                raise ValueError("Classifier classes must be HC=0 and PD=1.")
        except Exception as exc:
            self.pipeline = None
            self.error = f"Invalid classifier artifact: {type(exc).__name__}: {exc}"

    def predict(self, measurements: dict, age: int, task: str) -> dict:
        result = {"score_pd": None, "status": "unavailable",
                  "interpretation": "Experimental classifier score, not a diagnosis or personal disease probability.",
                  "model_features": FEATURE_NAMES}
        if age < 50:
            return {**result, "status": "outside_evaluated_population"}
        if not measurements["quality"]["usable"]:
            return {**result, "status": "insufficient_audio_quality"}
        if self.pipeline is None:
            return {**result, "reason": self.error}
        if task not in self.metadata["tasks"]:
            return {**result, "status": "unsupported_recording_task",
                    "supported_tasks": self.metadata["tasks"]}
        row = pd.DataFrame([[measurements["features"].get(k) for k in FEATURE_NAMES]],
                           columns=FEATURE_NAMES, dtype=float)
        probability = float(self.pipeline.predict_proba(row)[0, 1])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Classifier returned an invalid score.")
        return {**result, "status": "experimental", "score_pd": probability,
                "classifier": self.metadata.get("classifier"),
                "training_tasks": self.metadata["tasks"]}
