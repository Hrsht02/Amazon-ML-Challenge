"""
Score calibration entry point. The actual isotonic-regression
implementation lives in models.ScoreCalibrator (kept next to
PairClassifier since they're fit together in training.py); this module
re-exports it under the name the package layout expects and adds
save/load helpers so a calibrator fit once during training can be reused
at inference time without retraining.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from .models import ScoreCalibrator  # noqa: F401  (re-exported)


def save_calibrator(calibrator: ScoreCalibrator, path: Path):
    with open(path, "wb") as f:
        pickle.dump(calibrator, f)


def load_calibrator(path: Path) -> ScoreCalibrator:
    with open(path, "rb") as f:
        return pickle.load(f)
