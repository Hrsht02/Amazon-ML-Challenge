"""Pairwise scoring model with automatic GPU->CPU fallback."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


class PairClassifier:
    _gpu_disabled = False

    def __init__(self, feature_names, params: dict, seed: int = 42):
        self.feature_names = feature_names
        self.params = dict(params)
        self.params["random_state"] = seed
        self.model = None
        self.backend = "lightgbm" if HAS_LGBM else "logreg"
        self.device_used = "logreg" if not HAS_LGBM else "cpu"

    def _try_gpu(self) -> bool:
        if PairClassifier._gpu_disabled or self.backend != "lightgbm":
            return False
        if os.environ.get("ER_DISABLE_GPU", "0") == "1":
            return False
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight=None):
        X = X[self.feature_names].fillna(0.0)
        if self.backend == "lightgbm":
            base = dict(self.params)
            if self._try_gpu():
                for device in ("cuda", "gpu"):
                    gpu_params = dict(base)
                    gpu_params.pop("force_col_wise", None)
                    gpu_params["device_type"] = device
                    try:
                        print(f"[MODEL] Trying LightGBM device={device}...", flush=True)
                        self.model = lgb.LGBMClassifier(**gpu_params)
                        self.model.fit(X, y, sample_weight=sample_weight)
                        self.device_used = device
                        print(f"[MODEL] LightGBM device={device}", flush=True)
                        return self
                    except Exception as e:
                        print(f"[MODEL] device={device} failed -> trying next backend: {type(e).__name__}: {e}", flush=True)
                        self.model = None

                PairClassifier._gpu_disabled = True
                print("[MODEL] All GPU backends failed; disabling GPU for the rest of this run.", flush=True)

            base["device_type"] = "cpu"
            self.model = lgb.LGBMClassifier(**base)
            print("[MODEL] Training LightGBM on CPU", flush=True)
            self.model.fit(X, y, sample_weight=sample_weight)
            self.device_used = "cpu"
        else:
            print("[MODEL] LightGBM unavailable -> LogisticRegression fallback", flush=True)
            self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
            self.model.fit(X, y, sample_weight=sample_weight)
            self.device_used = "logreg"
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_names].fillna(0.0)
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self) -> pd.Series:
        if self.backend == "lightgbm":
            imp = self.model.booster_.feature_importance(importance_type="gain")
            return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)
        coef = np.abs(self.model.coef_[0])
        return pd.Series(coef, index=self.feature_names).sort_values(ascending=False)


class ScoreCalibrator:
    """Isotonic regression fit on out-of-fold raw scores."""
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.fitted = False

    def fit(self, raw_scores: np.ndarray, y: np.ndarray):
        if len(np.unique(y)) < 2:
            self.fitted = False
            return self
        self.iso.fit(raw_scores, y)
        self.fitted = True
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return raw_scores
        return self.iso.predict(raw_scores)
