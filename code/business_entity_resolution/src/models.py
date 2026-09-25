"""
Pairwise scoring model.

LightGBM (MIT-licensed, gradient-boosted trees, nowhere near an 8B
"parameter" budget in any meaningful sense) is the primary model — the
brief expects tree ensembles to be strong here because the input is
structured similarity features, not raw text, and validates that
assumption via the ablation script rather than asserting it blindly.
A logistic-regression baseline is included as the sanity-check /
ablation floor.

Calibration (isotonic regression) is applied on top of the raw model
score so the entity-level decision layer (decision.py) can reason about
scores as approximate probabilities, and so thresholds tuned on one
country/source distribution transfer better to another (e.g. the
train-unseen France rows at test time).
"""
from __future__ import annotations
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
    def __init__(self, feature_names, params: dict, seed: int = 42):
        self.feature_names = feature_names
        self.params = dict(params)
        self.params["random_state"] = seed
        self.model = None
        self.backend = "lightgbm" if HAS_LGBM else "logreg"

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight=None):
        X = X[self.feature_names].fillna(0.0)
        if self.backend == "lightgbm":
            self.model = lgb.LGBMClassifier(**self.params)
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
            self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_names].fillna(0.0)
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self) -> pd.Series:
        if self.backend == "lightgbm":
            imp = self.model.booster_.feature_importance(importance_type="gain")
            return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)
        else:
            coef = np.abs(self.model.coef_[0])
            return pd.Series(coef, index=self.feature_names).sort_values(ascending=False)


class ScoreCalibrator:
    """Isotonic regression, fit on out-of-fold raw scores so it doesn't
    just memorize the training scores' own distribution."""

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
