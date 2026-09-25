"""
Training pipeline.

Leakage safety: all cross-validation is GROUPED BY SOURCE-1 ENTITY
(sklearn GroupKFold on source1_entity_id). A random pair-level split
would let two candidate pairs for the *same* S1 entity land on opposite
sides of the split, leaking information about that entity's true match
pattern into validation — the brief calls this out explicitly and it's
easy to get wrong.

Positive/negative construction: rather than sampling negatives from the
full S1xS2xS3 cross product (which would mostly be trivially-easy
negatives that teach the model nothing), every candidate the blocking
stage proposes for an S1 entity is used as a training example: positive
if it's in that entity's ground-truth match list, negative otherwise.
Because blocking already selected these candidates for being
name/address-similar, this negative pool is a "hard negative" pool by
construction. On top of that we run explicit hard-negative re-weighting
rounds: fit a model, find the highest-scoring negatives, upweight them,
and refit — this sharpens the decision boundary on the pairs that are
actually confusable without inventing extra unfounded candidates that
would never appear in candidate_pairs.tsv.
"""
from __future__ import annotations
from typing import Dict, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import ModelConfig
from .models import PairClassifier, ScoreCalibrator
from .features import compute_pair_features, get_feature_names
from .normalization import Normalizer


def build_positive_set(gt_df: pd.DataFrame) -> Dict[str, Set[str]]:
    return {s1_id: set(row.match_list) for s1_id, row in gt_df.iterrows()}


def label_candidates(candidates_df: pd.DataFrame, positives: Dict[str, Set[str]]) -> pd.DataFrame:
    df = candidates_df.copy()
    df["label"] = df.apply(
        lambda r: int(r.candidate_id in positives.get(r.source1_entity_id, set())), axis=1)
    return df


def compute_features_for_candidates(candidates_df: pd.DataFrame,
                                     s1_df: pd.DataFrame, lut: Dict[str, pd.Series],
                                     normalizer: Normalizer) -> pd.DataFrame:
    """lut: entity_id -> row, covering both S2 and S3 (see data_loader.s2s3_lookup)."""
    # cache S1 normalizations
    s1_name_cache, s1_addr_cache = {}, {}
    feat_rows = []
    for row in candidates_df.itertuples(index=False):
        s1_id, cand_id = row.source1_entity_id, row.candidate_id
        if s1_id not in s1_name_cache:
            s1_row = s1_df.loc[s1_id]
            s1_name_cache[s1_id] = normalizer.normalize_name(s1_row.business_name)
            s1_addr_cache[s1_id] = normalizer.normalize_address(s1_row.business_address)
        n1, a1 = s1_name_cache[s1_id], s1_addr_cache[s1_id]
        c1_country = s1_df.loc[s1_id].country

        cand_row = lut[cand_id]
        n2 = normalizer.normalize_name(cand_row.business_name)
        a2 = normalizer.normalize_address(cand_row.business_address)

        feats = compute_pair_features(n1, a1, c1_country, n2, a2, cand_row.country,
                                       n_strategies=getattr(row, "n_strategies", 0))
        feats["source1_entity_id"] = s1_id
        feats["candidate_id"] = cand_id
        feat_rows.append(feats)
    return pd.DataFrame(feat_rows)


def grouped_oof_predictions(feat_df: pd.DataFrame, y: np.ndarray, cfg: ModelConfig,
                             feature_names, sample_weight=None) -> np.ndarray:
    """Out-of-fold raw model scores via GroupKFold on source1_entity_id, so
    calibration and threshold search never see in-sample scores."""
    groups = feat_df["source1_entity_id"].values
    n_groups = len(set(groups))
    n_splits = min(cfg.n_splits_cv, max(2, n_groups))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(feat_df))
    sw = sample_weight if sample_weight is not None else np.ones(len(feat_df))
    for tr_idx, va_idx in gkf.split(feat_df, y, groups):
        clf = PairClassifier(feature_names, cfg.lgbm_params, seed=cfg.seed)
        clf.fit(feat_df.iloc[tr_idx], y[tr_idx], sample_weight=sw[tr_idx])
        oof[va_idx] = clf.predict_proba(feat_df.iloc[va_idx])
    return oof


def hard_negative_reweight(feat_df: pd.DataFrame, y: np.ndarray, oof_scores: np.ndarray,
                            per_s1_top_n: int = 8) -> np.ndarray:
    """Upweight the highest-scoring NEGATIVES per S1 entity (the ones the
    current model is most confused about) for the next training round."""
    weights = np.ones(len(feat_df))
    df = feat_df.copy()
    df["_y"] = y
    df["_score"] = oof_scores
    df["_idx"] = np.arange(len(df))
    for s1_id, grp in df[df["_y"] == 0].groupby("source1_entity_id"):
        hard = grp.sort_values("_score", ascending=False).head(per_s1_top_n)
        weights[hard["_idx"].values] += 2.0  # triple weight vs. base 1.0
    return weights


def train_full_pipeline(candidates_df: pd.DataFrame, s1_df: pd.DataFrame,
                         lut: Dict[str, pd.Series], gt_df: pd.DataFrame,
                         normalizer: Normalizer, cfg: ModelConfig):
    """Returns (trained_classifier, calibrator, feature_df_with_oof, feature_names)."""
    positives = build_positive_set(gt_df)
    labeled = label_candidates(candidates_df, positives)
    feat_df = compute_features_for_candidates(labeled[["source1_entity_id", "candidate_id", "n_strategies"]],
                                               s1_df, lut, normalizer)
    feat_df = feat_df.merge(labeled[["source1_entity_id", "candidate_id", "label"]],
                             on=["source1_entity_id", "candidate_id"], how="left")
    y = feat_df["label"].values.astype(int)
    feature_names = [c for c in get_feature_names() if c in feat_df.columns]

    sample_weight = np.ones(len(feat_df))
    oof = grouped_oof_predictions(feat_df, y, cfg, feature_names, sample_weight)

    for _ in range(cfg.hard_negative_rounds):
        sample_weight = hard_negative_reweight(feat_df, y, oof, cfg.hard_negatives_per_s1)
        oof = grouped_oof_predictions(feat_df, y, cfg, feature_names, sample_weight)

    calibrator = ScoreCalibrator().fit(oof, y)
    feat_df["oof_raw_score"] = oof
    feat_df["oof_calibrated_score"] = calibrator.transform(oof)

    # final model trained on ALL data (with final sample weights) for inference use
    final_clf = PairClassifier(feature_names, cfg.lgbm_params, seed=cfg.seed)
    final_clf.fit(feat_df, y, sample_weight=sample_weight)

    return final_clf, calibrator, feat_df, feature_names
