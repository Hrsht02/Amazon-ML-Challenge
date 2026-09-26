"""Leakage-safe grouped OOF training with fold-level checkpoints."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

from .config import ModelConfig
from .models import PairClassifier
from .features import get_feature_names, compute_pair_features
from .normalization import Normalizer


def build_positive_set(gt_df: pd.DataFrame):
    return {s1_id: set(row.match_list) for s1_id, row in gt_df.iterrows()}


def label_candidates(candidates_df: pd.DataFrame, positives):
    df = candidates_df.copy()
    df["label"] = [
        int(r.candidate_id in positives.get(r.source1_entity_id, set()))
        for r in df.itertuples(index=False)
    ]
    return df


def compute_features_for_candidates(candidates_df, s1_df, lut, normalizer: Normalizer):
    s1_name_cache, s1_addr_cache = {}, {}
    feat_rows = []
    for row in candidates_df.itertuples(index=False):
        s1_id, cand_id = row.source1_entity_id, row.candidate_id
        if s1_id not in s1_name_cache:
            s1_row = s1_df.loc[s1_id]
            s1_name_cache[s1_id] = normalizer.normalize_name(s1_row.business_name)
            s1_addr_cache[s1_id] = normalizer.normalize_address(s1_row.business_address)
        cand_row = lut[cand_id]
        feats = compute_pair_features(
            s1_name_cache[s1_id], s1_addr_cache[s1_id], s1_df.loc[s1_id].country,
            normalizer.normalize_name(cand_row.business_name),
            normalizer.normalize_address(cand_row.business_address),
            cand_row.country, n_strategies=getattr(row, "ann_n_strategies", 0)
        )
        for col in (
            "ann_similarity", "ann_ascii_name_similarity",
            "ann_unicode_name_similarity", "ann_address_similarity",
            "ann_name_rank", "ann_unicode_name_rank", "ann_address_rank"
        ):
            if hasattr(row, col):
                feats[col] = float(getattr(row, col))
        feats["source1_entity_id"] = s1_id
        feats["candidate_id"] = cand_id
        feat_rows.append(feats)
    return pd.DataFrame(feat_rows)


def _metric_print(y, scores, prefix):
    try:
        auc = roc_auc_score(y, scores)
        ap = average_precision_score(y, scores)
        ll = log_loss(y, np.clip(scores, 1e-7, 1-1e-7))
        print(f"[OOF] {prefix} ROC-AUC={auc:.6f} AP={ap:.6f} logloss={ll:.6f}", flush=True)
    except Exception as e:
        print(f"[OOF] {prefix} metrics unavailable: {e}", flush=True)


def grouped_oof_predictions(
    feat_df, y, cfg: ModelConfig, feature_names,
    sample_weight=None, checkpoint_dir=None, tag="base"
):
    groups = feat_df["source1_entity_id"].values
    n_groups = len(set(groups))
    n_splits = min(cfg.n_splits_cv, max(2, n_groups))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(feat_df), dtype=np.float32)
    sw = sample_weight if sample_weight is not None else np.ones(len(feat_df), dtype=np.float32)

    cp = Path(checkpoint_dir) if checkpoint_dir else None
    if cp:
        cp.mkdir(parents=True, exist_ok=True)

    splits = list(gkf.split(feat_df, y, groups))
    print(f"[OOF] {tag}: {n_splits} grouped folds, rows={len(feat_df):,}", flush=True)

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        fold_path = cp / f"{tag}_fold_{fold}.npy" if cp else None
        if fold_path and fold_path.exists():
            pred = np.load(fold_path)
            if len(pred) == len(va_idx):
                oof[va_idx] = pred.astype(np.float32)
                print(f"[OOF] {tag} fold {fold}/{n_splits}: RESUMED", flush=True)
                continue

        print(f"[OOF] {tag} fold {fold}/{n_splits}: train={len(tr_idx):,} valid={len(va_idx):,}", flush=True)
        clf = PairClassifier(feature_names, cfg.lgbm_params, seed=cfg.seed + fold)
        clf.fit(feat_df.iloc[tr_idx], y[tr_idx], sample_weight=sw[tr_idx])
        pred = clf.predict_proba(feat_df.iloc[va_idx]).astype(np.float32)
        oof[va_idx] = pred
        if fold_path:
            tmp = Path(str(fold_path) + ".tmp.npy")
            np.save(tmp, pred)
            tmp.replace(fold_path)
        _metric_print(y[va_idx], pred, f"{tag} fold {fold}")
        print(f"[OOF] {tag} fold {fold}/{n_splits}: saved checkpoint", flush=True)

    if cp:
        np.save(cp / f"{tag}_oof.npy", oof)
    _metric_print(y, oof, f"{tag} ALL")
    return oof


def hard_negative_reweight(feat_df, y, oof_scores, per_s1_top_n=8):
    weights = np.ones(len(feat_df), dtype=np.float32)
    df = feat_df[["source1_entity_id"]].copy()
    df["_y"] = y
    df["_score"] = oof_scores
    df["_idx"] = np.arange(len(df))
    for _, grp in df[df["_y"] == 0].groupby("source1_entity_id"):
        hard = grp.sort_values("_score", ascending=False).head(per_s1_top_n)
        weights[hard["_idx"].values] += 2.0
    return weights


def train_full_pipeline(candidates_df, s1_df, lut, gt_df, normalizer, cfg):
    positives = build_positive_set(gt_df)
    labeled = label_candidates(candidates_df, positives)
    feat_df = compute_features_for_candidates(
        labeled.drop(columns=["label"]), s1_df, lut, normalizer
    )
    feat_df["label"] = labeled["label"].values.astype(np.int8)
    y = feat_df["label"].to_numpy(dtype=np.int8)
    feature_names = [
        c for c in get_feature_names() + [
            "ann_similarity", "ann_ascii_name_similarity",
            "ann_unicode_name_similarity", "ann_address_similarity",
            "ann_name_rank", "ann_unicode_name_rank", "ann_address_rank"
        ] if c in feat_df.columns
    ]
    return feat_df, y, feature_names
