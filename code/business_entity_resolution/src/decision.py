"""
Entity-level decision layer.

The model produces a calibrated score per (S1, candidate) pair; this
module turns the full score distribution for one S1 entity into a final
set of predicted matches. It is deliberately NOT `score > 0.5`:

  - threshold is chosen by grid search directly on macro F0.5 (the actual
    competition metric), using out-of-fold scores so it isn't fit on
    in-sample predictions
  - a secondary RELATIVE rule (score >= rel_margin * top_score) lets
    multiple genuinely strong candidates survive for the same S1 entity
    (multi-match is common in the audit: most entities have 2-4 matches)
    while still cutting off a long tail of mediocre candidates that
    happen to clear only the absolute floor
  - the singleton case fully drops out of this: if nothing clears the
    absolute threshold, the prediction is the empty list, which is
    exactly what a true singleton needs to score 1.0
"""
from __future__ import annotations
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd

from .config import DecisionConfig
from .evaluation import macro_micro_f05


def entity_decision(scores: List[Tuple[str, float]], abs_threshold: float,
                     rel_margin: float, max_matches: int) -> List[str]:
    """scores: list of (candidate_id, calibrated_score) for one S1 entity."""
    if not scores:
        return []
    scores = sorted(scores, key=lambda kv: -kv[1])
    top_score = scores[0][1]
    if top_score < abs_threshold:
        return []
    keep = []
    for cid, s in scores:
        if s >= abs_threshold and s >= rel_margin * top_score:
            keep.append(cid)
        if len(keep) >= max_matches:
            break
    return keep


def apply_decisions(feat_df: pd.DataFrame, score_col: str, abs_threshold: float,
                     rel_margin: float, cfg: DecisionConfig) -> Dict[str, Set[str]]:
    preds = {}
    for s1_id, grp in feat_df.groupby("source1_entity_id"):
        scores = list(zip(grp["candidate_id"], grp[score_col]))
        preds[s1_id] = set(entity_decision(scores, abs_threshold, rel_margin, cfg.max_matches_per_s1))
    return preds


def search_thresholds(feat_df: pd.DataFrame, score_col: str, truth: Dict[str, Set[str]],
                       cfg: DecisionConfig, rel_margins=None
                       ) -> Tuple[float, float, float]:
    """Grid search (abs_threshold, rel_margin) directly on macro F0.5.
    Returns (best_abs_threshold, best_rel_margin, best_macro_f05)."""
    best = (cfg.min_absolute_score, 0.0, -1.0)
    if rel_margins is None:
        rel_margins = getattr(cfg, "relative_margins", (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
    for rel_margin in rel_margins:
        for abs_t in cfg.threshold_grid:
            if abs_t < cfg.min_absolute_score:
                continue
            preds = apply_decisions(feat_df, score_col, abs_t, rel_margin, cfg)
            m = macro_micro_f05(preds, truth)
            if m["macro_f0.5"] > best[2]:
                best = (abs_t, rel_margin, m["macro_f0.5"])
    return best
