"""
Validation error analysis: for every false positive / false negative
produced by the current decision layer, print the S1 record, the
candidate record, the model's score, and a rough category so mistakes
can be traced back to a normalization/feature/blocking gap instead of
staying a single opaque "wrong" bucket.
"""
from __future__ import annotations
from typing import Dict, Set
import pandas as pd


def categorize(name_sim: float, addr_sim: float, country_mismatch: float,
               is_non_latin: float) -> str:
    if is_non_latin:
        return "transliteration/script_mismatch"
    if country_mismatch:
        return "country_mismatch"
    if name_sim > 0.85 and addr_sim < 0.3:
        return "name_high_addr_low (possible common-name collision or partial address)"
    if name_sim < 0.4 and addr_sim > 0.85:
        return "addr_high_name_low (possible DBA/trade-name or renamed business)"
    if 0.55 <= name_sim <= 0.85:
        return "moderate_name_similarity (typo/abbreviation/word-order candidate)"
    return "other"


def analyze_errors(feat_df: pd.DataFrame, predictions: Dict[str, Set[str]],
                    truth: Dict[str, Set[str]], s1_df: pd.DataFrame, lut,
                    top_n: int = 20) -> pd.DataFrame:
    rows = []
    for s1_id, true_set in truth.items():
        pred_set = predictions.get(s1_id, set())
        fps = pred_set - true_set
        fns = true_set - pred_set
        grp = feat_df[feat_df.source1_entity_id == s1_id].set_index("candidate_id")
        for cid in fps:
            r = grp.loc[cid] if cid in grp.index else None
            rows.append(_row(s1_id, cid, "false_positive", r, s1_df, lut))
        for cid in fns:
            r = grp.loc[cid] if cid in grp.index else None
            rows.append(_row(s1_id, cid, "false_negative", r, s1_df, lut))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("error_type").head(top_n * 2)


def _row(s1_id, cid, err_type, feat_row, s1_df, lut):
    s1_name = s1_df.loc[s1_id].business_name if s1_id in s1_df.index else ""
    s1_addr = s1_df.loc[s1_id].business_address if s1_id in s1_df.index else ""
    cand_name = lut[cid].business_name if cid in lut else "<not in candidate set / blocking miss>"
    cand_addr = lut[cid].business_address if cid in lut else ""
    if feat_row is not None:
        score = feat_row.get("calibrated_score", feat_row.get("oof_calibrated_score", float("nan")))
        name_sim = feat_row.get("name_token_sort_ratio", 0.0)
        addr_sim = feat_row.get("addr_token_sort_ratio", 0.0)
        country_mm = feat_row.get("country_mismatch", 0.0)
        non_latin = feat_row.get("name_either_non_latin", 0.0)
        category = categorize(name_sim, addr_sim, country_mm, non_latin)
    else:
        score, name_sim, addr_sim, category = float("nan"), None, None, "blocking_miss (never a candidate)"
    return {
        "error_type": err_type, "source1_entity_id": s1_id, "candidate_id": cid,
        "s1_name": s1_name, "s1_addr": s1_addr,
        "cand_name": cand_name, "cand_addr": cand_addr,
        "score": score, "category": category,
    }
