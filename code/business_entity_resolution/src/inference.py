"""
End-to-end inference: raw S1/S2/S3 dataframes -> candidate_pairs.tsv +
matching_results.tsv, using an already-trained classifier + calibrator +
chosen decision thresholds.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Set
import pandas as pd

from .blocking import SourceIndex, build_candidates
from .config import BlockingConfig, DecisionConfig
from .normalization import Normalizer
from .data_loader import s2s3_lookup
from .training import compute_features_for_candidates
from .decision import apply_decisions
from .models import PairClassifier, ScoreCalibrator


def write_id_list_tsv(path: Path, id_lists: Dict[str, Set[str]], id_order,
                       col1="source1_entity_id", col2="matched_entity_ids"):
    rows = []
    for s1_id in id_order:
        ids = sorted(id_lists.get(s1_id, set()))
        rows.append({col1: s1_id, col2: ",".join(ids)})
    df = pd.DataFrame(rows, columns=[col1, col2])
    df.to_csv(path, sep="\t", index=False)
    return df


def run_inference(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame,
                   normalizer: Normalizer, clf: PairClassifier, calibrator: ScoreCalibrator,
                   abs_threshold: float, rel_margin: float,
                   blocking_cfg: BlockingConfig, decision_cfg: DecisionConfig,
                   output_dir: Path):
    s2_index = SourceIndex(s2_df, normalizer, blocking_cfg)
    s3_index = SourceIndex(s3_df, normalizer, blocking_cfg)

    candidates_df = build_candidates(s1_df, s2_index, s3_index, normalizer, blocking_cfg)
    lut = s2s3_lookup(s2_df, s3_df)

    candidate_lists = {}
    for s1_id, grp in candidates_df.groupby("source1_entity_id"):
        candidate_lists[s1_id] = set(grp["candidate_id"])

    output_dir.mkdir(parents=True, exist_ok=True)
    write_id_list_tsv(output_dir / "candidate_pairs.tsv", candidate_lists, s1_df["entity_id"].tolist(),
                       col2="candidate_entity_ids")

    if candidates_df.empty:
        matched = {s1_id: set() for s1_id in s1_df["entity_id"]}
        write_id_list_tsv(output_dir / "matching_results.tsv", matched, s1_df["entity_id"].tolist())
        return candidates_df, matched

    feat_df = compute_features_for_candidates(
        candidates_df[["source1_entity_id", "candidate_id", "n_strategies"]], s1_df, lut, normalizer)

    raw_scores = clf.predict_proba(feat_df)
    feat_df["raw_score"] = raw_scores
    feat_df["calibrated_score"] = calibrator.transform(raw_scores)

    predictions = apply_decisions(feat_df, "calibrated_score", abs_threshold, rel_margin, decision_cfg)
    # ensure EVERY S1 entity has a row, including ones blocking found nothing for
    for s1_id in s1_df["entity_id"]:
        predictions.setdefault(s1_id, set())

    write_id_list_tsv(output_dir / "matching_results.tsv", predictions, s1_df["entity_id"].tolist())
    return feat_df, predictions
