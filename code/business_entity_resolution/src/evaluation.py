"""
Exact implementation of the competition metric (macro F0.5 over Source 1
entities) plus the auxiliary metrics called for in the brief: micro
P/R/F0.5, singleton accuracy / false-positive rate, candidate recall and
reduction ratio.
"""
from __future__ import annotations
from typing import Dict, List, Set
import math


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    denom = (b2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def entity_prf(pred: Set[str], truth: Set[str]) -> Dict[str, float]:
    """Per-Source-1-entity precision/recall/F0.5, matching the challenge's
    stated singleton rule exactly:
      - true singleton (truth empty), predicted empty  -> F0.5 = 1.0
      - true singleton, predicted non-empty             -> F0.5 = 0.0
      - non-singleton, predicted empty                  -> F0.5 = 0.0 (recall 0)
    """
    if not truth and not pred:
        return {"precision": 1.0, "recall": 1.0, "f0.5": 1.0}
    if not truth and pred:
        return {"precision": 0.0, "recall": 0.0, "f0.5": 0.0}
    if truth and not pred:
        return {"precision": 0.0, "recall": 0.0, "f0.5": 0.0}
    tp = len(pred & truth)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(truth) if truth else 0.0
    return {"precision": precision, "recall": recall, "f0.5": f_beta(precision, recall, 0.5)}


def macro_micro_f05(predictions: Dict[str, Set[str]], truth: Dict[str, Set[str]]) -> Dict[str, float]:
    """predictions / truth: {source1_entity_id: set(candidate_ids)}.
    Every key in `truth` must be scored (missing predictions treated as empty set)."""
    per_entity = {}
    tp_total = fp_total = fn_total = 0
    for s1_id, true_set in truth.items():
        pred_set = predictions.get(s1_id, set())
        stats = entity_prf(pred_set, true_set)
        per_entity[s1_id] = stats
        tp_total += len(pred_set & true_set)
        fp_total += len(pred_set - true_set)
        fn_total += len(true_set - pred_set)

    n = len(per_entity)
    macro_p = sum(v["precision"] for v in per_entity.values()) / n if n else 0.0
    macro_r = sum(v["recall"] for v in per_entity.values()) / n if n else 0.0
    macro_f05 = sum(v["f0.5"] for v in per_entity.values()) / n if n else 0.0

    micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    micro_f05 = f_beta(micro_p, micro_r, 0.5)

    singleton_ids = [s1 for s1, t in truth.items() if not t]
    non_singleton_ids = [s1 for s1, t in truth.items() if t]
    singleton_acc = (sum(1 for s1 in singleton_ids if not predictions.get(s1, set())) / len(singleton_ids)
                      if singleton_ids else float("nan"))
    singleton_fp_rate = (sum(1 for s1 in singleton_ids if predictions.get(s1, set())) / len(singleton_ids)
                          if singleton_ids else float("nan"))

    return {
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f0.5": macro_f05,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f0.5": micro_f05,
        "n_entities": n,
        "n_singletons": len(singleton_ids),
        "singleton_accuracy": singleton_acc,
        "singleton_false_positive_rate": singleton_fp_rate,
        "n_non_singletons": len(non_singleton_ids),
        "per_entity": per_entity,
    }


def candidate_recall(candidates: Dict[str, Set[str]], truth: Dict[str, Set[str]]) -> Dict[str, float]:
    """Recall ceiling imposed by blocking: fraction of true positive pairs
    that appear anywhere in the candidate set."""
    total_pos = 0
    recovered = 0
    n_candidate_pairs = 0
    for s1_id, true_set in truth.items():
        cand = candidates.get(s1_id, set())
        n_candidate_pairs += len(cand)
        total_pos += len(true_set)
        recovered += len(true_set & cand)
    recall = recovered / total_pos if total_pos else float("nan")
    return {
        "candidate_recall": recall,
        "total_positive_pairs": total_pos,
        "recovered_positive_pairs": recovered,
        "total_candidate_pairs": n_candidate_pairs,
        "avg_candidates_per_s1": n_candidate_pairs / len(truth) if truth else 0.0,
    }
