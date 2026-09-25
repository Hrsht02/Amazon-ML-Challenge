"""
End-to-end CLI.

    python -m src.generate_submission --stage all

runs: data audit -> blocking -> candidate recall report -> training
(with grouped CV + hard-negative rounds) -> threshold search -> ablation
table -> error analysis -> test inference -> submission files ->
local format validation.

Each stage can also be run independently (see --stage).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PATHS, BLOCKING, MODEL, DECISION
from src.data_loader import load_split, s2s3_lookup
from src.preprocessing import fit_normalizer, quick_audit
from src.blocking import SourceIndex, build_candidates
from src.training import train_full_pipeline, build_positive_set
from src.evaluation import macro_micro_f05, candidate_recall
from src.decision import search_thresholds, apply_decisions
from src.error_analysis import analyze_errors
from src.inference import run_inference
from src.calibration import save_calibrator, load_calibrator
import pickle


def build_train_candidates(train, normalizer):
    s2_index = SourceIndex(train["s2"], normalizer, BLOCKING)
    s3_index = SourceIndex(train["s3"], normalizer, BLOCKING)
    candidates_df = build_candidates(train["s1"], s2_index, s3_index, normalizer, BLOCKING)
    return candidates_df


def run_ablation(train, normalizer, candidates_df, feat_df, truth):
    """Lightweight ablation comparing rule-based baselines against the
    trained pipeline, all evaluated with the SAME candidate set / grouped
    logic so the comparison is fair."""
    from src.decision import entity_decision
    rows = []

    cand_recall = candidate_recall(
        {s1: set(g.candidate_id) for s1, g in candidates_df.groupby("source1_entity_id")}, truth)

    def eval_rule(score_col, label):
        best_f = -1
        best_t = None
        for t in [i / 100 for i in range(30, 100, 2)]:
            preds = {}
            for s1_id, grp in feat_df.groupby("source1_entity_id"):
                scores = list(zip(grp.candidate_id, grp[score_col]))
                preds[s1_id] = set(entity_decision(scores, t, 0.0, DECISION.max_matches_per_s1))
            m = macro_micro_f05(preds, truth)
            if m["macro_f0.5"] > best_f:
                best_f, best_t = m["macro_f0.5"], t
        rows.append({"method": label, "candidate_recall": round(cand_recall["candidate_recall"], 3),
                      "best_threshold": best_t, "macro_f0.5": round(best_f, 3)})

    # A. exact normalized name+address match only
    feat_df["_exact_rule"] = ((feat_df["name_exact_alnum"] == 1) | (feat_df["addr_exact_alnum"] == 1)).astype(float)
    eval_rule("_exact_rule", "A: exact name-or-address match")

    # B. fuzzy name only (token_sort_ratio)
    eval_rule("name_token_sort_ratio", "B: fuzzy name similarity only")

    # C. fuzzy name + address average
    feat_df["_name_addr_avg"] = 0.5 * (feat_df["name_token_sort_ratio"] + feat_df["addr_token_sort_ratio"])
    eval_rule("_name_addr_avg", "C: fuzzy name+address average")

    # D. trained model, OOF (uncalibrated)
    eval_rule("oof_raw_score", "D: trained model (raw, uncalibrated)")

    # E. trained model, OOF calibrated + relative-margin decision search (the real pipeline)
    abs_t, rel_m, f05 = search_thresholds(feat_df, "oof_calibrated_score", truth, DECISION)
    preds = apply_decisions(feat_df, "oof_calibrated_score", abs_t, rel_m, DECISION)
    m = macro_micro_f05(preds, truth)
    rows.append({"method": "E: full pipeline (calibrated + entity-level decision)",
                  "candidate_recall": round(cand_recall["candidate_recall"], 3),
                  "best_threshold": f"abs={abs_t}, rel_margin={rel_m}",
                  "macro_f0.5": round(m["macro_f0.5"], 3)})
    return pd.DataFrame(rows), (abs_t, rel_m, f05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                     choices=["all", "audit", "train", "predict", "validate"])
    ap.add_argument("--dataset-dir", default=str(PATHS.dataset_dir))
    ap.add_argument("--output-dir", default=str(PATHS.output_dir))
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    models_dir = PATHS.models_dir

    print("=" * 70)
    print("STAGE: DATA LOAD + AUDIT")
    print("=" * 70)
    train = load_split(dataset_dir, "train")
    test = load_split(dataset_dir, "test")
    normalizer = fit_normalizer(train["s1"], train["s2"], train["s3"])
    audit = quick_audit(train["s1"], train["s2"], train["s3"], train["gt"])
    print(json.dumps(audit, indent=2, default=str))

    test_audit = quick_audit(test["s1"], test["s2"], test["s3"])
    print("\nTest-set country distribution (note: may include countries unseen "
          "in training, e.g. France — must NOT be filtered):")
    print(json.dumps(test_audit["country_dist_s1"], indent=2))

    if args.stage == "audit":
        return

    print("\n" + "=" * 70)
    print("STAGE: BLOCKING (train)")
    print("=" * 70)
    candidates_df = build_train_candidates(train, normalizer)
    truth = build_positive_set(train["gt"])
    cand_sets = {s1: set(g.candidate_id) for s1, g in candidates_df.groupby("source1_entity_id")}
    rec = candidate_recall(cand_sets, truth)
    print(json.dumps(rec, indent=2, default=str))
    if rec["candidate_recall"] < 1.0:
        print(f"NOTE: blocking recall ceiling is {rec['candidate_recall']:.2%} on this training "
              f"set — any true match not proposed here can never be recovered downstream. "
              f"If this is low on the full dataset, widen BLOCKING top-k / rare_token_max_df "
              f"in config.py before trusting the F0.5 numbers below.")

    print("\n" + "=" * 70)
    print("STAGE: TRAINING (grouped CV + hard-negative reweighting)")
    print("=" * 70)
    lut = s2s3_lookup(train["s2"], train["s3"])
    clf, calibrator, feat_df, feature_names = train_full_pipeline(
        candidates_df, train["s1"], lut, train["gt"], normalizer, MODEL)
    print(f"Model backend: {clf.backend}, n_features: {len(feature_names)}")
    print("Top 15 features by importance:")
    print(clf.feature_importance().head(15).to_string())

    m_raw = macro_micro_f05(
        {s1: set(g[g.oof_raw_score > 0.5].candidate_id) for s1, g in feat_df.groupby("source1_entity_id")},
        truth)
    print(f"\nOOF macro F0.5 @ raw_score>0.5 (naive threshold, for reference): "
          f"{m_raw['macro_f0.5']:.4f}")

    print("\n" + "=" * 70)
    print("STAGE: THRESHOLD SEARCH (optimizing macro F0.5 directly, OOF scores)")
    print("=" * 70)
    abs_t, rel_m, best_f05 = search_thresholds(feat_df, "oof_calibrated_score", truth, DECISION)
    print(f"Best: abs_threshold={abs_t}, rel_margin={rel_m}, OOF macro F0.5={best_f05:.4f}")

    print("\n" + "=" * 70)
    print("STAGE: ABLATION")
    print("=" * 70)
    ablation_df, _ = run_ablation(train, normalizer, candidates_df, feat_df, truth)
    print(ablation_df.to_string(index=False))
    ablation_df.to_csv(models_dir / "ablation_table.csv", index=False)

    print("\n" + "=" * 70)
    print("STAGE: ERROR ANALYSIS (top OOF errors at chosen threshold)")
    print("=" * 70)
    final_preds = apply_decisions(feat_df, "oof_calibrated_score", abs_t, rel_m, DECISION)
    err_df = analyze_errors(feat_df, final_preds, truth, train["s1"], lut, top_n=15)
    if not err_df.empty:
        print(err_df[["error_type", "source1_entity_id", "candidate_id",
                       "s1_name", "cand_name", "score", "category"]].to_string(index=False))
        err_df.to_csv(models_dir / "error_analysis.csv", index=False)
    else:
        print("No errors on this training set at the chosen threshold.")

    # persist model + calibrator + thresholds for the predict stage
    with open(models_dir / "classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    save_calibrator(calibrator, models_dir / "calibrator.pkl")
    with open(models_dir / "decision_config.json", "w") as f:
        json.dump({"abs_threshold": abs_t, "rel_margin": rel_m}, f)
    print(f"\nSaved model artifacts to {models_dir}")

    if args.stage == "train":
        return

    print("\n" + "=" * 70)
    print("STAGE: TEST INFERENCE")
    print("=" * 70)
    feat_test, predictions = run_inference(
        test["s1"], test["s2"], test["s3"], normalizer, clf, calibrator,
        abs_t, rel_m, BLOCKING, DECISION, output_dir)
    n_with_matches = sum(1 for v in predictions.values() if v)
    print(f"Test S1 entities: {len(test['s1'])}, predicted >=1 match: {n_with_matches}, "
          f"predicted singleton: {len(test['s1']) - n_with_matches}")
    print(f"Wrote {output_dir / 'candidate_pairs.tsv'} and {output_dir / 'matching_results.tsv'}")

    print("\n" + "=" * 70)
    print("STAGE: SUBMISSION VALIDATION")
    print("=" * 70)
    import subprocess
    validator = PATHS.root / "utils" / "validate_submission.py"
    if validator.exists():
        result = subprocess.run([sys.executable, str(validator),
                                  "--matching", str(output_dir / "matching_results.tsv"),
                                  "--candidate", str(output_dir / "candidate_pairs.tsv"),
                                  "--test-dir", str(dataset_dir / "test")],
                                 capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            print("VALIDATION FAILED — do not submit.")
            sys.exit(1)
    else:
        print(f"Validator not found at {validator}; skipping (run utils/validate_submission.py manually).")


if __name__ == "__main__":
    main()
