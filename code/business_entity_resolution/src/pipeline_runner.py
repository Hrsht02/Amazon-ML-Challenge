"""Scalable, visible and crash-safe runner.

Re-running the exact same --run-id resumes from the latest completed
checkpoint. A new --run-id starts a clean experiment. Final submission files
remain exactly the required TSV files at output/matching_results.tsv and
output/candidate_pairs.tsv.
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PATHS, BLOCKING, MODEL, DECISION, config_dict
from .data_loader import load_split
from .preprocessing import fit_normalizer, quick_audit
from .faiss_blocking import build_ann_indexes
from .batch_features import compute_features_batch
from .training import grouped_oof_predictions, hard_negative_reweight
from .features import get_feature_names
from .models import PairClassifier, ScoreCalibrator
from .decision import apply_decisions, search_thresholds
from .evaluation import macro_micro_f05
from .checkpoint import atomic_json, atomic_pickle, load_pickle, mark_done, is_done, print_resources


def _run_dir(out, rid):
    return Path(out) / "run_history" / rid


def _json(p, x):
    atomic_json(Path(p), x)


def _log(rd, event, **kw):
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **kw}
    with (rd / "events.jsonl").open("a", encoding="utf8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print(f"[{rec['time']}] {event} {kw}", flush=True)


def _banner(rd, title):
    print("\n" + "=" * 88, flush=True)
    print(title, flush=True)
    print("=" * 88, flush=True)
    print_resources()
    _log(rd, "stage_start", stage=title)


def _audit(train, test, rd):
    x = {
        "train": quick_audit(train["s1"], train["s2"], train["s3"], train["gt"]),
        "test": quick_audit(test["s1"], test["s2"], test["s3"]),
    }
    _json(rd / "audit.json", x)
    print(json.dumps(x, indent=2, default=str), flush=True)
    print(
        f"[AUDIT] train S1={x['train']['n_s1']:,} "
        f"S2={x['train']['n_s2']:,} S3={x['train']['n_s3']:,} | "
        f"avg matches/S1={x['train'].get('avg_matches_per_s1', float('nan')):.4f} | "
        f"singleton={x['train'].get('singleton_rate', float('nan')):.4%}",
        flush=True,
    )


def _sampling_state_path(rd):
    return rd / "sampling_state.json"


def _sample_train(train, norm, i2, i3, rd):
    """Build bounded training pairs while checkpointing the exact S1 position.

    The scan continues after the 3M pair cap only to measure full training
    candidate recall. If Colab dies, the next run resumes from next_start.
    """
    state_path = _sampling_state_path(rd)
    chunk_dir = rd / "sampling_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    gt = {sid: set(r.match_list) for sid, r in train["gt"].iterrows()}

    state = {
        "version": 2,
        "next_start": 0,
        "tp": 0,
        "rp": 0,
        "cp": 0,
        "sampled_rows": 0,
        "closed": False,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if state_path.exists():
        state.update(json.loads(state_path.read_text()))
        print(
            f"[SAMPLING] RESUMING from S1 row {state['next_start']:,}/"
            f"{len(train['s1']):,}; sampled={state['sampled_rows']:,}; "
            f"recall={state['rp']/state['tp'] if state['tp'] else 0:.4%}",
            flush=True,
        )
    else:
        print("[SAMPLING] Starting from S1 row 0", flush=True)

    bs = BLOCKING.s1_batch_size
    pending = []
    pending_rows = 0
    t0 = time.time()
    start = int(state["next_start"])

    for start in range(start, len(train["s1"]), bs):
        b = train["s1"].iloc[start:start + bs]
        c = pd.concat([i2.search(b), i3.search(b)], ignore_index=True)
        if not c.empty:
            c = (
                c.sort_values(
                    ["source1_entity_id", "candidate_id", "ann_similarity"],
                    ascending=[True, True, False],
                )
                .drop_duplicates(["source1_entity_id", "candidate_id"])
            )

            for sid, g in c.groupby("source1_entity_id", sort=False):
                truth = gt.get(sid, set())
                ids = set(g.candidate_id)
                state["tp"] += len(truth)
                state["rp"] += len(truth & ids)
                state["cp"] += len(g)

                if not state["closed"]:
                    pos = g[g.candidate_id.isin(truth)]
                    neg = g[~g.candidate_id.isin(truth)].sort_values(
                        "ann_similarity", ascending=False
                    )
                    keep_n = MODEL.negatives_per_positive * max(1, len(pos)) if len(pos) else 3
                    keep = pd.concat([pos, neg.head(keep_n)])
                    pending.append(keep)
                    pending_rows += len(keep)
                    state["sampled_rows"] += len(keep)
                    if state["sampled_rows"] >= MODEL.max_train_pairs:
                        state["closed"] = True
                        print(
                            f"[SAMPLING] Training-pair cap reached: {state['sampled_rows']:,} "
                            f"(cap={MODEL.max_train_pairs:,}). Continuing only for recall audit.",
                            flush=True,
                        )

        batch_done = min(start + len(b), len(train["s1"]))
        should_checkpoint = (
            ((start // bs + 1) % MODEL.checkpoint_every_batches == 0)
            or batch_done == len(train["s1"])
        )

        if should_checkpoint:
            if pending:
                chunk_id = int(batch_done)
                chunk_path = chunk_dir / f"chunk_{chunk_id:09d}.pkl.gz"
                if not chunk_path.exists():
                    chunk = pd.concat(pending, ignore_index=True)
                    chunk.to_pickle(chunk_path, compression="gzip")
                    print(
                        f"[SAMPLING] checkpoint chunk saved: {chunk_path.name} "
                        f"rows={len(chunk):,}",
                        flush=True,
                    )
                pending.clear()
                pending_rows = 0

            state["next_start"] = batch_done
            state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            state["elapsed_sec"] = time.time() - t0
            atomic_json(state_path, state)

            recall = state["rp"] / state["tp"] if state["tp"] else 0.0
            progress = batch_done / len(train["s1"])
            mode = "RECALL-ONLY" if state["closed"] else "SAMPLING"
            _log(
                rd,
                "sampling_progress",
                mode=mode,
                rows=state["sampled_rows"],
                s1_processed=batch_done,
                total_s1=len(train["s1"]),
                progress=f"{progress:.2%}",
                candidate_recall=recall,
                candidate_pairs_seen=state["cp"],
                elapsed_min=round((time.time() - t0) / 60, 2),
            )
            print_resources()

    # If the process ended normally, flush any last pending rows.
    if pending:
        chunk_id = len(train["s1"])
        chunk_path = chunk_dir / f"chunk_{chunk_id:09d}.pkl.gz"
        pd.concat(pending, ignore_index=True).to_pickle(chunk_path, compression="gzip")

    print("[SAMPLING] Full S1 scan completed.", flush=True)
    print(
        f"[SAMPLING] Candidate recall = {state['rp']/state['tp'] if state['tp'] else 1.0:.6%}",
        flush=True,
    )

    chunks = sorted(chunk_dir.glob("chunk_*.pkl.gz"))
    parts = [pd.read_pickle(p) for p in chunks]
    cand = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=[
            "source1_entity_id", "candidate_id", "ann_similarity",
            "ann_ascii_name_similarity", "ann_unicode_name_similarity",
            "ann_address_similarity", "ann_name_rank",
            "ann_unicode_name_rank", "ann_address_rank", "ann_n_strategies"
        ]
    )

    if len(cand) > MODEL.max_train_pairs:
        labels = np.array([
            int(r.candidate_id in gt.get(r.source1_entity_id, set()))
            for r in cand.itertuples(index=False)
        ], dtype=np.int8)
        pos, neg = cand[labels == 1], cand[labels == 0]
        take_neg = max(0, MODEL.max_train_pairs - len(pos))
        neg = (
            neg.sample(n=min(take_neg, len(neg)), random_state=MODEL.seed)
            if take_neg else neg.iloc[:0]
        )
        cand = pd.concat([pos, neg], ignore_index=True)

    met = {
        "candidate_recall": state["rp"] / state["tp"] if state["tp"] else 1.0,
        "total_positive_pairs": state["tp"],
        "recovered_positive_pairs": state["rp"],
        "candidate_pairs_seen": state["cp"],
        "training_pairs": len(cand),
        "max_train_pairs": MODEL.max_train_pairs,
        "s1_scanned": len(train["s1"]),
    }
    _json(rd / "blocking_train_metrics.json", met)
    atomic_pickle(rd / "training_candidates.pkl.gz", cand)
    print(json.dumps(met, indent=2), flush=True)
    print_resources()
    mark_done(rd, "sampling", **{k: v for k, v in met.items() if isinstance(v, (int, float))})
    return cand, gt


def _feature_names(feats):
    extra = [
        "ann_similarity", "ann_ascii_name_similarity",
        "ann_unicode_name_similarity", "ann_address_similarity",
        "ann_name_rank", "ann_unicode_name_rank", "ann_address_rank",
        "ann_n_strategies",
    ]
    return [x for x in get_feature_names() + extra if x in feats.columns]


def _train(train, norm, i2, i3, rd):
    _banner(rd, "TRAINING: candidate generation + feature construction + OOF")
    if is_done(rd, "training"):
        print("[TRAINING] Completed checkpoint exists; loading artifacts.", flush=True)
        return _load(rd)

    cand_path = rd / "training_candidates.pkl.gz"
    if cand_path.exists():
        print("[TRAINING] Loading saved candidate pairs checkpoint.", flush=True)
        cand = load_pickle(cand_path)
        gt = {sid: set(r.match_list) for sid, r in train["gt"].iterrows()}
    else:
        cand, gt = _sample_train(train, norm, i2, i3, rd)

    feat_path = rd / "training_features.pkl.gz"
    if feat_path.exists():
        print("[TRAINING] Loading saved feature matrix checkpoint.", flush=True)
        feats = load_pickle(feat_path)
    else:
        feats = compute_features_batch(cand, train["s1"], train["s2"], train["s3"], norm)
        feats["label"] = np.array([
            int(r.candidate_id in gt.get(r.source1_entity_id, set()))
            for r in feats.itertuples(index=False)
        ], dtype=np.int8)
        atomic_pickle(feat_path, feats)
        print(f"[TRAINING] Feature matrix checkpoint saved: rows={len(feats):,}, cols={len(feats.columns)}", flush=True)

    y = feats["label"].to_numpy(dtype=np.int8)
    names = _feature_names(feats)
    print(f"[TRAINING] rows={len(feats):,} positives={int(y.sum()):,} positive_rate={y.mean():.6%}", flush=True)
    print(f"[TRAINING] features={len(names)}", flush=True)
    print("[TRAINING] Feature names:", ", ".join(names), flush=True)
    print_resources()

    oof_dir = rd / "oof_checkpoints"
    weights = np.ones(len(y), dtype=np.float32)

    oof = grouped_oof_predictions(
        feats, y, MODEL, names, weights, checkpoint_dir=oof_dir, tag="round0"
    )
    for round_id in range(1, MODEL.hard_negative_rounds + 1):
        weights = hard_negative_reweight(feats, y, oof, MODEL.hard_negatives_per_s1)
        print(
            f"[TRAINING] hard-negative round {round_id}: "
            f"weighted_rows={(weights > 1).sum():,}",
            flush=True,
        )
        oof = grouped_oof_predictions(
            feats, y, MODEL, names, weights,
            checkpoint_dir=oof_dir, tag=f"round{round_id}"
        )

    cal = ScoreCalibrator().fit(oof, y)
    calibrated = cal.transform(oof)
    feats["oof_calibrated_score"] = calibrated.astype("float32")

    if (rd / "classifier.pkl").exists() and (rd / "calibrator.pkl").exists():
        print("[TRAINING] Final model artifacts already exist; reusing them.", flush=True)
        with open(rd / "classifier.pkl", "rb") as f:
            clf = pickle.load(f)
        with open(rd / "calibrator.pkl", "rb") as f:
            cal = pickle.load(f)
    else:
        print("[TRAINING] Fitting final LightGBM on ALL training pairs...", flush=True)
        clf = PairClassifier(names, MODEL.lgbm_params, MODEL.seed).fit(feats, y, weights)
        with open(rd / "classifier.pkl", "wb") as f:
            pickle.dump(clf, f)
        with open(rd / "calibrator.pkl", "wb") as f:
            pickle.dump(cal, f)
        print(f"[TRAINING] Final model saved; device={clf.device_used}", flush=True)

    with open(rd / "normalizer.pkl", "wb") as f:
        pickle.dump(norm, f)

    print("[VALIDATION] Searching absolute threshold × relative margin for macro F0.5...", flush=True)
    at, rm, best = search_thresholds(feats, "oof_calibrated_score", gt, DECISION)
    pred = apply_decisions(feats, "oof_calibrated_score", at, rm, DECISION)
    metrics = macro_micro_f05(pred, gt)
    compact = {k: v for k, v in metrics.items() if k != "per_entity"}
    compact.update({
        "abs_threshold": at, "rel_margin": rm,
        "best_oof_f05": best, "training_pairs": len(cand),
        "candidate_recall": json.loads((rd / "blocking_train_metrics.json").read_text())["candidate_recall"],
        "model_device": getattr(clf, "device_used", "unknown"),
        "n_features": len(names),
    })
    _json(rd / "validation_metrics.json", compact)
    _json(rd / "decision_config.json", {"abs_threshold": at, "rel_margin": rm})
    _json(rd / "config.json", config_dict())
    print(json.dumps(compact, indent=2, default=str), flush=True)

    mark_done(rd, "training", best_oof_f05=best, candidate_recall=compact["candidate_recall"])
    print_resources()
    return clf, cal, norm, at, rm


def _load(rd):
    with open(rd / "classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open(rd / "calibrator.pkl", "rb") as f:
        cal = pickle.load(f)
    with open(rd / "normalizer.pkl", "rb") as f:
        norm = pickle.load(f)
    d = json.loads((rd / "decision_config.json").read_text())
    return clf, cal, norm, float(d["abs_threshold"]), float(d["rel_margin"])


def _predict(test, norm, clf, cal, at, rm, i2, i3, out, rd):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mp, cp = out / "matching_results.tsv", out / "candidate_pairs.tsv"

    if is_done(rd, "prediction") and mp.exists() and cp.exists():
        print("[PREDICT] Completed checkpoint exists; reusing submission files.", flush=True)
        return

    # Resume-safe append position. We write each batch to a separate part and
    # assemble the two required TSVs only after all batches are complete.
    pred_dir = rd / "prediction_parts"
    pred_dir.mkdir(parents=True, exist_ok=True)
    bs = BLOCKING.s1_batch_size

    for start in range(0, len(test["s1"]), bs):
        part_id = start // bs
        mp_part = pred_dir / f"matching_{part_id:06d}.tsv"
        cp_part = pred_dir / f"candidate_{part_id:06d}.tsv"
        if mp_part.exists() and cp_part.exists():
            print(f"[PREDICT] batch {part_id+1} already complete -> resume skip", flush=True)
            continue

        b = test["s1"].iloc[start:start + bs]
        c = pd.concat([i2.search(b), i3.search(b)], ignore_index=True)
        if not c.empty:
            c = (
                c.sort_values(
                    ["source1_entity_id", "candidate_id", "ann_similarity"],
                    ascending=[True, True, False],
                )
                .drop_duplicates(["source1_entity_id", "candidate_id"])
            )
            feats = compute_features_batch(c, test["s1"], test["s2"], test["s3"], norm, progress_every=20_000)
            feats["score"] = cal.transform(clf.predict_proba(feats)).astype("float32")
            pred = apply_decisions(feats, "score", at, rm, DECISION)
            cg = c.groupby("source1_entity_id").candidate_id.agg(set).to_dict()
        else:
            pred, cg = {}, {}

        with mp_part.open("w", encoding="utf8") as fm, cp_part.open("w", encoding="utf8") as fc:
            for sid in b.entity_id:
                fc.write(f"{sid}\t{','.join(sorted(cg.get(sid, set())))}\n")
                fm.write(f"{sid}\t{','.join(sorted(pred.get(sid, set())))}\n")

        done = min(start + len(b), len(test["s1"]))
        _log(rd, "inference_progress", processed=done, total=len(test["s1"]), progress=f"{done/len(test['s1']):.2%}")
        print_resources()
        gc.collect()

    # Assemble exactly the required headers and one row per test S1 entity.
    with mp.open("w", encoding="utf8") as fm, cp.open("w", encoding="utf8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for start in range(0, len(test["s1"]), bs):
            part_id = start // bs
            mp_part = pred_dir / f"matching_{part_id:06d}.tsv"
            cp_part = pred_dir / f"candidate_{part_id:06d}.tsv"
            with mp_part.open("r", encoding="utf8") as f:
                shutil.copyfileobj(f, fm)
            with cp_part.open("r", encoding="utf8") as f:
                shutil.copyfileobj(f, fc)

    if sum(1 for _ in mp.open("r", encoding="utf8")) != len(test["s1"]) + 1:
        raise RuntimeError("matching_results.tsv row-count validation failed")
    if sum(1 for _ in cp.open("r", encoding="utf8")) != len(test["s1"]) + 1:
        raise RuntimeError("candidate_pairs.tsv row-count validation failed")

    shutil.copy2(mp, rd / "matching_results.tsv")
    shutil.copy2(cp, rd / "candidate_pairs.tsv")
    mark_done(rd, "prediction", rows=len(test["s1"]))
    print(f"[PREDICT] FINAL matching_results.tsv: {mp}", flush=True)
    print(f"[PREDICT] FINAL candidate_pairs.tsv: {cp}", flush=True)


def _load_train_or_empty_audit(train):
    return {
        "s1": pd.DataFrame(columns=train["s1"].columns),
        "s2": pd.DataFrame(columns=train["s2"].columns),
        "s3": pd.DataFrame(columns=train["s3"].columns),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["audit", "train", "predict", "all"], default="all")
    ap.add_argument("--dataset-dir", default=str(PATHS.dataset_dir))
    ap.add_argument("--output-dir", default=str(PATHS.output_dir))
    ap.add_argument("--run-id", help="same run-id resumes; a new run-id starts fresh")
    ap.add_argument("--force-index", action="store_true")
    a = ap.parse_args()

    rid = a.run_id or time.strftime("%Y%m%d_%H%M%S")
    rd = _run_dir(a.output_dir, rid)
    rd.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 88, flush=True)
    print(f"BUSINESS ENTITY RESOLUTION | RUN_ID={rid} | STAGE={a.stage}", flush=True)
    print("#" * 88, flush=True)
    print(f"Dataset: {a.dataset_dir}", flush=True)
    print(f"Output:  {a.output_dir}", flush=True)
    print(f"Resume:  checkpoints in {rd}", flush=True)
    print_resources()

    try:
        dataset_dir = Path(a.dataset_dir)
        output_dir = Path(a.output_dir)

        if a.stage == "audit":
            train = load_split(dataset_dir, "train")
            test = load_split(dataset_dir, "test")
            _audit(train, test, rd)
            mark_done(rd, "audit")
            return

        train = load_split(dataset_dir, "train")
        if not is_done(rd, "audit"):
            _audit(train, _load_train_or_empty_audit(train), rd)
            mark_done(rd, "audit")

        # Normalizer checkpoint.
        norm_path = rd / "normalizer.pkl"
        if norm_path.exists():
            with open(norm_path, "rb") as f:
                norm = pickle.load(f)
            print("[NORMALIZER] Reused checkpoint.", flush=True)
        else:
            print("[NORMALIZER] Fitting bounded data-driven normalizer...", flush=True)
            norm = fit_normalizer(train["s1"], train["s2"], train["s3"])
            with open(norm_path, "wb") as f:
                pickle.dump(norm, f)
            print("[NORMALIZER] Saved.", flush=True)

        if a.stage in ("train", "all"):
            i2, i3 = build_ann_indexes(
                train["s2"], train["s3"],
                output_dir / "ann_indexes_train_v3",
                BLOCKING, a.force_index
            )
            clf, cal, norm, at, rm = _train(train, norm, i2, i3, rd)
        else:
            if not a.run_id:
                ap.error("--run-id is required for --stage predict")
            clf, cal, norm, at, rm = _load(rd)

        if a.stage in ("predict", "all"):
            del train
            gc.collect()
            test = load_split(dataset_dir, "test")
            ti2, ti3 = build_ann_indexes(
                test["s2"], test["s3"],
                output_dir / "ann_indexes_test_v3",
                BLOCKING, False
            )
            _predict(test, norm, clf, cal, at, rm, ti2, ti3, output_dir, rd)

            validator = dataset_dir.parent / "utils" / "validate_submission.py"
            if validator.exists():
                r = subprocess.run([
                    sys.executable, str(validator),
                    "--matching", str(output_dir / "matching_results.tsv"),
                    "--candidate", str(output_dir / "candidate_pairs.tsv"),
                    "--test-dir", str(dataset_dir / "test")
                ], capture_output=True, text=True)
                (rd / "validator.txt").write_text(r.stdout + "\n" + r.stderr)
                print(r.stdout, flush=True)
                if r.returncode != 0:
                    raise RuntimeError("Submission validation failed; see validator.txt")
            mark_done(rd, "validated")
            print("[VALIDATION] Submission files passed local structural validation.", flush=True)

        _json(rd / "run_manifest.json", {
            "run_id": rid, "stage": a.stage,
            "dataset_dir": a.dataset_dir, "output_dir": a.output_dir,
            "config": config_dict(),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        print(f"\nRUN_ID={rid}", flush=True)
        print_resources()
    except Exception as e:
        _json(rd / "failure.json", {
            "error": str(e),
            "traceback": __import__("traceback").format_exc(),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        print(f"[FATAL] {type(e).__name__}: {e}", flush=True)
        print(f"[FATAL] Resume with the SAME --run-id: {rid}", flush=True)
        raise


if __name__ == "__main__":
    main()
