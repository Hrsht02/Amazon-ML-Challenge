"""Scalable, resumable runner for the Business Entity Resolution challenge."""
from __future__ import annotations
import argparse,json,pickle,shutil,time,traceback,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd
from .config import PATHS,BLOCKING,MODEL,DECISION,config_dict
from .data_loader import load_split
from .preprocessing import fit_normalizer,quick_audit
from .faiss_blocking import build_ann_indexes
from .batch_features import compute_features_batch
from .training import grouped_oof_predictions,hard_negative_reweight
from .features import get_feature_names
from .models import PairClassifier,ScoreCalibrator
from .decision import apply_decisions,search_thresholds
from .evaluation import macro_micro_f05

def _run_dir(out,rid): return Path(out)/"run_history"/rid
def _json(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,default=str))
def _log(rd,event,**kw):
    rec={"time":time.strftime("%Y-%m-%dT%H:%M:%S"),"event":event,**kw}
    with (rd/"events.jsonl").open("a",encoding="utf8") as f:f.write(json.dumps(rec)+"\n")
    print(f"[{rec['time']}] {event}",kw)

def _audit(train,test,rd):
    x={"train":quick_audit(train["s1"],train["s2"],train["s3"],train["gt"]),
       "test":quick_audit(test["s1"],test["s2"],test["s3"])}
    _json(rd/"audit.json",x);print(json.dumps(x,indent=2,default=str))

def _sample_train(train,norm,i2,i3,rd):
    gt={sid:set(r.match_list) for sid,r in train["gt"].iterrows()}
    parts=[];tp=rp=cp=0;bs=BLOCKING.s1_batch_size; sampling_closed=False
    for start in range(0,len(train["s1"]),bs):
        b=train["s1"].iloc[start:start+bs]
        c=pd.concat([i2.search(b),i3.search(b)],ignore_index=True)
        if c.empty: continue
        c=c.sort_values(["source1_entity_id","candidate_id","ann_similarity"],ascending=[True,True,False]).drop_duplicates(["source1_entity_id","candidate_id"])
        cp+=len(c)
        for sid,g in c.groupby("source1_entity_id",sort=False):
            truth=gt.get(sid,set());ids=set(g.candidate_id);tp+=len(truth);rp+=len(truth&ids)
            pos=g[g.candidate_id.isin(truth)]
            neg=g[~g.candidate_id.isin(truth)].sort_values("ann_similarity",ascending=False)
            keep=pd.concat([pos,neg.head(MODEL.negatives_per_positive*max(1,len(pos)) if len(pos) else 3)])
            if not sampling_closed:
                parts.append(keep[["source1_entity_id","candidate_id"]])
                if sum(map(len,parts))>=MODEL.max_train_pairs: sampling_closed=True
        if (start//bs+1)%10==0:_log(rd,"sampling_progress",rows=sum(map(len,parts)),candidate_recall=rp/tp if tp else 0)
    cand=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=["source1_entity_id","candidate_id"])
    if len(cand)>MODEL.max_train_pairs:
        labels=np.array([int(r.candidate_id in gt.get(r.source1_entity_id,set())) for r in cand.itertuples(index=False)],dtype=np.int8)
        pos=cand[labels==1]
        neg=cand[labels==0]
        take_neg=max(0,MODEL.max_train_pairs-len(pos))
        neg=neg.sample(n=min(take_neg,len(neg)),random_state=MODEL.seed) if take_neg else neg.iloc[:0]
        cand=pd.concat([pos,neg],ignore_index=True)
    met={"candidate_recall":rp/tp if tp else 1.0,"total_positive_pairs":tp,"recovered_positive_pairs":rp,"candidate_pairs_seen":cp,"training_pairs":len(cand)}
    _json(rd/"blocking_train_metrics.json",met);print(json.dumps(met,indent=2));return cand,gt

def _train(train,norm,i2,i3,rd):
    cand,gt=_sample_train(train,norm,i2,i3,rd)
    feats=compute_features_batch(cand,train["s1"],train["s2"],train["s3"],norm)
    y=np.array([int(r.candidate_id in gt.get(r.source1_entity_id,set())) for r in cand.itertuples(index=False)],dtype=np.int8)
    names=[x for x in get_feature_names() if x in feats.columns]
    oof=grouped_oof_predictions(feats,y,MODEL,names);weights=np.ones(len(y))
    for _ in range(MODEL.hard_negative_rounds):
        weights=hard_negative_reweight(feats,y,oof,MODEL.hard_negatives_per_s1)
        oof=grouped_oof_predictions(feats,y,MODEL,names,weights)
    cal=ScoreCalibrator().fit(oof,y);feats["oof_calibrated_score"]=cal.transform(oof)
    clf=PairClassifier(names,MODEL.lgbm_params,MODEL.seed).fit(feats,y,weights)
    at,rm,best=search_thresholds(feats,"oof_calibrated_score",gt,DECISION)
    metrics=macro_micro_f05(apply_decisions(feats,"oof_calibrated_score",at,rm,DECISION),gt)
    _json(rd/"validation_metrics.json",metrics|{"abs_threshold":at,"rel_margin":rm,"best_oof_f05":best,"training_pairs":len(cand)})
    with open(rd/"classifier.pkl","wb") as f:pickle.dump(clf,f)
    with open(rd/"calibrator.pkl","wb") as f:pickle.dump(cal,f)
    with open(rd/"normalizer.pkl","wb") as f:pickle.dump(norm,f)
    _json(rd/"decision_config.json",{"abs_threshold":at,"rel_margin":rm});_json(rd/"config.json",config_dict())
    return clf,cal,norm,at,rm

def _predict(test,norm,clf,cal,at,rm,i2,i3,out,rd):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);mp=out/"matching_results.tsv";cp=out/"candidate_pairs.tsv"
    with mp.open("w",encoding="utf8") as fm,cp.open("w",encoding="utf8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n");fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for start in range(0,len(test["s1"]),BLOCKING.s1_batch_size):
            b=test["s1"].iloc[start:start+BLOCKING.s1_batch_size]
            c=pd.concat([i2.search(b),i3.search(b)],ignore_index=True)
            if not c.empty:
                c=c.sort_values(["source1_entity_id","candidate_id","ann_similarity"],ascending=[True,True,False]).drop_duplicates(["source1_entity_id","candidate_id"])
                feats=compute_features_batch(c[["source1_entity_id","candidate_id"]],test["s1"],test["s2"],test["s3"],norm)
                feats["score"]=cal.transform(clf.predict_proba(feats));pred=apply_decisions(feats,"score",at,rm,DECISION)
                cg=c.groupby("source1_entity_id").candidate_id.agg(set).to_dict()
            else: pred={};cg={}
            for sid in b.entity_id:
                fc.write(f"{sid}\t{','.join(sorted(cg.get(sid,set())))}\n")
                fm.write(f"{sid}\t{','.join(sorted(pred.get(sid,set())))}\n")
            done=min(start+len(b),len(test["s1"]))
            if done%50000 < len(b):_log(rd,"inference_progress",processed=done,total=len(test["s1"]))
    shutil.copy2(mp,rd/"matching_results.tsv");shutil.copy2(cp,rd/"candidate_pairs.tsv")

def _load(rd):
    with open(rd/"classifier.pkl","rb") as f:clf=pickle.load(f)
    with open(rd/"calibrator.pkl","rb") as f:cal=pickle.load(f)
    with open(rd/"normalizer.pkl","rb") as f:norm=pickle.load(f)
    d=json.loads((rd/"decision_config.json").read_text());return clf,cal,norm,float(d["abs_threshold"]),float(d["rel_margin"])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=["audit","train","predict","all"],default="all")
    ap.add_argument("--dataset-dir",default=str(PATHS.dataset_dir));ap.add_argument("--output-dir",default=str(PATHS.output_dir))
    ap.add_argument("--run-id",help="new run id for train/all; existing run id for predict")
    ap.add_argument("--force-index",action="store_true");a=ap.parse_args()
    if a.stage=="predict":
        if not a.run_id: ap.error("--run-id is required for --stage predict")
        rid=a.run_id;rd=_run_dir(a.output_dir,rid)
        if not (rd/"classifier.pkl").exists(): ap.error(f"No trained artifacts found in {rd}")
    else:
        rid=a.run_id or time.strftime("%Y%m%d_%H%M%S");rd=_run_dir(a.output_dir,rid);rd.mkdir(parents=True,exist_ok=True)
    try:
        train=load_split(Path(a.dataset_dir),"train");test=load_split(Path(a.dataset_dir),"test");_audit(train,test,rd)
        if a.stage=="audit":return
        if a.stage=="predict":
            clf,cal,norm,at,rm=_load(rd)
        else:
            norm=fit_normalizer(train["s1"],train["s2"],train["s3"])
            i2,i3=build_ann_indexes(train["s2"],train["s3"],Path(a.output_dir)/"ann_indexes_train_v2",BLOCKING,a.force_index)
            clf,cal,norm,at,rm=_train(train,norm,i2,i3,rd)
        if a.stage in ("predict","all"):
            ti2,ti3=build_ann_indexes(test["s2"],test["s3"],Path(a.output_dir)/"ann_indexes_test_v2",BLOCKING,False)
            _predict(test,norm,clf,cal,at,rm,ti2,ti3,a.output_dir,rd)
            validator=Path(a.dataset_dir).parent/"utils"/"validate_submission.py"
            if validator.exists():
                r=subprocess.run([sys.executable,str(validator),"--matching",str(Path(a.output_dir)/"matching_results.tsv"),"--candidate",str(Path(a.output_dir)/"candidate_pairs.tsv"),"--test-dir",str(Path(a.dataset_dir)/"test")],capture_output=True,text=True)
                (rd/"validator.txt").write_text(r.stdout+"\n"+r.stderr);print(r.stdout)
                if r.returncode!=0: raise RuntimeError("Submission validation failed; see run_history/<run_id>/validator.txt")
        _json(rd/"run_manifest.json",{"run_id":rid,"stage":a.stage,"dataset_dir":a.dataset_dir,"config":config_dict(),"finished":time.strftime("%Y-%m-%dT%H:%M:%S")})
        print(f"RUN_ID={rid}")
    except Exception as e:
        _json(rd/"failure.json",{"error":str(e),"traceback":traceback.format_exc()});raise

if __name__=="__main__":main()
