"""Scalable ANN blocking for millions of records.

Uses hashed character n-grams + FAISS IVF-PQ instead of sklearn TF-IDF +
exhaustive nearest-neighbor search. Indexes are persisted and reused.
"""
from __future__ import annotations
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
import faiss
from .normalization import to_alnum
from .config import BlockingConfig

class ANNSourceIndex:
    def __init__(self, df, index_dir, source, cfg):
        self.df=df.reset_index(drop=True)
        self.index_dir=Path(index_dir); self.index_dir.mkdir(parents=True,exist_ok=True)
        self.source=source; self.cfg=cfg
        self.vec=HashingVectorizer(analyzer="char",ngram_range=(2,5),
            n_features=cfg.ann_dim,alternate_sign=False,norm="l2",lowercase=False)
        self.name=None; self.addr=None

    def _new(self):
        q=faiss.IndexFlatIP(self.cfg.ann_dim)
        base=faiss.IndexIVFPQ(q,self.cfg.ann_dim,self.cfg.ann_nlist,
                              self.cfg.ann_pq_m,8,faiss.METRIC_INNER_PRODUCT)
        base.nprobe=self.cfg.ann_nprobe
        return faiss.IndexIDMap2(base)

    def _vec(self,texts):
        x=self.vec.transform(texts).astype("float32").toarray()
        faiss.normalize_L2(x)
        return x

    def _texts(self,col,start=0,end=None):
        return self.df[col].iloc[start:end].fillna("").astype(str).map(to_alnum).tolist()

    def build(self,force=False):
        npth=self.index_dir/f"{self.source}_name.faiss"
        apth=self.index_dir/f"{self.source}_address.faiss"
        if not force and npth.exists() and apth.exists():
            self.name=faiss.read_index(str(npth)); self.addr=faiss.read_index(str(apth)); return
        self.name=self._new(); self.addr=self._new()
        rng=np.random.default_rng(self.cfg.ann_dim+self.cfg.ann_nlist)
        n=len(self.df); take=min(n,max(100_000,self.cfg.ann_nlist*8))
        sample=rng.choice(n,take,replace=False) if n>take else np.arange(n)
        self.name.index.train(self._vec(self.df.iloc[sample].business_name.fillna("").astype(str).map(to_alnum).tolist()))
        self.addr.index.train(self._vec(self.df.iloc[sample].business_address.fillna("").astype(str).map(to_alnum).tolist()))
        bs=self.cfg.index_build_batch_size
        for s in range(0,n,bs):
            e=min(s+bs,n); ids=np.arange(s,e,dtype="int64")
            self.name.add_with_ids(self._vec(self._texts("business_name",s,e)),ids)
            self.addr.add_with_ids(self._vec(self._texts("business_address",s,e)),ids)
        faiss.write_index(self.name,str(npth)); faiss.write_index(self.addr,str(apth))
        with open(self.index_dir/f"{self.source}_meta.pkl","wb") as f:
            pickle.dump({"rows":n,"dim":self.cfg.ann_dim},f)

    def search(self,s1_df):
        if self.name is None: raise RuntimeError("Index not built")
        names=s1_df.business_name.fillna("").astype(str).map(to_alnum).tolist()
        addrs=s1_df.business_address.fillna("").astype(str).map(to_alnum).tolist()
        dn,inn=self.name.search(self._vec(names),self.cfg.ann_name_k)
        da,ina=self.addr.search(self._vec(addrs),self.cfg.ann_address_k)
        rows=[]
        for i,sid in enumerate(s1_df.entity_id.to_numpy()):
            cand={}
            for j,rid in enumerate(inn[i]):
                if rid>=0: cand[int(rid)]=max(cand.get(int(rid),0.0),float(dn[i,j]))
            for j,rid in enumerate(ina[i]):
                if rid>=0: cand[int(rid)]=max(cand.get(int(rid),0.0),float(da[i,j]))
            for rid,sim in sorted(cand.items(),key=lambda x:-x[1])[:self.cfg.max_candidates_per_s1]:
                rows.append((sid,self.df.iloc[rid].entity_id,sim))
        return pd.DataFrame(rows,columns=["source1_entity_id","candidate_id","ann_similarity"])

def build_ann_indexes(s2,s3,index_dir,cfg,force=False):
    i2=ANNSourceIndex(s2,Path(index_dir)/"S2","S2",cfg); i3=ANNSourceIndex(s3,Path(index_dir)/"S3","S3",cfg)
    i2.build(force); i3.build(force); return i2,i3
