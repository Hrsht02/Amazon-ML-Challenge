"""High-recall, resumable ANN blocking for millions of records.

The blocker keeps two name representations:
  1) ASCII-normalized character n-grams for robustness to punctuation/accents.
  2) Unicode-preserving character n-grams so non-Latin business names are not
     silently erased.

FAISS indexes are persisted on CPU. If a GPU-enabled FAISS build is available,
search automatically uses the GPU and falls back to CPU on any CUDA/FAISS
failure, so a temporary GPU problem does not invalidate the run.
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
        # Do not reset/copy the 5M-row dataframe.
        self.df = df
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.source = source
        self.cfg = cfg

        self.vec_ascii = HashingVectorizer(
            analyzer="char", ngram_range=(2, 5), n_features=cfg.ann_dim,
            alternate_sign=False, norm="l2", lowercase=False
        )
        self.vec_unicode = HashingVectorizer(
            analyzer="char", ngram_range=(2, 5), n_features=cfg.ann_dim,
            alternate_sign=False, norm="l2", lowercase=False
        )
        self.name = None
        self.name_unicode = None
        self.addr = None
        self._gpu = None
        self._gpu_enabled = bool(getattr(cfg, "use_faiss_gpu", True))
        self._gpu_failed = False

    def _new(self):
        q = faiss.IndexFlatIP(self.cfg.ann_dim)
        base = faiss.IndexIVFPQ(
            q, self.cfg.ann_dim, self.cfg.ann_nlist,
            self.cfg.ann_pq_m, 8, faiss.METRIC_INNER_PRODUCT
        )
        base.nprobe = self.cfg.ann_nprobe
        return faiss.IndexIDMap2(base)

    def _vec(self, texts, unicode=False):
        vectorizer = self.vec_unicode if unicode else self.vec_ascii
        x = vectorizer.transform(texts).astype("float32").toarray()
        faiss.normalize_L2(x)
        return x

    def _texts(self, col, start=0, end=None, unicode=False):
        vals = self.df[col].iloc[start:end].fillna("").astype(str)
        return vals.map(lambda s: to_alnum(s, fold_ascii=not unicode)).tolist()

    def _meta(self):
        return {
            "rows": len(self.df), "dim": self.cfg.ann_dim,
            "nlist": self.cfg.ann_nlist, "pq_m": self.cfg.ann_pq_m,
            "nprobe": self.cfg.ann_nprobe,
            "train_size": self.cfg.ann_train_size,
            "name_k": self.cfg.ann_name_k,
            "unicode_name_k": self.cfg.ann_unicode_name_k,
            "address_k": self.cfg.ann_address_k,
            "max_candidates": self.cfg.max_candidates_per_s1,
            "version": 3,
        }

    def build(self, force=False):
        paths = {
            "name": self.index_dir / f"{self.source}_name.faiss",
            "unicode": self.index_dir / f"{self.source}_name_unicode.faiss",
            "addr": self.index_dir / f"{self.source}_address.faiss",
            "meta": self.index_dir / f"{self.source}_meta.pkl",
        }
        expected = self._meta()

        if not force and all(p.exists() for p in paths.values()):
            with open(paths["meta"], "rb") as f:
                meta = pickle.load(f)
            if all(meta.get(k) == v for k, v in expected.items()):
                print(f"[FAISS] Reusing cached {self.source} indexes", flush=True)
                self.name = faiss.read_index(str(paths["name"]))
                self.name_unicode = faiss.read_index(str(paths["unicode"]))
                self.addr = faiss.read_index(str(paths["addr"]))
                return

        self.name = self._new()
        self.name_unicode = self._new()
        self.addr = self._new()

        rng = np.random.default_rng(42 + self.cfg.ann_nlist + self.cfg.ann_pq_m)
        n = len(self.df)
        take = min(n, self.cfg.ann_train_size)
        sample = rng.choice(n, take, replace=False) if n > take else np.arange(n)

        print(
            f"[FAISS] Building {self.source}: rows={n:,}, train_vectors={take:,}, "
            f"dim={self.cfg.ann_dim}, nlist={self.cfg.ann_nlist:,}, "
            f"nprobe={self.cfg.ann_nprobe}, name_k={self.cfg.ann_name_k}, "
            f"unicode_name_k={self.cfg.ann_unicode_name_k}, address_k={self.cfg.ann_address_k}",
            flush=True,
        )

        names = self.df.iloc[sample].business_name.fillna("").astype(str).tolist()
        addrs = self.df.iloc[sample].business_address.fillna("").astype(str).tolist()
        self.name.index.train(self._vec(names, unicode=False))
        print(f"[FAISS] {self.source} ASCII-name quantizer trained", flush=True)
        self.name_unicode.index.train(self._vec(names, unicode=True))
        print(f"[FAISS] {self.source} Unicode-name quantizer trained", flush=True)
        self.addr.index.train(self._vec(addrs, unicode=False))
        print(f"[FAISS] {self.source} address quantizer trained", flush=True)

        bs = self.cfg.index_build_batch_size
        for s in range(0, n, bs):
            e = min(s + bs, n)
            ids = np.arange(s, e, dtype="int64")
            self.name.add_with_ids(self._vec(self._texts("business_name", s, e), False), ids)
            self.name_unicode.add_with_ids(self._vec(self._texts("business_name", s, e, True), True), ids)
            self.addr.add_with_ids(self._vec(self._texts("business_address", s, e), False), ids)
            if ((s // bs) + 1) % 20 == 0 or e == n:
                print(f"[FAISS] {self.source} indexed {e:,}/{n:,} rows", flush=True)

        faiss.write_index(self.name, str(paths["name"]))
        faiss.write_index(self.name_unicode, str(paths["unicode"]))
        faiss.write_index(self.addr, str(paths["addr"]))
        with open(paths["meta"], "wb") as f:
            pickle.dump(expected, f)
        print(f"[FAISS] {self.source} indexes saved", flush=True)

    def _enable_gpu(self):
        if not self._gpu_enabled or self._gpu_failed or self._gpu is not None:
            return
        try:
            if not hasattr(faiss, "StandardGpuResources"):
                print("[FAISS] GPU FAISS bindings unavailable; using CPU", flush=True)
                self._gpu_failed = True
                return
            res = faiss.StandardGpuResources()
            self._gpu = {
                "res": res,
                "name": faiss.index_cpu_to_gpu(res, 0, self.name),
                "unicode": faiss.index_cpu_to_gpu(res, 0, self.name_unicode),
                "addr": faiss.index_cpu_to_gpu(res, 0, self.addr),
            }
            print("[FAISS] GPU search enabled", flush=True)
        except Exception as e:
            print(f"[FAISS] GPU initialization failed -> CPU fallback: {type(e).__name__}: {e}", flush=True)
            self._gpu = None
            self._gpu_failed = True

    def _search_one(self, index, x, k):
        return index.search(x, k)

    def search(self, s1_df):
        if self.name is None:
            raise RuntimeError("Index not built")

        names_ascii = s1_df.business_name.fillna("").astype(str).map(to_alnum).tolist()
        names_unicode = s1_df.business_name.fillna("").astype(str).map(
            lambda s: to_alnum(s, fold_ascii=False)
        ).tolist()
        addrs = s1_df.business_address.fillna("").astype(str).map(to_alnum).tolist()

        self._enable_gpu()
        indexes = self._gpu if self._gpu is not None else {
            "name": self.name, "unicode": self.name_unicode, "addr": self.addr
        }

        try:
            dn, inn = self._search_one(indexes["name"], self._vec(names_ascii), self.cfg.ann_name_k)
            du, inu = self._search_one(indexes["unicode"], self._vec(names_unicode, True), self.cfg.ann_unicode_name_k)
            da, ina = self._search_one(indexes["addr"], self._vec(addrs), self.cfg.ann_address_k)
        except Exception as e:
            if self._gpu is None:
                raise
            print(f"[FAISS] GPU search failed -> disabling GPU and retrying batch on CPU: {type(e).__name__}: {e}", flush=True)
            self._gpu = None
            self._gpu_failed = True
            dn, inn = self.name.search(self._vec(names_ascii), self.cfg.ann_name_k)
            du, inu = self.name_unicode.search(self._vec(names_unicode, True), self.cfg.ann_unicode_name_k)
            da, ina = self.addr.search(self._vec(addrs), self.cfg.ann_address_k)

        rows = []
        for i, sid in enumerate(s1_df.entity_id.to_numpy()):
            cand = {}
            def add_hits(ids, sims, source, k):
                for rank, rid in enumerate(ids[i], start=1):
                    if rid < 0:
                        continue
                    rid = int(rid)
                    rec = cand.setdefault(rid, {
                        "ascii_name": 0.0, "unicode_name": 0.0, "address": 0.0,
                        "name_rank": 10**9, "unicode_name_rank": 10**9,
                        "address_rank": 10**9,
                    })
                    rec[source] = max(rec[source], float(sims[i, rank - 1]))
                    rec[f"{source}_rank"] = min(rec[f"{source}_rank"], rank)

            add_hits(inn, dn, "ascii_name", self.cfg.ann_name_k)
            add_hits(inu, du, "unicode_name", self.cfg.ann_unicode_name_k)
            add_hits(ina, da, "address", self.cfg.ann_address_k)

            ranked = []
            for rid, rec in cand.items():
                best = max(rec["ascii_name"], rec["unicode_name"], rec["address"])
                evidence = int(rec["ascii_name"] > 0) + int(rec["unicode_name"] > 0) + int(rec["address"] > 0)
                ranked.append((rid, best, evidence, rec))
            ranked.sort(key=lambda x: (-x[2], -x[1], x[0]))

            for rid, best, evidence, rec in ranked[:self.cfg.max_candidates_per_s1]:
                rows.append((
                    sid, self.df.iloc[rid].entity_id,
                    best, rec["ascii_name"], rec["unicode_name"], rec["address"],
                    rec["name_rank"], rec["unicode_name_rank"], rec["address_rank"],
                    evidence,
                ))

        return pd.DataFrame(rows, columns=[
            "source1_entity_id", "candidate_id", "ann_similarity",
            "ann_ascii_name_similarity", "ann_unicode_name_similarity",
            "ann_address_similarity", "ann_name_rank", "ann_unicode_name_rank",
            "ann_address_rank", "ann_n_strategies"
        ])


def build_ann_indexes(s2, s3, index_dir, cfg, force=False):
    i2 = ANNSourceIndex(s2, Path(index_dir) / "S2", "S2", cfg)
    i3 = ANNSourceIndex(s3, Path(index_dir) / "S3", "S3", cfg)
    i2.build(force)
    i3.build(force)
    return i2, i3
