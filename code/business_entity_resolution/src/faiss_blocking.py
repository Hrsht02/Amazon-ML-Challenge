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
import time
from concurrent.futures import ThreadPoolExecutor
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
        self._gpu_count = 1

    def _new(self):
        q = faiss.IndexFlatIP(self.cfg.ann_dim)
        base = faiss.IndexIVFPQ(
            q, self.cfg.ann_dim, self.cfg.ann_nlist,
            self.cfg.ann_pq_m, 8, faiss.METRIC_INNER_PRODUCT
        )
        base.nprobe = self.cfg.ann_nprobe
        return base

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
            "train_size": self.cfg.ann_train_size,
            "version": 5,
        }

    def _final_paths(self):
        return {
            "name": self.index_dir / f"{self.source}_name.faiss",
            "unicode": self.index_dir / f"{self.source}_name_unicode.faiss",
            "addr": self.index_dir / f"{self.source}_address.faiss",
            "meta": self.index_dir / f"{self.source}_meta.pkl",
        }

    def _partial_paths(self):
        return {
            "name": self.index_dir / f"{self.source}_name.partial.faiss",
            "unicode": self.index_dir / f"{self.source}_name_unicode.partial.faiss",
            "addr": self.index_dir / f"{self.source}_address.partial.faiss",
            "state": self.index_dir / f"{self.source}_build_state.pkl",
        }

    def _gpu_resources(self, gpu_id):
        if not hasattr(faiss, "StandardGpuResources"):
            return None
        res = faiss.StandardGpuResources()
        try:
            res.setTempMemory(512 * 1024 * 1024)
        except Exception:
            pass
        return res

    def _to_gpu(self, cpu_index, gpu_id, res):
        if res is None:
            return cpu_index
        try:
            return faiss.index_cpu_to_gpu(res, int(gpu_id), cpu_index)
        except Exception as e:
            print(f"[FAISS][{self.source}] GPU {gpu_id} build fallback -> CPU: {type(e).__name__}: {e}", flush=True)
            return cpu_index

    def _to_cpu(self, index):
        if type(index).__name__.startswith("Gpu") and hasattr(faiss, "index_gpu_to_cpu"):
            return faiss.index_gpu_to_cpu(index)
        return index

    def _set_nprobe_single(self, idx):
        try:
            base = faiss.downcast_index(idx)
            base.nprobe = int(self.cfg.ann_nprobe)
            return
        except Exception:
            pass
        try:
            base = faiss.downcast_index(idx.index)
            base.nprobe = int(self.cfg.ann_nprobe)
        except Exception:
            pass

    def _write_checkpoint(self, indexes, paths, rows_done, expected):
        t0 = time.time()
        for key in ("name", "unicode", "addr"):
            cpu_idx = self._to_cpu(indexes[key])
            self._set_nprobe_single(cpu_idx)
            tmp = paths[key].with_suffix(paths[key].suffix + ".tmp")
            faiss.write_index(cpu_idx, str(tmp))
            tmp.replace(paths[key])
        state = {"rows_added": int(rows_done), "rows": len(self.df), "version": 5, "meta": expected}
        tmp = paths["state"].with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(state, f)
        tmp.replace(paths["state"])
        print(f"[FAISS][{self.source}] CHECKPOINT {rows_done:,}/{len(self.df):,} saved in {time.time()-t0:.1f}s", flush=True)

    def build(self, force=False, gpu_id=0):
        paths = self._final_paths()
        partial = self._partial_paths()
        expected = self._meta()
        keys = ("rows", "dim", "nlist", "pq_m", "train_size")

        if not force and all(p.exists() for p in paths.values()):
            with open(paths["meta"], "rb") as f:
                meta = pickle.load(f)
            if all(meta.get(k) == expected.get(k) for k in keys) and int(meta.get("version", 0)) in (3, 4, 5):
                print(f"[FAISS][{self.source}] Reusing cached indexes", flush=True)
                self.name = faiss.read_index(str(paths["name"]))
                self.name_unicode = faiss.read_index(str(paths["unicode"]))
                self.addr = faiss.read_index(str(paths["addr"]))
                self._set_nprobe_cpu()
                return

        n = len(self.df)
        bs = int(self.cfg.index_build_batch_size)
        checkpoint_rows = max(bs, int(getattr(self.cfg, "index_checkpoint_rows", 500_000)))
        ngpu = int(getattr(faiss, "get_num_gpus", lambda: 0)())
        gpu_ok = bool(getattr(self.cfg, "use_faiss_gpu", True) and ngpu > gpu_id and hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu"))

        print(f"[FAISS][{self.source}] START rows={n:,} GPU={gpu_ok} device={gpu_id} visible_gpus={ngpu} batch={bs:,} checkpoint={checkpoint_rows:,}", flush=True)

        rows_done = 0
        cpu_indexes = {}
        if not force and all(p.exists() for p in partial.values()):
            try:
                with open(partial["state"], "rb") as f:
                    state = pickle.load(f)
                rows_done = int(state["rows_added"])
                cpu_indexes = {k: faiss.read_index(str(partial[k])) for k in ("name", "unicode", "addr")}
                if state.get("rows") != n or state.get("version") != 5 or any(cpu_indexes[k].ntotal != rows_done for k in cpu_indexes):
                    raise ValueError("checkpoint metadata/index size mismatch")
                if any(state.get("meta", {}).get(k) != expected.get(k) for k in keys):
                    raise ValueError("checkpoint configuration mismatch")
                print(f"[FAISS][{self.source}] RESUME from {rows_done:,}/{n:,}", flush=True)
            except Exception as e:
                print(f"[FAISS][{self.source}] checkpoint ignored: {type(e).__name__}: {e}", flush=True)
                rows_done, cpu_indexes = 0, {}

        res = self._gpu_resources(gpu_id) if gpu_ok else None

        if rows_done == 0:
            self.name, self.name_unicode, self.addr = self._new(), self._new(), self._new()
            rng = np.random.default_rng(42 + self.cfg.ann_nlist + self.cfg.ann_pq_m)
            take = min(n, self.cfg.ann_train_size)
            sample = rng.choice(n, take, replace=False) if n > take else np.arange(n)
            print(f"[FAISS][{self.source}] TRAIN 3 indexes using {take:,} vectors", flush=True)
            names = self.df.iloc[sample].business_name.fillna("").astype(str).tolist()
            addrs = self.df.iloc[sample].business_address.fillna("").astype(str).tolist()

            t0 = time.time(); x = self._vec(names, False)
            self.name = self._to_gpu(self.name, gpu_id, res); self.name.train(x)
            print(f"[FAISS][{self.source}] ASCII-name trained in {time.time()-t0:.1f}s", flush=True); del x

            t0 = time.time(); x = self._vec(names, True)
            self.name_unicode = self._to_gpu(self.name_unicode, gpu_id, res); self.name_unicode.train(x)
            print(f"[FAISS][{self.source}] Unicode-name trained in {time.time()-t0:.1f}s", flush=True); del x

            t0 = time.time(); x = self._vec(addrs, False)
            self.addr = self._to_gpu(self.addr, gpu_id, res); self.addr.train(x)
            print(f"[FAISS][{self.source}] Address trained in {time.time()-t0:.1f}s", flush=True); del x, names, addrs, sample
        else:
            self.name = self._to_gpu(cpu_indexes["name"], gpu_id, res)
            self.name_unicode = self._to_gpu(cpu_indexes["unicode"], gpu_id, res)
            self.addr = self._to_gpu(cpu_indexes["addr"], gpu_id, res)

        started = time.time()
        for s in range(rows_done, n, bs):
            e = min(s + bs, n)
            t_batch = time.time()
            ids = np.arange(s, e, dtype="int64")

            t0 = time.time(); x = self._vec(self._texts("business_name", s, e), False); v1 = time.time()-t0
            t0 = time.time(); self.name.add_with_ids(x, ids); a1 = time.time()-t0; del x

            t0 = time.time(); x = self._vec(self._texts("business_name", s, e, True), True); v2 = time.time()-t0
            t0 = time.time(); self.name_unicode.add_with_ids(x, ids); a2 = time.time()-t0; del x

            t0 = time.time(); x = self._vec(self._texts("business_address", s, e), False); v3 = time.time()-t0
            t0 = time.time(); self.addr.add_with_ids(x, ids); a3 = time.time()-t0; del x

            elapsed = time.time()-started
            rate = e/max(elapsed, 1e-9)
            eta = (n-e)/max(rate, 1e-9)
            print(f"[FAISS][{self.source}] {e:,}/{n:,} ({100*e/n:6.2f}%) | batch={time.time()-t_batch:.1f}s | vec={v1:.1f}/{v2:.1f}/{v3:.1f}s | add={a1:.1f}/{a2:.1f}/{a3:.1f}s | rate={rate:,.0f} rows/s | ETA={eta/60:.1f}m", flush=True)

            if e % checkpoint_rows == 0 or e == n:
                self._write_checkpoint({"name": self.name, "unicode": self.name_unicode, "addr": self.addr}, partial, e, expected)

        self.name, self.name_unicode, self.addr = self._to_cpu(self.name), self._to_cpu(self.name_unicode), self._to_cpu(self.addr)
        self._set_nprobe_cpu()
        t0 = time.time()
        for key, idx in (("name", self.name), ("unicode", self.name_unicode), ("addr", self.addr)):
            tmp = paths[key].with_suffix(paths[key].suffix + ".tmp")
            faiss.write_index(idx, str(tmp)); tmp.replace(paths[key])
        with open(paths["meta"], "wb") as f:
            pickle.dump(expected, f)
        for p in partial.values():
            try: p.unlink()
            except FileNotFoundError: pass
        print(f"[FAISS][{self.source}] COMPLETE rows={n:,} final_save={time.time()-t0:.1f}s", flush=True)

    def _set_nprobe_cpu(self):
        """Apply the current search-time nprobe to loaded CPU indexes."""
        for idx in (self.name, self.name_unicode, self.addr):
            try:
                base = faiss.downcast_index(idx.index)
                base.nprobe = int(self.cfg.ann_nprobe)
            except Exception:
                pass

    def _enable_gpu(self):
        if not self._gpu_enabled or self._gpu_failed or self._gpu is not None:
            return
        try:
            if not hasattr(faiss, "StandardGpuResources"):
                print("[FAISS] GPU FAISS bindings unavailable; using CPU", flush=True)
                self._gpu_failed = True
                return
            self._set_nprobe_cpu()
            ngpu = int(getattr(faiss, "get_num_gpus", lambda: 0)())
            self._gpu_count = max(1, ngpu)
            if ngpu >= 2 and hasattr(faiss, "index_cpu_to_all_gpus"):
                self._gpu = {
                    "name": faiss.index_cpu_to_all_gpus(self.name),
                    "unicode": faiss.index_cpu_to_all_gpus(self.name_unicode),
                    "addr": faiss.index_cpu_to_all_gpus(self.addr),
                }
                print(f"[FAISS] Multi-GPU search enabled: {ngpu} GPUs", flush=True)
            else:
                res = faiss.StandardGpuResources()
                self._gpu = {
                    "res": res,
                    "name": faiss.index_cpu_to_gpu(res, 0, self.name),
                    "unicode": faiss.index_cpu_to_gpu(res, 0, self.name_unicode),
                    "addr": faiss.index_cpu_to_gpu(res, 0, self.addr),
                }
                print("[FAISS] GPU search enabled: GPU 0", flush=True)
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
                        "ascii_name_rank": 10**9,
                        "unicode_name_rank": 10**9,
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
                    rec["ascii_name_rank"], rec["unicode_name_rank"], rec["address_rank"],
                    evidence,
                ))

        return pd.DataFrame(rows, columns=[
            "source1_entity_id", "candidate_id", "ann_similarity",
            "ann_ascii_name_similarity", "ann_unicode_name_similarity",
            "ann_address_similarity", "ann_name_rank", "ann_unicode_name_rank",
            "ann_address_rank", "ann_n_strategies"
        ])


def build_ann_indexes(s2, s3, index_dir, cfg, force=False):
    root = Path(index_dir)
    i2 = ANNSourceIndex(s2, root / "S2", "S2", cfg)
    i3 = ANNSourceIndex(s3, root / "S3", "S3", cfg)
    ngpu = int(getattr(faiss, "get_num_gpus", lambda: 0)())
    if bool(getattr(cfg, "use_faiss_gpu", True)) and ngpu >= 2:
        print("[FAISS] FAST MODE: S2 -> GPU 0, S3 -> GPU 1, concurrent build", flush=True)
        with ThreadPoolExecutor(max_workers=2) as ex:
            f2 = ex.submit(i2.build, force, 0)
            f3 = ex.submit(i3.build, force, 1)
            f2.result(); f3.result()
    else:
        print(f"[FAISS] FAST MODE: {ngpu} GPU(s) available; sequential source build", flush=True)
        i2.build(force, 0); i3.build(force, 0)
    return i2, i3
