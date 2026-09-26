"""Fast, memory-bounded pair feature computation."""
from __future__ import annotations
import time
import pandas as pd
from .fast_features import compute_features_batch_fast

def compute_features_batch(candidates, s1, s2, s3, normalizer, progress_every: int = 50_000):
    t0 = time.time()
    print(f"[FEATURES] Fast vectorized construction for {len(candidates):,} pair features...", flush=True)
    df = compute_features_batch_fast(
        candidates, s1, s2, s3, normalizer, progress_every=progress_every
    )
    print(f"[FEATURES] Completed {len(df):,} rows in {time.time()-t0:.1f}s", flush=True)
    return df
