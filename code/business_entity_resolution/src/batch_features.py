"""Memory-bounded pair feature computation with visible progress."""
from __future__ import annotations
import time
import pandas as pd
from .features import compute_pair_features, get_feature_names
from .normalization import Normalizer


ANN_FEATURES = [
    "ann_similarity", "ann_ascii_name_similarity", "ann_unicode_name_similarity",
    "ann_address_similarity", "ann_name_rank", "ann_unicode_name_rank",
    "ann_address_rank", "ann_n_strategies",
]


def compute_features_batch(candidates, s1, s2, s3, normalizer: Normalizer, progress_every: int = 50_000):
    if candidates.empty:
        return pd.DataFrame(columns=get_feature_names() + ANN_FEATURES + ["source1_entity_id", "candidate_id"])

    print(f"[FEATURES] Computing {len(candidates):,} pair features...", flush=True)
    t0 = time.time()
    s1i = s1.set_index("entity_id", drop=False)
    allc = pd.concat([s2, s3], ignore_index=True).drop_duplicates("entity_id").set_index("entity_id", drop=False)
    ids1 = candidates.source1_entity_id.unique()
    ids2 = candidates.candidate_id.unique()

    n1 = {i: normalizer.normalize_name(s1i.loc[i].business_name) for i in ids1}
    a1 = {i: normalizer.normalize_address(s1i.loc[i].business_address) for i in ids1}
    n2 = {i: normalizer.normalize_name(allc.loc[i].business_name) for i in ids2}
    a2 = {i: normalizer.normalize_address(allc.loc[i].business_address) for i in ids2}

    out = []
    for idx, r in enumerate(candidates.itertuples(index=False), start=1):
        x = compute_pair_features(
            n1[r.source1_entity_id], a1[r.source1_entity_id],
            s1i.loc[r.source1_entity_id].country,
            n2[r.candidate_id], a2[r.candidate_id],
            allc.loc[r.candidate_id].country,
            n_strategies=getattr(r, "ann_n_strategies", 0),
        )
        for col in ANN_FEATURES:
            if col != "ann_n_strategies" and hasattr(r, col):
                x[col] = float(getattr(r, col))
        x["source1_entity_id"] = r.source1_entity_id
        x["candidate_id"] = r.candidate_id
        out.append(x)
        if idx % progress_every == 0 or idx == len(candidates):
            print(
                f"[FEATURES] {idx:,}/{len(candidates):,} "
                f"({idx/len(candidates):.1%}) elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    df = pd.DataFrame(out)
    numeric = [c for c in df.columns if c not in ("source1_entity_id", "candidate_id")]
    df[numeric] = df[numeric].astype("float32")
    return df
