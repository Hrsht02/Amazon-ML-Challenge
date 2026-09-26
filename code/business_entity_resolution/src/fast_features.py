"""Vectorized pair feature construction for large candidate batches."""
from __future__ import annotations
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein, JaroWinkler
from .features import get_feature_names
from .normalization import char_ngrams

ANN_FEATURES = [
    "ann_similarity", "ann_ascii_name_similarity", "ann_unicode_name_similarity",
    "ann_address_similarity", "ann_name_rank", "ann_unicode_name_rank",
    "ann_address_rank", "ann_n_strategies",
]

def _jaccard(a, b):
    if not a and not b: return 0.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0

def _overlap(a, b):
    if not a or not b: return 0.0
    return len(a & b) / min(len(a), len(b))

def _set_metrics(a_sets, b_sets):
    n = len(a_sets)
    jac = np.empty(n, dtype=np.float32)
    ov = np.empty(n, dtype=np.float32)
    shared = np.empty(n, dtype=np.float32)
    for i, (a, b) in enumerate(zip(a_sets, b_sets)):
        inter = len(a & b)
        union = len(a | b)
        jac[i] = inter / union if union else 0.0
        ov[i] = inter / min(len(a), len(b)) if a and b else 0.0
        shared[i] = inter
    return jac, ov, shared

def _cpdist(q, c, scorer, scale=1.0):
    if not q:
        return np.empty(0, dtype=np.float32)
    return (process.cpdist(q, c, scorer=scorer, workers=-1, dtype=np.float32) * scale).astype(np.float32)

def _norm_arrays(records, key):
    return [records[i][key] for i in range(len(records))]

def compute_features_batch_fast(candidates, s1, s2, s3, normalizer, progress_every=50_000):
    if candidates.empty:
        return pd.DataFrame(columns=get_feature_names() + ANN_FEATURES + ["source1_entity_id", "candidate_id"])

    s1i = s1.set_index("entity_id", drop=False)
    allc = pd.concat([s2, s3], ignore_index=True).drop_duplicates("entity_id").set_index("entity_id", drop=False)
    s1_ids = candidates["source1_entity_id"].to_numpy()
    c_ids = candidates["candidate_id"].to_numpy()
    unique_s1 = pd.unique(s1_ids)
    unique_c = pd.unique(c_ids)

    n1 = {i: normalizer.normalize_name(s1i.at[i, "business_name"]) for i in unique_s1}
    a1 = {i: normalizer.normalize_address(s1i.at[i, "business_address"]) for i in unique_s1}
    n2 = {i: normalizer.normalize_name(allc.at[i, "business_name"]) for i in unique_c}
    a2 = {i: normalizer.normalize_address(allc.at[i, "business_address"]) for i in unique_c}

    qn = [n1[i] for i in s1_ids]; cn = [n2[i] for i in c_ids]
    qa = [a1[i] for i in s1_ids]; ca = [a2[i] for i in c_ids]
    out = {}

    # Preserve the exact semantics of the existing feature implementation.
    out["name_exact_raw"] = np.fromiter((x["raw"].lower() == y["raw"].lower() and x["raw"] != "" for x,y in zip(qn,cn)), dtype=np.float32)
    out["name_exact_alnum"] = np.fromiter((x["alnum"] == y["alnum"] and x["alnum"] != "" for x,y in zip(qn,cn)), dtype=np.float32)
    out["name_exact_suffix_stripped"] = np.fromiter((x["suffix_stripped"] == y["suffix_stripped"] and x["suffix_stripped"] != "" for x,y in zip(qn,cn)), dtype=np.float32)
    out["name_exact_sorted_tokens"] = np.fromiter((x["sorted_tokens"] == y["sorted_tokens"] and x["sorted_tokens"] != "" for x,y in zip(qn,cn)), dtype=np.float32)
    nal = [x["alnum"] for x in qn]; nbl = [y["alnum"] for y in cn]
    out["name_levenshtein_sim"] = _cpdist(nal, nbl, Levenshtein.normalized_similarity, 0.01)
    out["name_jaro_winkler"] = _cpdist(nal, nbl, JaroWinkler.similarity, 0.01)
    out["name_token_sort_ratio"] = _cpdist(nal, nbl, fuzz.token_sort_ratio, 0.01)
    out["name_token_set_ratio"] = _cpdist(nal, nbl, fuzz.token_set_ratio, 0.01)
    out["name_partial_ratio"] = _cpdist(nal, nbl, fuzz.partial_ratio, 0.01)
    nsets = [set(x["tokens"]) for x in qn]; cnsets = [set(x["tokens"]) for x in cn]
    out["name_token_jaccard"], out["name_token_overlap_coeff"], out["name_shared_token_count"] = _set_metrics(nsets, cnsets)
    ncore = [x["core_token_set"] for x in qn]; ccore = [y["core_token_set"] for y in cn]
    out["name_core_token_jaccard"], out["name_core_token_overlap_coeff"], _ = _set_metrics(ncore, ccore)
    out["name_char_ngram_jaccard"] = np.fromiter((_jaccard(char_ngrams(x["alnum"]), char_ngrams(y["alnum"])) for x,y in zip(qn,cn)), dtype=np.float32)
    nlen = np.fromiter((len(x["alnum"]) for x in qn), dtype=np.float32); clen = np.fromiter((len(x["alnum"]) for x in cn), dtype=np.float32)
    out["name_len_ratio"] = np.divide(np.minimum(nlen,clen), np.maximum(nlen,clen), out=np.zeros(len(candidates),dtype=np.float32), where=np.maximum(nlen,clen)>0)
    out["name_len_diff"] = np.abs(nlen-clen).astype(np.float32)
    out["name_prefix4_match"] = np.fromiter((x["alnum"][:4] == y["alnum"][:4] and len(x["alnum"]) >= 4 for x,y in zip(qn,cn)), dtype=np.float32)
    out["name_suffix4_match"] = np.fromiter((x["alnum"][-4:] == y["alnum"][-4:] and len(x["alnum"]) >= 4 for x,y in zip(qn,cn)), dtype=np.float32)
    out["name_either_non_latin"] = np.fromiter((bool(x.get("is_non_latin") or y.get("is_non_latin")) for x,y in zip(qn,cn)), dtype=np.float32)
    out["name_both_non_latin"] = np.fromiter((bool(x.get("is_non_latin") and y.get("is_non_latin")) for x,y in zip(qn,cn)), dtype=np.float32)

    aal = [x["alnum"] for x in qa]; abl = [y["alnum"] for y in ca]
    out["addr_either_empty"] = np.fromiter((bool(x["is_empty"] or y["is_empty"]) for x,y in zip(qa,ca)), dtype=np.float32)
    out["addr_exact_alnum"] = np.fromiter((x["alnum"] == y["alnum"] and x["alnum"] != "" for x,y in zip(qa,ca)), dtype=np.float32)
    out["addr_levenshtein_sim"] = _cpdist(aal, abl, Levenshtein.normalized_similarity, 0.01)
    out["addr_token_sort_ratio"] = _cpdist(aal, abl, fuzz.token_sort_ratio, 0.01)
    out["addr_token_set_ratio"] = _cpdist(aal, abl, fuzz.token_set_ratio, 0.01)
    out["addr_partial_ratio"] = _cpdist(aal, abl, fuzz.partial_ratio, 0.01)
    asets = [set(x["tokens"]) for x in qa]; bssets = [set(y["tokens"]) for y in ca]
    out["addr_token_jaccard"], out["addr_token_overlap_coeff"], out["addr_shared_token_count"] = _set_metrics(asets, bssets)
    acore = [x["core_token_set"] for x in qa]; bcore = [y["core_token_set"] for y in ca]
    out["addr_core_token_jaccard"], _, _ = _set_metrics(acore, bcore)
    out["addr_char_ngram_jaccard"] = np.fromiter((_jaccard(char_ngrams(x["alnum"]), char_ngrams(y["alnum"])) for x,y in zip(qa,ca)), dtype=np.float32)
    d1 = [x["digit_tokens"] for x in qa]; d2 = [y["digit_tokens"] for y in ca]
    out["addr_digit_jaccard"] = np.fromiter((_jaccard(x,y) for x,y in zip(d1,d2)), dtype=np.float32)
    out["addr_digit_signature_exact"] = np.fromiter((x["digit_signature"] == y["digit_signature"] and x["digit_signature"] != "" for x,y in zip(qa,ca)), dtype=np.float32)
    out["addr_has_shared_digit_token"] = np.fromiter((bool(x & y) for x,y in zip(d1,d2)), dtype=np.float32)
    alen = np.fromiter((len(x["alnum"]) for x in qa), dtype=np.float32); blen = np.fromiter((len(x["alnum"]) for x in ca), dtype=np.float32)
    out["addr_len_ratio"] = np.divide(np.minimum(alen,blen), np.maximum(alen,blen), out=np.zeros(len(candidates),dtype=np.float32), where=np.maximum(alen,blen)>0)
    out["addr_either_non_latin"] = np.fromiter((bool(x.get("is_non_latin") or y.get("is_non_latin")) for x,y in zip(qa,ca)), dtype=np.float32)

    c1 = [str(s1i.at[i, "country"]).strip().lower() for i in s1_ids]
    c2 = [str(allc.at[i, "country"]).strip().lower() for i in c_ids]
    out["country_exact"] = np.fromiter((bool(x) and x==y for x,y in zip(c1,c2)), dtype=np.float32)
    out["country_known_both"] = np.fromiter((bool(x) and bool(y) for x,y in zip(c1,c2)), dtype=np.float32)
    out["country_mismatch"] = np.fromiter((bool(x) and bool(y) and x!=y for x,y in zip(c1,c2)), dtype=np.float32)
    ns = out["name_token_sort_ratio"]; ass = out["addr_token_sort_ratio"]
    out["cross_name_x_addr"] = ns * ass
    out["cross_name_plus_addr"] = 0.5*(ns+ass)
    out["cross_name_high_addr_high"] = ((ns>0.8)&(ass>0.6)).astype(np.float32)
    out["cross_name_high_addr_low"] = ((ns>0.8)&(ass<0.3)).astype(np.float32)
    out["cross_name_low_addr_high"] = ((ns<0.4)&(ass>0.8)).astype(np.float32)
    out["cross_name_exact_country_match"] = ((out["name_exact_alnum"]==1)&(out["country_exact"]==1)).astype(np.float32)
    out["cross_addr_digit_and_name_sim"] = out["addr_has_shared_digit_token"] * ns
    out["cross_min_name_addr"] = np.minimum(ns,ass)
    out["cross_max_name_addr"] = np.maximum(ns,ass)
    out["blocking_n_strategies"] = candidates["ann_n_strategies"].to_numpy(dtype=np.float32, copy=False) if "ann_n_strategies" in candidates else np.zeros(len(candidates),dtype=np.float32)

    result = pd.DataFrame(out)
    for col in ANN_FEATURES:
        if col in candidates:
            result[col] = candidates[col].to_numpy(dtype=np.float32, copy=False)
    result["source1_entity_id"] = s1_ids
    result["candidate_id"] = c_ids
    numeric = result.columns.difference(["source1_entity_id","candidate_id"], sort=False)
    result[numeric] = result[numeric].astype("float32")
    return result[get_feature_names() + [c for c in ANN_FEATURES if c in result.columns] + ["source1_entity_id","candidate_id"]]