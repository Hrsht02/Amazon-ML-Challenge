"""
Pairwise feature engineering for a candidate (S1, candidate) pair.

Features are grouped: name, address, cross-field, country, blocking
evidence. Every string metric is computed on the ALNUM (ASCII-folded,
punctuation-stripped) representation so it degrades gracefully on the
non-Latin-script rows identified in the audit — a same-script exact
match still scores 1.0, a script-mismatched pair naturally collapses
toward 0 on the alnum channel, and `is_non_latin` flags let the model
learn to trust the address/digit channel more in that situation instead
of the name channel.
"""
from __future__ import annotations
from typing import Dict

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler

from .normalization import char_ngrams


def _safe_ratio(fn, a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fn(a, b) / 100.0


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    u = a | b
    if not u:
        return 0.0
    return len(a & b) / len(u)


def _overlap_coeff(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def name_features(n1: dict, n2: dict) -> Dict[str, float]:
    f = {}
    f["name_exact_raw"] = float(n1["raw"].lower() == n2["raw"].lower() and n1["raw"] != "")
    f["name_exact_alnum"] = float(n1["alnum"] == n2["alnum"] and n1["alnum"] != "")
    f["name_exact_suffix_stripped"] = float(
        n1["suffix_stripped"] == n2["suffix_stripped"] and n1["suffix_stripped"] != "")
    f["name_exact_sorted_tokens"] = float(
        n1["sorted_tokens"] == n2["sorted_tokens"] and n1["sorted_tokens"] != "")

    f["name_levenshtein_sim"] = _safe_ratio(Levenshtein.normalized_similarity, n1["alnum"], n2["alnum"])
    f["name_jaro_winkler"] = _safe_ratio(JaroWinkler.similarity, n1["alnum"], n2["alnum"])
    f["name_token_sort_ratio"] = _safe_ratio(fuzz.token_sort_ratio, n1["alnum"], n2["alnum"])
    f["name_token_set_ratio"] = _safe_ratio(fuzz.token_set_ratio, n1["alnum"], n2["alnum"])
    f["name_partial_ratio"] = _safe_ratio(fuzz.partial_ratio, n1["alnum"], n2["alnum"])

    t1, t2 = set(n1["tokens"]), set(n2["tokens"])
    f["name_token_jaccard"] = _jaccard(t1, t2)
    f["name_token_overlap_coeff"] = _overlap_coeff(t1, t2)
    f["name_shared_token_count"] = float(len(t1 & t2))

    c1, c2 = n1["core_token_set"], n2["core_token_set"]
    f["name_core_token_jaccard"] = _jaccard(c1, c2)
    f["name_core_token_overlap_coeff"] = _overlap_coeff(c1, c2)

    g1, g2 = char_ngrams(n1["alnum"]), char_ngrams(n2["alnum"])
    f["name_char_ngram_jaccard"] = _jaccard(g1, g2)

    len1, len2 = len(n1["alnum"]), len(n2["alnum"])
    f["name_len_ratio"] = (min(len1, len2) / max(len1, len2)) if max(len1, len2) > 0 else 0.0
    f["name_len_diff"] = abs(len1 - len2)

    f["name_prefix4_match"] = float(n1["alnum"][:4] == n2["alnum"][:4] and len(n1["alnum"]) >= 4)
    f["name_suffix4_match"] = float(n1["alnum"][-4:] == n2["alnum"][-4:] and len(n1["alnum"]) >= 4)

    f["name_either_non_latin"] = float(n1.get("is_non_latin") or n2.get("is_non_latin"))
    f["name_both_non_latin"] = float(n1.get("is_non_latin") and n2.get("is_non_latin"))
    return f


def address_features(a1: dict, a2: dict) -> Dict[str, float]:
    f = {}
    f["addr_either_empty"] = float(a1["is_empty"] or a2["is_empty"])
    f["addr_exact_alnum"] = float(a1["alnum"] == a2["alnum"] and a1["alnum"] != "")

    f["addr_levenshtein_sim"] = _safe_ratio(Levenshtein.normalized_similarity, a1["alnum"], a2["alnum"])
    f["addr_token_sort_ratio"] = _safe_ratio(fuzz.token_sort_ratio, a1["alnum"], a2["alnum"])
    f["addr_token_set_ratio"] = _safe_ratio(fuzz.token_set_ratio, a1["alnum"], a2["alnum"])
    f["addr_partial_ratio"] = _safe_ratio(fuzz.partial_ratio, a1["alnum"], a2["alnum"])

    t1, t2 = set(a1["tokens"]), set(a2["tokens"])
    f["addr_token_jaccard"] = _jaccard(t1, t2)
    f["addr_token_overlap_coeff"] = _overlap_coeff(t1, t2)
    f["addr_shared_token_count"] = float(len(t1 & t2))

    c1, c2 = a1["core_token_set"], a2["core_token_set"]
    f["addr_core_token_jaccard"] = _jaccard(c1, c2)

    g1, g2 = char_ngrams(a1["alnum"]), char_ngrams(a2["alnum"])
    f["addr_char_ngram_jaccard"] = _jaccard(g1, g2)

    d1, d2 = a1["digit_tokens"], a2["digit_tokens"]
    f["addr_digit_jaccard"] = _jaccard(d1, d2)
    f["addr_digit_signature_exact"] = float(
        a1["digit_signature"] == a2["digit_signature"] and a1["digit_signature"] != "")
    f["addr_has_shared_digit_token"] = float(len(d1 & d2) > 0)

    len1, len2 = len(a1["alnum"]), len(a2["alnum"])
    f["addr_len_ratio"] = (min(len1, len2) / max(len1, len2)) if max(len1, len2) > 0 else 0.0

    f["addr_either_non_latin"] = float(a1.get("is_non_latin") or a2.get("is_non_latin"))
    return f


def country_features(c1: str, c2: str) -> Dict[str, float]:
    c1n, c2n = (c1 or "").strip().lower(), (c2 or "").strip().lower()
    return {
        "country_exact": float(c1n == c2n and c1n != ""),
        "country_known_both": float(bool(c1n) and bool(c2n)),
        "country_mismatch": float(bool(c1n) and bool(c2n) and c1n != c2n),
    }


def cross_features(nf: Dict[str, float], af: Dict[str, float], cf: Dict[str, float]) -> Dict[str, float]:
    name_sim = nf["name_token_sort_ratio"]
    addr_sim = af["addr_token_sort_ratio"]
    f = {}
    f["cross_name_x_addr"] = name_sim * addr_sim
    f["cross_name_plus_addr"] = 0.5 * (name_sim + addr_sim)
    f["cross_name_high_addr_high"] = float(name_sim > 0.8 and addr_sim > 0.6)
    f["cross_name_high_addr_low"] = float(name_sim > 0.8 and addr_sim < 0.3)
    f["cross_name_low_addr_high"] = float(name_sim < 0.4 and addr_sim > 0.8)
    f["cross_name_exact_country_match"] = float(nf["name_exact_alnum"] == 1.0 and cf["country_exact"] == 1.0)
    f["cross_addr_digit_and_name_sim"] = af["addr_has_shared_digit_token"] * name_sim
    f["cross_min_name_addr"] = min(name_sim, addr_sim)
    f["cross_max_name_addr"] = max(name_sim, addr_sim)
    return f


def compute_pair_features(name1: dict, addr1: dict, country1: str,
                           name2: dict, addr2: dict, country2: str,
                           n_strategies: int = 0) -> Dict[str, float]:
    nf = name_features(name1, name2)
    af = address_features(addr1, addr2)
    cf = country_features(country1, country2)
    xf = cross_features(nf, af, cf)
    feats = {}
    feats.update(nf)
    feats.update(af)
    feats.update(cf)
    feats.update(xf)
    feats["blocking_n_strategies"] = float(n_strategies)
    return feats


FEATURE_NAMES = None  # populated lazily by get_feature_names()


def get_feature_names() -> list:
    global FEATURE_NAMES
    if FEATURE_NAMES is None:
        from .normalization import Normalizer
        norm = Normalizer()
        n = norm.normalize_name("Example Corp")
        a = norm.normalize_address("123 Main St")
        feats = compute_pair_features(n, a, "US", n, a, "US", n_strategies=1)
        FEATURE_NAMES = sorted(feats.keys())
    return FEATURE_NAMES
