"""
Multi-strategy candidate generation ("blocking").

Blocking determines the recall ceiling: any true match a blocker never
proposes can never be recovered by the scoring model. So we run several
independent, cheap strategies and take their UNION, rather than trusting
one. Each strategy returns, per S1 entity, a ranked top-k list of S2/S3
entity_ids plus which strategy proposed them (kept as a feature).

Strategies implemented (see BlockingConfig for the top-k of each):
  A. exact normalized-name match              (deterministic, high precision)
  B. name char n-gram TF-IDF cosine, top-k     (handles typos/abbreviations)
  C. address char n-gram TF-IDF cosine, top-k  (handles address noise)
  D. token inverted index on core name tokens  (handles word reordering)
  E. rare-token inverted index (name+address)  (handles distinctive tokens
                                                 surviving heavy corruption)
  F. digit-signature exact match on address    (house/PIN numbers)
  G. country bucket (soft prior, used only to break ties / cap fan-out,
     NEVER used as a hard filter — France and unseen countries must not
     be excluded)

All strategies operate per-source (S1 vs S2, S1 vs S3 run independently)
so source-specific noise doesn't get diluted, then results are unioned.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .config import BlockingConfig
from .normalization import Normalizer, char_ngrams


class SourceIndex:
    """Precomputed representations + TF-IDF/NN indexes for one source
    (S2 or S3), built once and reused across every S1 query."""

    def __init__(self, df: pd.DataFrame, normalizer: Normalizer, cfg: BlockingConfig):
        self.df = df
        self.ids = df["entity_id"].tolist()
        self.cfg = cfg
        self.norm_name = {eid: normalizer.normalize_name(row.business_name)
                           for eid, row in df.iterrows()}
        self.norm_addr = {eid: normalizer.normalize_address(row.business_address)
                           for eid, row in df.iterrows()}

        # TF-IDF (char n-gram) indexes for name and address
        name_texts = [self.norm_name[i]["alnum"] for i in self.ids]
        addr_texts = [self.norm_addr[i]["alnum"] for i in self.ids]

        self.name_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=cfg.char_ngram_range, min_df=1)
        self.addr_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=cfg.char_ngram_range, min_df=1)

        self._name_valid = [i for i, t in enumerate(name_texts) if t]
        self._addr_valid = [i for i, t in enumerate(addr_texts) if t]

        if self._name_valid:
            self.name_X = self.name_vec.fit_transform([name_texts[i] for i in self._name_valid])
            self.name_nn = NearestNeighbors(metric="cosine",
                                             n_neighbors=min(cfg.topk_name_tfidf, len(self._name_valid))).fit(self.name_X)
        else:
            self.name_X, self.name_nn = None, None

        if self._addr_valid:
            self.addr_X = self.addr_vec.fit_transform([addr_texts[i] for i in self._addr_valid])
            self.addr_nn = NearestNeighbors(metric="cosine",
                                             n_neighbors=min(cfg.topk_address_tfidf, len(self._addr_valid))).fit(self.addr_X)
        else:
            self.addr_X, self.addr_nn = None, None

        # exact-normalized-name -> [ids]
        self.exact_name_index: Dict[str, List[str]] = defaultdict(list)
        for i in self.ids:
            key = self.norm_name[i]["suffix_stripped"]
            if key:
                self.exact_name_index[key].append(i)

        # token inverted index (core tokens, name)
        self.name_token_index: Dict[str, List[str]] = defaultdict(list)
        for i in self.ids:
            for tok in self.norm_name[i]["core_token_set"]:
                self.name_token_index[tok].append(i)

        # rare-token index over BOTH name and address core tokens
        token_doc_count = defaultdict(int)
        doc_tokens: Dict[str, Set[str]] = {}
        for i in self.ids:
            toks = self.norm_name[i]["core_token_set"] | self.norm_addr[i]["core_token_set"]
            doc_tokens[i] = toks
            for t in toks:
                token_doc_count[t] += 1
        self.rare_tokens = {t for t, c in token_doc_count.items()
                             if c <= cfg.rare_token_max_df and len(t) >= cfg.min_token_len}
        self.rare_token_index: Dict[str, List[str]] = defaultdict(list)
        for i in self.ids:
            for t in (doc_tokens[i] & self.rare_tokens):
                self.rare_token_index[t].append(i)

        # digit signature -> [ids]
        self.digit_sig_index: Dict[str, List[str]] = defaultdict(list)
        for i in self.ids:
            sig = self.norm_addr[i]["digit_signature"]
            if sig:
                self.digit_sig_index[sig].append(i)


def _nn_lookup(nn: NearestNeighbors, X_query, valid_pos_to_id: List[str], vec_matrix_ids: List[str], k: int):
    if nn is None or X_query is None or X_query.nnz == 0:
        return []
    k = min(k, vec_matrix_ids and len(vec_matrix_ids) or 0)
    if k == 0:
        return []
    dist, idx = nn.kneighbors(X_query, n_neighbors=k)
    out = []
    for d, ix in zip(dist[0], idx[0]):
        sim = 1.0 - d
        out.append((vec_matrix_ids[ix], sim))
    return out


def generate_candidates_for_entity(
    s1_id: str, name_repr: dict, addr_repr: dict, country: str,
    index: SourceIndex, normalizer: Normalizer, cfg: BlockingConfig,
) -> Dict[str, Set[str]]:
    """Returns {candidate_id: {strategy_names that proposed it}}."""
    proposals: Dict[str, Set[str]] = defaultdict(set)

    # A. exact normalized name
    key = name_repr["suffix_stripped"]
    if key and key in index.exact_name_index:
        for cid in index.exact_name_index[key]:
            proposals[cid].add("exact_name")

    # B. name TF-IDF char n-gram cosine
    if index.name_nn is not None and name_repr["alnum"]:
        q = index.name_vec.transform([name_repr["alnum"]])
        vm_ids = [index.ids[i] for i in index._name_valid]
        for cid, sim in _nn_lookup(index.name_nn, q, None, vm_ids, cfg.topk_name_tfidf):
            if sim > 0:
                proposals[cid].add("name_tfidf")

    # C. address TF-IDF char n-gram cosine
    if index.addr_nn is not None and addr_repr["alnum"]:
        q = index.addr_vec.transform([addr_repr["alnum"]])
        vm_ids = [index.ids[i] for i in index._addr_valid]
        for cid, sim in _nn_lookup(index.addr_nn, q, None, vm_ids, cfg.topk_address_tfidf):
            if sim > 0:
                proposals[cid].add("addr_tfidf")

    # D. token inverted index (core name tokens) — count overlaps, keep top-k
    tok_overlap_counts = defaultdict(int)
    for tok in name_repr["core_token_set"]:
        for cid in index.name_token_index.get(tok, []):
            tok_overlap_counts[cid] += 1
    for cid, _ in sorted(tok_overlap_counts.items(), key=lambda kv: -kv[1])[:cfg.topk_name_char_ngram]:
        proposals[cid].add("name_token_overlap")

    # E. rare-token index (name + address)
    rare_hits = set()
    for tok in (name_repr["core_token_set"] | addr_repr["core_token_set"]):
        if tok in index.rare_tokens:
            rare_hits.update(index.rare_token_index.get(tok, []))
    for cid in rare_hits:
        proposals[cid].add("rare_token")

    # F. digit signature exact match
    sig = addr_repr["digit_signature"]
    if sig and sig in index.digit_sig_index:
        for cid in index.digit_sig_index[sig]:
            proposals[cid].add("digit_signature")

    return proposals


def build_candidates(s1_df: pd.DataFrame, s2_index: SourceIndex, s3_index: SourceIndex,
                      normalizer: Normalizer, cfg: BlockingConfig) -> pd.DataFrame:
    """Returns a long dataframe: source1_entity_id, candidate_id, strategies(set)"""
    rows = []
    for s1_id, row in s1_df.iterrows():
        name_repr = normalizer.normalize_name(row.business_name)
        addr_repr = normalizer.normalize_address(row.business_address)
        all_props: Dict[str, Set[str]] = defaultdict(set)
        for idx in (s2_index, s3_index):
            props = generate_candidates_for_entity(s1_id, name_repr, addr_repr,
                                                     row.country, idx, normalizer, cfg)
            for cid, strategies in props.items():
                all_props[cid] |= strategies

        # cap fan-out: keep the candidates with the most corroborating strategies
        # (ties broken by insertion order); this bounds compute for the feature
        # + scoring stage without touching recall for well-evidenced candidates.
        ranked = sorted(all_props.items(), key=lambda kv: -len(kv[1]))
        ranked = ranked[:cfg.max_candidates_per_s1]

        for cid, strategies in ranked:
            rows.append({
                "source1_entity_id": s1_id,
                "candidate_id": cid,
                "n_strategies": len(strategies),
                "strategies": "|".join(sorted(strategies)),
            })
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_id",
                                        "n_strategies", "strategies"])
