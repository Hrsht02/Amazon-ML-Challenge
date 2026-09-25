"""
Field normalization.

Design principle from the audit: Source 1 is ~100% Latin-script while
Source 2/3 mix in 10-28% non-Latin tokens (Devanagari, Tamil, Bengali,
French accents). A single normalized string throws away information, so
we keep SEVERAL representations per field and let the feature layer / model
decide which signal to trust for a given pair:

  raw            - original string, whitespace-trimmed only
  lower          - casefolded
  ascii_fold     - unicode NFKD + strip combining marks + drop non-ASCII
                    (turns "Métropolitain" -> "Metropolitain", and turns
                    Devanagari/Tamil/Bengali runs into "" - which is itself
                    informative: a script mismatch is a feature)
  alnum          - ascii_fold, punctuation stripped, single-spaced
  tokens         - list of alnum tokens (len >= 2)
  sorted_tokens  - tokens sorted alphabetically, joined (order-invariant compare)
  suffix_stripped- alnum with a trailing legal-entity suffix removed
  core_tokens    - suffix_stripped tokens with generic/very-common business
                   words removed (e.g. "the", "and", "services") — used for
                   rare-token blocking

Legal suffixes are seeded with a small curated multi-country list AND
extended by mining frequent trailing tokens from the training business
names (data-driven, not just hard-coded), so the set generalizes past
US/India without assuming a fixed country set.
"""
from __future__ import annotations
import re
import unicodedata
from collections import Counter
from typing import Iterable, List, Set

# Seed list is intentionally small and multi-jurisdiction; it is EXTENDED
# (not replaced) by mine_legal_suffixes() at pipeline-build time.
SEED_LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "llc", "llp", "lp", "pllc", "ltd", "limited", "plc",
    "pvt", "private", "pl",
    "sarl", "eurl", "sas", "sa", "sasu", "sci",
    "gmbh", "ag", "bv", "nv", "kg", "oy", "ab",
}

GENERIC_BUSINESS_WORDS = {
    "the", "and", "of", "for", "group", "services", "service",
    "solutions", "solution", "enterprises", "enterprise", "industries",
    "industry", "international", "national", "global", "associates",
    "partners", "holdings", "trading", "traders", "store", "stores",
}

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_MULTISPACE_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d+")


def ascii_fold(s: str) -> str:
    if not s:
        return ""
    norm = unicodedata.normalize("NFKD", s)
    return "".join(c for c in norm if not unicodedata.combining(c) and ord(c) < 128)


def basic_clean(s: str) -> str:
    s = (s or "").strip()
    s = unicodedata.normalize("NFKC", s)
    s = _MULTISPACE_RE.sub(" ", s)
    return s.strip()


def to_alnum(s: str, fold_ascii: bool = True) -> str:
    s = ascii_fold(s) if fold_ascii else s
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _MULTISPACE_RE.sub(" ", s)
    return s.strip()


def tokenize(s: str, min_len: int = 2) -> List[str]:
    return [t for t in to_alnum(s).split(" ") if len(t) >= min_len]


def sorted_token_string(tokens: Iterable[str]) -> str:
    return " ".join(sorted(tokens))


def strip_legal_suffix(alnum_name: str, suffixes: Set[str]) -> str:
    toks = alnum_name.split(" ")
    while toks and toks[-1] in suffixes:
        toks = toks[:-1]
    # also handle a suffix stuck as the FIRST token is rare, so only trailing
    return " ".join(toks).strip()


def mine_legal_suffixes(business_names: Iterable[str], min_count: int = 3,
                         max_new: int = 40) -> Set[str]:
    """Learn additional frequent trailing tokens from training names.

    Only tokens that (a) occur frequently as the LAST token of a name and
    (b) are short-ish (legal suffixes are rarely long words) are added, to
    avoid accidentally treating a common real business-name word as a
    suffix.
    """
    last_tok_counts = Counter()
    for name in business_names:
        toks = tokenize(name)
        if toks:
            last_tok_counts[toks[-1]] += 1
    mined = set()
    for tok, cnt in last_tok_counts.most_common():
        if cnt < min_count:
            break
        if tok in SEED_LEGAL_SUFFIXES:
            continue
        if len(tok) <= 6:  # legal suffixes are short abbreviations/words
            mined.add(tok)
        if len(mined) >= max_new:
            break
    return mined


def extract_digit_signature(s: str) -> str:
    """Concatenated digit runs in order — a crude 'numeric fingerprint' of
    an address (house/building numbers, PIN/ZIP if present)."""
    return "".join(_DIGIT_RE.findall(s or ""))


def extract_digit_tokens(s: str) -> Set[str]:
    return set(_DIGIT_RE.findall(s or ""))


def char_ngrams(s: str, n_range=(2, 4)) -> Set[str]:
    s = to_alnum(s).replace(" ", "_")
    grams = set()
    for n in range(n_range[0], n_range[1] + 1):
        if len(s) < n:
            continue
        grams.update(s[i:i + n] for i in range(len(s) - n + 1))
    return grams


class Normalizer:
    """Bundles the suffix set (seeded + mined) and exposes a single method
    that builds every representation for a raw string in one pass."""

    def __init__(self, legal_suffixes: Set[str] = None,
                 generic_words: Set[str] = None):
        self.legal_suffixes = set(SEED_LEGAL_SUFFIXES)
        if legal_suffixes:
            self.legal_suffixes |= legal_suffixes
        self.generic_words = GENERIC_BUSINESS_WORDS if generic_words is None else generic_words

    @classmethod
    def fit(cls, training_business_names: Iterable[str]) -> "Normalizer":
        mined = mine_legal_suffixes(training_business_names)
        return cls(legal_suffixes=mined)

    def normalize_name(self, raw: str) -> dict:
        raw = raw or ""
        alnum = to_alnum(raw)
        tokens = tokenize(raw)
        suffix_stripped = strip_legal_suffix(alnum, self.legal_suffixes)
        core_tokens = [t for t in suffix_stripped.split(" ")
                       if t and t not in self.generic_words]
        return {
            "raw": basic_clean(raw),
            "lower": raw.lower().strip(),
            "ascii_fold": ascii_fold(raw),
            "alnum": alnum,
            "tokens": tokens,
            "sorted_tokens": sorted_token_string(tokens),
            "suffix_stripped": suffix_stripped,
            "core_tokens": core_tokens,
            "core_token_set": set(core_tokens),
            "is_non_latin": bool(raw) and alnum == "" and len(re.sub(r"\s", "", raw)) > 0,
        }

    def normalize_address(self, raw: str) -> dict:
        raw = raw or ""
        alnum = to_alnum(raw)
        tokens = tokenize(raw)
        digit_sig = extract_digit_signature(raw)
        digit_toks = extract_digit_tokens(raw)
        core_tokens = [t for t in tokens if t not in self.generic_words]
        return {
            "raw": basic_clean(raw),
            "alnum": alnum,
            "tokens": tokens,
            "sorted_tokens": sorted_token_string(tokens),
            "core_token_set": set(core_tokens),
            "digit_signature": digit_sig,
            "digit_tokens": digit_toks,
            "is_empty": len(raw.strip()) == 0,
            "is_non_latin": bool(raw) and alnum == "" and len(re.sub(r"\s", "", raw)) > 0,
        }
