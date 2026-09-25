# Business Entity Resolution — Methodology Document

> **Status note (read first):** this document describes the pipeline
> design and validates its mechanics against (a) the true schema/noise
> characteristics observed in the 50-row sample pack provided, and (b) a
> synthetic-but-realistic near-duplicate test built from that sample. It
> does **not** yet report performance numbers from the real training set,
> because the sample's `train_ground_truth.tsv` references entities that
> are entirely absent from its own `train_source1/2/3.tsv` (0 of 175
> matched IDs and 49 of 50 `source1_entity_id`s have no counterpart in the
> sample source files) — the sample is a schema/format sample, not a
> coherent mini matching problem, and cannot be used to compute a
> trustworthy F0.5. All numbers below marked *(sample)* or *(synthetic)*
> should be re-run against the real `dataset/train` before being reported
> as the competition entry's actual performance; the pipeline itself
// requires no changes to do so — see README.md §3.

## 1. Problem formulation

We treat this as probabilistic record linkage: each Source 1 record is
compared against a bounded candidate set of Source 2/3 records, a
classifier estimates P(same real-world business | pair of records), and
a per-entity decision layer converts that score distribution into a
final match set. Because the competition metric (macro F0.5, averaged
per Source 1 entity) penalizes false merges twice as heavily as it does
missed matches, and singletons are scored to the same standard as
multi-match entities, the design goal throughout is **precision at the
entity level**, not pairwise accuracy in the abstract.

## 2. Data analysis

Audited on the provided sample (`dataset/train`, `dataset/test`, 50 rows
per file — see the status note above for its limits as a *statistical*
sample, though its *structural* characteristics are real):

- **Schema**: `entity_id, business_name, business_address, country`,
  tab-separated, CRLF line endings — read cleanly with
  `pd.read_csv(sep="\t", dtype=str, keep_default_na=False)`.
- **Country**: train covers US/India only (US ≈ 58-66%, India ≈
  34-42% depending on source); **test additionally contains France**
  (~14-18% of test rows) with legal suffixes (`SARL`, `EURL`, `SAS`) not
  seen in training. Confirms the brief's warning: country must be an
  open-set feature, never a hard filter.
- **Script/transliteration**: Source 1 is ~100% Latin-script in the
  sample; **Source 2 shows 12% non-Latin business names, Source 3 shows
  18%** (Devanagari, Tamil, Bengali scripts observed directly). This is
  the single most consequential noise pattern found — a purely
  Latin-alphabet fuzzy-matching pipeline would silently fail on a
  meaningful slice of S2/S3 records with no name-side signal at all.
- **Postal codes are rare**: 6-digit Indian PIN codes: 0% of rows in the
  sample; 5-digit US ZIP: 4-6%. This contradicts treating postal-code
  blocking as a primary strategy — it's kept as one signal among several,
  not load-bearing.
- **Match structure**: per-S1 match counts (train, sample) range 0-9;
  singleton rate 8%; most entities have 2-4 matches; ~76% of matched
  entities have evidence from **both** S2 and S3, the rest split roughly
  evenly S2-only/S3-only — multi-match and cross-source evidence are both
  the norm, not the edge case.
- **Duplicates/missing**: no exact duplicate rows within any source in
  the sample; `business_address` empty in 4-8% of S2/S3 rows (must be
  handled without crashing string/feature code — done via
  `keep_default_na=False` + explicit `is_empty` feature flags).

## 3. Normalization

Implemented in `src/normalization.py`. Rather than collapsing each field
to one canonical string, we keep multiple parallel representations
(`raw`, `lower`, `ascii_fold`, `alnum`, `tokens`, `sorted_tokens`,
`suffix_stripped`, `core_tokens`, plus address-specific `digit_signature`
/ `digit_tokens`) and let the feature layer consume whichever is
appropriate per metric. `ascii_fold` (Unicode NFKD + combining-mark
strip + non-ASCII drop) is the key response to the script-mismatch
finding above: a same-script exact match still scores 1.0 on the alnum
channel, while a genuinely script-mismatched pair collapses toward 0 on
that channel — and an explicit `is_non_latin` flag lets the model learn
to lean on address/digit evidence instead of name evidence for those
rows, rather than silently mis-scoring them.

Legal-suffix stripping uses a small curated multi-jurisdiction seed list
(`Inc/Corp/LLC/Ltd/Pvt/Private/SARL/EURL/SAS/GmbH/...`) **extended** by
mining frequent trailing name-tokens from the training data
(`mine_legal_suffixes`), so the suffix vocabulary is not hard-locked to
US/India and can pick up patterns specific to the full dataset (or
France) that the seed list misses.

## 4. Candidate generation (blocking)

Implemented in `src/blocking.py`. Six independent strategies run
per-source (S1×S2 and S1×S3 separately) and their outputs are unioned,
capped at `max_candidates_per_s1` (default 60, prioritizing candidates
with the most corroborating strategies when the cap binds):

| Strategy | What it catches |
|---|---|
| A. exact normalized (suffix-stripped) name | clean/near-clean duplicates |
| B. name char n-gram TF-IDF cosine (top-k, `NearestNeighbors`) | typos, abbreviations, minor reorderings |
| C. address char n-gram TF-IDF cosine (top-k) | address noise, partial addresses |
| D. core-name-token inverted index (overlap-ranked) | word-order transpositions |
| E. rare-token inverted index (name+address tokens with document frequency ≤ 3) | distinctive tokens surviving heavy corruption elsewhere in the string |
| F. address digit-signature exact match | shared house/building/PIN numbers regardless of surrounding text |

Validated on a synthetic-but-realistic test (real S1 rows + programmatically
noised near-duplicates — typo/abbreviation/case noise — plus pure-noise
distractor records): **candidate recall = 1.00** (57/57 true pairs
recovered) at an average of ~57 candidates per S1 entity. On the disjoint
sample pack, candidate recall is (expectedly, per the status note) 0.0,
since no true match exists in that pack's own candidate pool by
construction. **The recall ceiling must be re-measured on the real
training set** before trusting downstream numbers — this is a one-command
step (`--stage all` prints it) and `config.BlockingConfig` exposes every
top-k/threshold knob to widen if recall comes in low.

## 5. Feature engineering

Implemented in `src/features.py`, ~50 features per pair (see
`get_feature_names()` for the live list), grouped as:

- **Name** (name_*): exact-match variants at 4 normalization levels,
  Levenshtein/Jaro-Winkler/token-sort/token-set/partial-ratio
  similarities (via `rapidfuzz`), token Jaccard/overlap, char n-gram
  Jaccard, length ratio/diff, prefix/suffix match, and
  `name_either_non_latin` / `name_both_non_latin` script-mismatch flags.
- **Address** (addr_*): the same string-similarity family plus
  digit-token Jaccard, digit-signature exact match, and an
  `addr_either_empty` flag so missing addresses degrade gracefully
  instead of poisoning similarity scores.
- **Country** (country_*): exact match, both-known flag, mismatch flag —
  used as a *feature*, never a filter, so France/unseen countries are
  handled identically to trained-on countries.
- **Cross-field** (cross_*): name×address product, name+address average,
  high/low quadrant indicators (e.g. `cross_name_high_addr_low` — likely
  a common-name collision — vs. `cross_name_low_addr_high` — likely a
  DBA/trade-name situation), min/max of the two channels.
- **Blocking evidence**: `blocking_n_strategies`, how many independent
  blockers proposed this pair — itself a useful prior.

## 6. Matching model

LightGBM (`PairClassifier` in `src/models.py`), gradient-boosted trees
over the structured similarity features above — MIT-licensed, far under
the 8B-parameter ceiling in any meaningful reading of that constraint. A
logistic-regression fallback (`sklearn`, BSD) is wired in automatically
if LightGBM is unavailable, so the pipeline degrades gracefully rather
than hard-failing. The ablation harness (`run_ablation` in
`generate_submission.py`) exists specifically to check this choice
against simpler rule baselines (exact match, fuzzy-name-only,
fuzzy-name+address-average) on real training data rather than assuming
tree models win — **that ablation table must be regenerated on the real
dataset**; the version produced against the disjoint sample pack is
uninformative (all methods score 0.08 uniformly, an artifact of the
sample's ground-truth mismatch, not a real comparison).

## 7. Hard-negative mining

Rather than sampling negatives from the full cross-product (mostly
trivially-easy negatives), **every blocked candidate is used as a
training example** (positive if in that S1's ground-truth list,
negative otherwise) — since blocking already selected these for
name/address similarity, this pool is a hard-negative pool by
construction. On top of that, `training.hard_negative_reweight` runs
`ModelConfig.hard_negative_rounds` (default 2) rounds of: fit → find the
highest-scoring negatives per S1 entity (out-of-fold) → upweight them
3× → refit. This sharpens the boundary on the pairs the model is
currently most confused by, without inventing candidates that would
never appear in `candidate_pairs.tsv`.

## 8. Calibration

Isotonic regression (`ScoreCalibrator` in `src/models.py`), fit on
**out-of-fold** scores from the grouped CV (never in-sample scores), so
the calibrated score is a genuine held-out-data probability estimate.
This matters specifically because the decision layer's absolute
threshold is chosen once and then applied to test data that includes an
unseen country (France) — a calibrated score is more likely to transfer
across that distribution shift than a raw tree-ensemble margin would.

## 9. Entity-level decision system

Implemented in `src/decision.py`. For each S1 entity, candidates are
ranked by calibrated score; a candidate is kept if it clears **both** an
absolute floor (`abs_threshold`, grid-searched) **and** a relative margin
against that entity's own top score (`score >= rel_margin * top_score`).
The relative-margin term is what allows genuine multi-match (common in
the audit — most entities have 2-4 matches) to survive without simply
keeping "top-3 always." If nothing clears the absolute floor, the
prediction is the empty set — the exact behavior a true singleton needs
to score F0.5 = 1.0, and the reason singleton handling doesn't need a
separate classifier: it falls out of the same threshold search that
directly optimizes macro F0.5.

## 10. Threshold optimization

`decision.search_thresholds` grid-searches `(abs_threshold, rel_margin)`
directly against **macro F0.5 on out-of-fold scores** (never in-sample),
over a fine absolute-threshold grid (`config.DECISION.threshold_grid`,
step 0.005) crossed with 7 relative-margin settings. This is the
mechanism that makes the whole system optimize the actual competition
metric rather than a proxy like log-loss or 0.5-thresholded accuracy —
the naive `score > 0.5` baseline is reported alongside for comparison in
the training log.

## 11. Validation methodology

All cross-validation is **grouped by `source1_entity_id`**
(`sklearn.model_selection.GroupKFold`, `src/training.py`), so no S1
entity's candidate pairs are ever split across train/validation within a
fold — avoiding the leakage the brief specifically warns a random
pair-level split would cause. `src/evaluation.py` implements the exact
competition metric (macro F0.5 over S1 entities, with the exact
zero/one singleton edge cases) plus macro/micro P/R, singleton accuracy,
singleton false-positive rate, and candidate recall/reduction ratio, all
computed the same way in training, ablation, and error analysis so
numbers are comparable across the pipeline.

## 12. Error analysis

`src/error_analysis.py` walks every OOF false positive/negative for the
chosen threshold and tags each with a rough category (script mismatch,
country mismatch, name-high/address-low vs. address-high/name-low
quadrants, moderate-similarity band) alongside the two records and the
model score, printed by `generate_submission.py`'s error-analysis stage
and saved to `models/error_analysis.csv`. On the disjoint sample this
surfaces only `blocking_miss` rows (expected, since no true candidate
was ever in that pack's own pool) — on the real dataset this is the
primary tool for iterating on normalization/feature/blocking gaps.

## 13. Ablation studies

`generate_submission.run_ablation` compares, on the same candidate pool
and the same grouped-CV logic: (A) exact name-or-address rule, (B) fuzzy
name similarity only, (C) fuzzy name+address average, (D) trained model
raw/uncalibrated, (E) full pipeline (calibrated + entity-level decision).
**Must be re-run on the real training set** — the sample-pack run
(reported by the code, not hand-picked) shows all five methods tied at
0.08 macro F0.5, which is an artifact of the sample's ground-truth
mismatch (see status note), not a real result.

## 14. Computational complexity

Blocking avoids full S1×(S2∪S3) comparison: TF-IDF retrieval uses
`sklearn.neighbors.NearestNeighbors` over sparse char-n-gram matrices
(sub-quadratic query cost), and the token/rare-token/digit-signature
strategies are inverted-index lookups (O(matching postings) per query,
not O(|S2∪S3|)). `SourceIndex` builds each source's indexes once and
reuses them across every S1 query. Feature computation is cached per-S1
normalization (`training.compute_features_for_candidates`) so a
business name/address string is normalized once even if the same S1
entity has many candidates.

## 15. Compliance

- No external database, API, geocoding service, or entity-resolution
  service is called anywhere in this codebase — `grep -r "requests\.\|urllib\|http" src/`
  returns nothing; the only I/O is local file reads (`pandas.read_csv`)
  and local model artifacts (`pickle`).
- No external data augmentation: all normalization vocabulary (legal
  suffixes, generic business words) is either a small curated seed list
  or mined directly from the provided training `business_name` column.
- Country is treated as an open-set string feature throughout (see §5,
  §9) — no code path filters or branches on a fixed country list; the
  pipeline was exercised against test data containing France (unseen in
  training) without modification.
- Model: LightGBM (MIT License), well under the 8B-parameter ceiling.
  Fallback: scikit-learn `LogisticRegression` (BSD License).
- Reproducibility: fixed seed (`RANDOM_SEED = 42`), pinned
  `requirements.txt`, single-command reproduction path in README.md.
