# Business Entity Resolution — Pipeline

Multi-stage, leakage-safe entity resolution pipeline: multi-strategy blocking →
rich pairwise features → LightGBM pair classifier with grouped-CV hard-negative
mining → isotonic calibration → entity-level decision layer (macro-F0.5-optimized
thresholds, explicit singleton + multi-match handling) → submission files.

No external data, APIs, or lookups are used anywhere in this pipeline — every
signal is derived from the three provided TSV sources.

## 1. Install environment

```bash
cd code/business_entity_resolution
python3 -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt
```

## 2. Directory layout expected by the code

Place (or symlink) the challenge data so that, relative to this package's
grandparent directory (referred to below as `<root>`, i.e. the folder that
contains both `code/` and `output/`):

```
<root>/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── utils/
│   └── validate_submission.py     # organizer's copy if available, else the
│                                   # one included in this package is used
├── output/                        # created automatically
├── models/                        # created automatically (trained artifacts)
└── code/business_entity_resolution/   # this package
```

`src/config.py`'s `Paths` class resolves `<root>` automatically as three
levels up from `src/`, or pass `--dataset-dir` / `--output-dir` explicitly.

## 3. Run everything (train → validate → predict → validate submission)

```bash
python3 -m src.generate_submission --stage all \
    --dataset-dir ../../dataset \
    --output-dir ../../output
```

This prints, in order: a data audit, blocking candidate-recall report,
grouped-CV training log with hard-negative rounds, the chosen decision
thresholds, an ablation table (rule baselines vs. the trained model vs. the
full entity-level decision layer), a validation error-analysis sample, then
runs inference on the test set and finally the local format validator.
It exits non-zero if validation fails — **do not submit if this prints
`VALIDATION FAILED`.**

## 4. Run stages independently

```bash
python3 -m src.generate_submission --stage audit    # data audit only
python3 -m src.generate_submission --stage train     # audit + blocking + train + save artifacts
python3 -m src.generate_submission --stage predict    # requires a prior --stage train run's artifacts
```

(`predict`-only mode reloads `models/classifier.pkl`, `models/calibrator.pkl`,
and `models/decision_config.json` saved by a previous `train`/`all` run —
see `src/generate_submission.py` if you want to wire that reload path in
explicitly for a two-command train/predict split.)

## 5. Validate the submission files manually

```bash
python3 ../../utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test
```

Prints `PASS` (exit 0) or a numbered list of issues (exit 1).

## 6. Reproducibility notes

- Random seed fixed via `RANDOM_SEED = 42` / `ModelConfig.seed` in
  `src/config.py`, threaded through LightGBM and the GroupKFold splitter.
- All cross-validation is **grouped by `source1_entity_id`**
  (`sklearn.model_selection.GroupKFold`) so no S1 entity's candidate pairs
  ever straddle the train/validation boundary within a fold.
- `output/candidate_pairs.tsv` is written from the exact same candidate set
  that is fed into the trained classifier for scoring — not an earlier,
  unfiltered blocking pass — satisfying the "last stage before scoring"
  requirement.
- Model: LightGBM (MIT license), gradient-boosted decision trees — no
  parameter-count concerns versus the 8B ceiling. A logistic-regression
  fallback (`sklearn`, BSD license) is used automatically if LightGBM isn't
  importable, so the pipeline still runs in a minimal environment.

## 7. Package structure

```
src/
├── config.py              # paths, seeds, all tunable hyperparameters
├── data_loader.py          # robust TSV loading + ground-truth parsing
├── preprocessing.py        # fits the data-driven Normalizer, quick audits
├── normalization.py        # multi-representation text normalization
├── blocking.py             # multi-strategy candidate generation
├── features.py             # pairwise feature engineering
├── models.py                # PairClassifier (LightGBM) + ScoreCalibrator
├── training.py              # grouped-CV training + hard-negative mining
├── calibration.py           # calibrator save/load helpers
├── decision.py               # entity-level decision layer + threshold search
├── inference.py              # test-set blocking→features→scoring→decisions
├── evaluation.py             # exact macro/micro F0.5 + candidate-recall metrics
├── error_analysis.py         # categorized FP/FN report
└── generate_submission.py    # CLI entry point tying every stage together
```
