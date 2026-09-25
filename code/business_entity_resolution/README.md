# Business Entity Resolution — Scalable Pipeline

This version keeps the required submission format while replacing the original
million-row TF-IDF/NearestNeighbors blocker with a persisted FAISS IVF-PQ ANN
blocker. FAISS uses compressed inverted-file/product-quantization indexes that
are designed for large vector collections; this avoids materializing a huge
5M-row sparse TF-IDF nearest-neighbor workload. The index is persisted and
reused between experiments. FAISS is MIT licensed.

## Required output remains unchanged

```
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

The challenge requires one matching row for every S1 test entity, permits zero,
one or many matches, and requires final matches to be a subset of the final
candidate set. The runner preserves those rules.

## Install

```powershell
cd code/business_entity_resolution
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For Colab:

```bash
pip install -r requirements.txt
```

## Recommended first run

Do **not** immediately spend a long run on the complete pipeline. First run:

```bash
python -m src.generate_submission --stage audit \
  --dataset-dir ../../dataset --output-dir ../../output
```

Then run the full pipeline:

```bash
python -m src.generate_submission --stage all \
  --dataset-dir ../../dataset --output-dir ../../output
```

Every run gets an immutable experiment directory:

```
output/run_history/<RUN_ID>/
├── audit.json
├── blocking_train_metrics.json
├── validation_metrics.json
├── classifier.pkl
├── calibrator.pkl
├── normalizer.pkl
├── decision_config.json
├── config.json
├── run_manifest.json
├── events.jsonl
├── matching_results.tsv
├── candidate_pairs.tsv
└── validator.txt
```

The root `output/matching_results.tsv` and `output/candidate_pairs.tsv`
remain the files used for submission.

## Re-run without losing the previous experiment

Never overwrite an old run. Use a new run id:

```bash
python -m src.generate_submission --stage all \
  --run-id experiment_02 \
  --dataset-dir ../../dataset --output-dir ../../output
```

The ANN indexes are cached separately:

```
output/ann_indexes_train_v2/
output/ann_indexes_test_v2/
```

If you change ANN/index parameters, rebuild them with:

```bash
python -m src.generate_submission --stage train --force-index \
  --run-id experiment_03 \
  --dataset-dir ../../dataset --output-dir ../../output
```

## Predict again from an existing trained run

This does not retrain the model:

```bash
python -m src.generate_submission --stage predict \
  --run-id experiment_02 \
  --dataset-dir ../../dataset --output-dir ../../output
```

It loads the saved classifier, calibrator, normalizer and decision thresholds
from that run.

## What changed

### 1. Scalable blocking

The old implementation used sklearn character TF-IDF matrices plus
NearestNeighbors over millions of records. At the actual dataset size this is
the wrong computational shape.

The new blocker:

- hashes character 2–5 grams into a fixed 128-dimensional vector;
- builds separate name and address FAISS IVF-PQ indexes for S2 and S3;
- searches in batches;
- keeps only the top ANN candidates per S1;
- persists the indexes;
- never filters by country, so unseen countries such as France remain eligible.

### 2. Bounded training memory

Training does not materialize every blocked pair.

For each S1 entity it keeps:

- all blocked positives that were recovered;
- the strongest ANN negatives;
- a bounded number of negatives per positive.

The final training sample is capped by `max_train_pairs`.

Candidate recall is recorded in `blocking_train_metrics.json`. If blocking
recall is poor, change the ANN parameters before trusting the model score.

### 3. Batch inference

Test S1 is processed in batches. Features are calculated only for the current
candidate batch, predictions are written incrementally, and every S1 entity
gets exactly one output row.

### 4. Reproducibility

Every experiment saves:

- exact configuration;
- trained model;
- calibrator;
- fitted normalizer;
- decision thresholds;
- audit;
- blocking metrics;
- validation metrics;
- validator output;
- final TSVs;
- event/progress log.

This lets you compare experiment 01 vs experiment 02 without losing the
previous output.

## Important competition constraints

The provided challenge specification says:

- use only the supplied data;
- do not use external entity-resolution APIs/databases/geocoding or internet
  business lookups;
- treat country as an open set;
- submit the two TSV files in the required format;
- use a final model within the stated MIT/Apache and parameter constraints.

This repository follows those constraints. FAISS itself is MIT licensed.

## Package layout

```
src/
├── config.py
├── data_loader.py
├── preprocessing.py
├── normalization.py
├── blocking.py                 # legacy blocker retained for reference
├── faiss_blocking.py           # scalable blocker used by the CLI
├── batch_features.py
├── features.py
├── models.py
├── training.py
├── calibration.py
├── decision.py
├── evaluation.py
├── inference.py
├── pipeline_runner.py
└── generate_submission.py
```
