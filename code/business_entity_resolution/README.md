# Business Entity Resolution — High-Recall, Resumable Pipeline

This is the scalable pipeline for the Amazon ML Challenge business entity
resolution task.

## Required submission format

The final submission directory is:

~~~text
output/
├── matching_results.tsv
└── candidate_pairs.tsv
~~~

The files remain TSV, with these exact headers:

~~~text
matching_results.tsv
source1_entity_id    matched_entity_ids

candidate_pairs.tsv
source1_entity_id    candidate_entity_ids
~~~

The runner writes exactly one row for every test Source-1 entity and checks the
row count before marking prediction complete.

## Current pipeline

~~~text
Source 1
   │
   ├── ASCII character ANN ───────┐
   ├── Unicode character ANN ─────┤
   └── address ANN ───────────────┘
                  │
                  ▼
          candidate union
                  │
                  ▼
       rich pairwise features
                  │
                  ▼
       grouped 5-fold OOF
                  │
                  ▼
          hard-negative rounds
                  │
                  ▼
       calibrated LightGBM score
                  │
                  ▼
       macro-F0.5 threshold search
                  │
                  ▼
          final full model
                  │
                  ▼
       test candidate prediction
~~~

### High-recall blocking

The previous blocker used only a 128-dimensional ASCII-normalized ANN
representation. The current blocker uses:

- 512-dimensional hashed character n-grams;
- ASCII-normalized name ANN;
- Unicode-preserving name ANN;
- address ANN;
- larger retrieval pools;
- ANN rank/similarity evidence retained as model features;
- no country filtering, so France and other unseen countries remain eligible;
- persisted FAISS indexes.

The Unicode path is important because the training audit contains a substantial
non-Latin name population.

## Crash-safe execution

Use the same RUN_ID when resuming.

~~~text
output/run_history/<RUN_ID>/
├── audit.json
├── normalizer.pkl
├── sampling_state.json
├── sampling_chunks/
├── training_candidates.pkl.gz
├── training_features.pkl.gz
├── blocking_train_metrics.json
├── oof_checkpoints/
│   ├── round0_fold_*.npy
│   ├── round1_fold_*.npy
│   └── round2_fold_*.npy
├── classifier.pkl
├── calibrator.pkl
├── normalizer.pkl
├── decision_config.json
├── validation_metrics.json
├── config.json
├── events.jsonl
├── prediction_parts/
├── matching_results.tsv
├── candidate_pairs.tsv
└── .done_*.json
~~~

If a runtime dies during candidate sampling, the next invocation resumes from
the saved S1 position.

If it dies during OOF training, completed folds are reused and only the missing
folds are retrained.

If it dies during test inference, completed prediction batches are skipped.

This is designed specifically for Colab runtimes, whose managed VMs can be
terminated and have maximum lifetimes. The persistent run directory therefore
belongs on Google Drive, not only on the temporary Colab filesystem.

## Progress / metrics printed during execution

The runner continuously prints:

- current stage;
- S1 rows processed;
- training-pair count;
- candidate recall;
- candidate pairs seen;
- elapsed time;
- RAM/CPU/GPU telemetry;
- feature-generation progress;
- OOF fold progress;
- ROC-AUC, average precision and log loss per OOF fold;
- LightGBM device selected;
- macro precision/recall/F0.5;
- micro precision/recall/F0.5;
- singleton accuracy and false-positive rate;
- chosen absolute threshold;
- chosen relative margin;
- prediction progress;
- final output paths.

The complete machine-readable event stream is saved to events.jsonl.

## CPU / GPU behavior

The code is CPU-safe and can run on ordinary Windows/Linux/Colab CPU runtimes.

For LightGBM, the model attempts CUDA and then the GPU backend when a CUDA-capable
runtime is detected. If the installed LightGBM build does not support the
requested GPU backend, it automatically falls back to CPU and disables further
GPU attempts for that run.

For FAISS, the standard faiss-cpu PyPI package is used for reproducibility.
If a custom GPU-enabled FAISS build is installed, the blocker can use it for
search and automatically retry a failed GPU search on CPU. The standard PyPI
CPU wheel does not itself provide FAISS GPU binaries.

A Colab runtime disappearing completely cannot be caught by Python. The recovery
mechanism is therefore persistent checkpoints: start a new runtime, mount Drive,
repeat setup, and execute the same RUN_ID.

## Colab

Use:

~~~text
colab/amazon_ml_challenge_resumable.ipynb
~~~

Recommended Drive layout:

~~~text
My Drive/
└── AmazonMLChallenge/
    ├── dataset/
    │   ├── train/
    │   └── test/
    └── output/
~~~

The notebook copies the dataset to the local Colab VM for faster TSV access,
while checkpoints, FAISS indexes and final outputs are stored under Drive.

Run:

~~~bash
python -u -m src.generate_submission \
  --stage all \
  --run-id experiment_06 \
  --dataset-dir /content/dataset \
  --output-dir /content/drive/MyDrive/AmazonMLChallenge/output
~~~

After a runtime reset, run the setup cells again and use the same
experiment_06 run ID. Do not use --force-index when resuming unless you
intentionally want to rebuild the ANN indexes.

## Install locally

~~~powershell
cd code/business_entity_resolution
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
~~~

Then:

~~~powershell
python -m src.generate_submission --stage audit
  --dataset-dir ../../dataset
  --output-dir ../../output
~~~

Training:

~~~powershell
python -m src.generate_submission --stage train
  --run-id experiment_local_01
  --dataset-dir ../../dataset
  --output-dir ../../output
~~~

Prediction from an existing trained run:

~~~powershell
python -m src.generate_submission --stage predict
  --run-id experiment_local_01
  --dataset-dir ../../dataset
  --output-dir ../../output
~~~

## Experiment discipline

A run ID is the unit of reproducibility.

- Same RUN_ID = resume/reuse checkpoints.
- New RUN_ID = new experiment.
- --force-index = intentionally rebuild ANN indexes.
- Never delete an old run until its metrics and artifacts have been reviewed.

The current pipeline records candidate recall before model validation because
candidate recall is the ceiling imposed by blocking: a matcher cannot recover a
true pair that never entered the candidate set.
