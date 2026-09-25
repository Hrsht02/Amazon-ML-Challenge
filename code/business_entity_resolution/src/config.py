"""Central configuration for the scalable Business Entity Resolution pipeline."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

@dataclass
class Paths:
    root: Path = Path(__file__).resolve().parents[3]
    dataset_dir: Path = None
    train_dir: Path = None
    test_dir: Path = None
    output_dir: Path = None
    models_dir: Path = None
    runs_dir: Path = None
    def __post_init__(self):
        self.dataset_dir=self.root/"dataset"; self.train_dir=self.dataset_dir/"train"
        self.test_dir=self.dataset_dir/"test"; self.output_dir=self.root/"output"
        self.models_dir=self.root/"models"; self.runs_dir=self.output_dir/"run_history"
        self.output_dir.mkdir(parents=True,exist_ok=True); self.models_dir.mkdir(parents=True,exist_ok=True)
        self.runs_dir.mkdir(parents=True,exist_ok=True)

@dataclass
class BlockingConfig:
    ann_dim:int=128
    ann_nlist:int=16384
    ann_pq_m:int=16
    ann_nprobe:int=32
    ann_name_k:int=80
    ann_address_k:int=60
    max_candidates_per_s1:int=80
    s1_batch_size:int=5000
    index_build_batch_size:int=50000

@dataclass
class ModelConfig:
    seed:int=42
    n_splits_cv:int=5
    max_train_pairs:int=3_000_000
    negatives_per_positive:int=6
    hard_negative_rounds:int=2
    hard_negatives_per_s1:int=8
    lgbm_params:dict=field(default_factory=lambda:dict(
        objective="binary",metric="binary_logloss",learning_rate=0.035,
        num_leaves=63,min_child_samples=30,feature_fraction=0.9,
        bagging_fraction=0.9,bagging_freq=1,n_estimators=700,
        verbosity=-1,n_jobs=-1
    ))

@dataclass
class DecisionConfig:
    threshold_grid:tuple=tuple(round(i/200,3) for i in range(5,200))
    min_absolute_score:float=0.03
    max_matches_per_s1:int=20
    relative_margins:tuple=(0.0,0.55,0.65,0.75,0.85,0.92,0.97)

PATHS=Paths(); BLOCKING=BlockingConfig(); MODEL=ModelConfig(); DECISION=DecisionConfig(); RANDOM_SEED=42

def config_dict()->dict[str,Any]:
    return {"blocking":asdict(BLOCKING),"model":asdict(MODEL),"decision":asdict(DECISION),"seed":RANDOM_SEED}
