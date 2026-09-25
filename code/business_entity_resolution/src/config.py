"""
Central configuration for the Business Entity Resolution pipeline.
Nothing here reaches the network or any external service — every value
is a path, a seed, or a hyperparameter tuned on the provided data only.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Paths:
    root: Path = Path(__file__).resolve().parents[3]  # .../ber
    dataset_dir: Path = None
    train_dir: Path = None
    test_dir: Path = None
    output_dir: Path = None
    models_dir: Path = None

    def __post_init__(self):
        self.dataset_dir = self.root / "dataset"
        self.train_dir = self.dataset_dir / "train"
        self.test_dir = self.dataset_dir / "test"
        self.output_dir = self.root / "output"
        self.models_dir = self.root / "models"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class BlockingConfig:
    # top-k candidates kept per blocking strategy, per S1 entity
    topk_name_tfidf: int = 20
    topk_name_char_ngram: int = 20
    topk_address_tfidf: int = 15
    topk_address_char_ngram: int = 15
    # char n-gram range used for the char-ngram TF-IDF blockers
    char_ngram_range: tuple = (2, 4)
    # rare-token blocking: a token is "rare" if it appears in <= this many records
    rare_token_max_df: int = 3
    # minimum length of a token to be eligible for rare-token / exact blocking
    min_token_len: int = 3
    # final safety cap on candidates per S1 entity after taking the union
    max_candidates_per_s1: int = 60


@dataclass
class FeatureConfig:
    char_ngram_range: tuple = (2, 4)


@dataclass
class ModelConfig:
    seed: int = 42
    n_splits_cv: int = 5
    lgbm_params: dict = field(default_factory=lambda: dict(
        objective="binary",
        metric="binary_logloss",
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        n_estimators=400,
        verbosity=-1,
    ))
    # hard negative mining rounds
    hard_negative_rounds: int = 2
    hard_negatives_per_s1: int = 8
    # negatives sampled per positive during initial (round-0) training
    random_negatives_per_positive: int = 5


@dataclass
class DecisionConfig:
    # grid search resolution for global score threshold
    threshold_grid: tuple = tuple(round(x, 3) for x in
                                   [i / 200 for i in range(1, 200)])
    # a candidate is never accepted below this score regardless of
    # entity-level rules (hard floor learned/validated, not assumed)
    min_absolute_score: float = 0.05
    # maximum matches ever returned for a single S1 entity (sanity cap)
    max_matches_per_s1: int = 15


PATHS = Paths()
BLOCKING = BlockingConfig()
FEATURES = FeatureConfig()
MODEL = ModelConfig()
DECISION = DecisionConfig()
RANDOM_SEED = 42
