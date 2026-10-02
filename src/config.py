"""
config.py — Single source of truth for experimental constants.

Every module imports from here. Nothing that affects a reported number should
be hard-coded anywhere else: a reader must be able to open this one file and
see the entire experimental configuration.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models" / "trained"
EXPERIMENTS_DIR = ROOT / "experiments"
FIGURES_DIR = ROOT / "dissertation" / "figures"
TABLES_DIR = ROOT / "dissertation" / "tables"

CSV_PATH = DATA_RAW / "creditcard.csv"

# --- Reproducibility -------------------------------------------------------
# Fixed across every experiment. Addresses lit review Gap 3: cross-study
# comparison is impossible when seeds and pipelines vary.
RANDOM_STATE = 42
N_FOLDS = 5

# --- Evaluation ------------------------------------------------------------
PRIMARY_METRIC = "auc_pr"
BOOTSTRAP_N = 1000
ALPHA = 0.05

# Cost model (Chapter 3 sec 3.4) - indicative, sensitivity-tested in Ch.6
COST_FN = 100.0   # missed fraud
COST_FP = 5.0     # false alarm

# Daily analyst review capacity, for precision@k / recall@k
ALERT_BUDGET = 100

# --- Techniques under comparison (lit review sec 3-4) ----------------------
BASE_CLASSIFIERS = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]

RESAMPLING_TECHNIQUES = [
    "none",                # baseline, no imbalance handling
    "random_undersampling",
    "random_oversampling",
    "smote",
    "smote_tomek",
    "smote_enn",
    "adasyn",
    "soa_s",
]

ALGORITHM_TECHNIQUES = [
    "class_weight_balanced",
    "balanced_random_forest",
    "rusboost",
    "easy_ensemble",
    "xgb_scale_pos_weight",
    "lgbm_is_unbalance",
    "xgb_focal_loss",
]

ALL_TECHNIQUES = RESAMPLING_TECHNIQUES + ALGORITHM_TECHNIQUES

# Technique settings (Chapter 3 sec 3.5). Library defaults unless stated.
# Every technique gets the same tuning budget, zero searched trials, so no
# technique is advantaged by extra tuning (rule 4, Gap 3).
SMOTE_K_NEIGHBORS = 5        # Chawla et al. (2002) default
SOA_LOF_NEIGHBORS = 20       # LocalOutlierFactor default, fraud class only
FOCAL_GAMMA = 2.0            # Lin et al. (2017) recommended value, fixed
FOCAL_MIN_HESSIAN = 1e-6     # focal-loss Hessian can go negative; clip it

# --- Deployment constraints (Gap 1) ----------------------------------------
LATENCY_BUDGET_MS = 100.0   # p99 target for real-time scoring
N_LATENCY_TRIALS = 1000

# Latency protocol (Chapter 5 sec 5.4). Batch size 1 is the headline: a
# real-time fraud API scores one transaction at a time. Larger batches
# characterise bulk re-scoring throughput and are reported separately.
LATENCY_BATCH_SIZES = (1, 100, 1_000, 10_000)
LATENCY_PERCENTILES = (50, 95, 99)
N_LATENCY_WARMUP = 100      # discarded single-record calls before timing
N_BATCH_TRIALS = 100        # timed calls per batch size > 1
N_BATCH_WARMUP = 5          # discarded calls per batch size > 1
# Timing runs single-threaded, so CPU time / wall time should not exceed ~1.
# Above this ratio the thread pinning failed and the row is not comparable.
SINGLE_THREAD_CPU_TOLERANCE = 1.1
# Pause after pinning, before timing, so worker threads still spin-waiting
# from earlier multi-threaded BLAS work can go idle (see single_threaded()).
THREAD_SETTLE_SECONDS = 0.5
# Quick per-fold latency logged by the sweep. The rigorous measurement is
# inference_profiler.py; this is a sanity figure recorded with each run.
SWEEP_LATENCY_TRIALS = 100
SWEEP_LATENCY_WARMUP = 10
RESULTS_RAW_CSV = EXPERIMENTS_DIR / "results_raw.csv"

# --- Experiment tracking ---------------------------------------------------
# experiment_runner.py writes here and inference_profiler.py reads from here.
MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
MLFLOW_EXPERIMENT = "imbalance_sweep"
# Fixed, so model artifacts land in the same place whatever directory the
# sweep is launched from (MLflow's default is relative to the cwd).
MLFLOW_ARTIFACT_LOCATION = (ROOT / "mlruns").as_uri()
MLFLOW_MODEL_ARTIFACT = "model"
# Recent MLflow (3.16 when written) defaults to skops, which refuses tree
# models unless every internal type is whitelisted, and cannot hold the
# custom focal-loss objective. cloudpickle stores every estimator as-is.
MLFLOW_SERIALIZATION_FORMAT = "cloudpickle"
