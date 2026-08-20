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

# --- Deployment constraints (Gap 1) ----------------------------------------
LATENCY_BUDGET_MS = 100.0   # p99 target for real-time scoring
N_LATENCY_TRIALS = 1000
