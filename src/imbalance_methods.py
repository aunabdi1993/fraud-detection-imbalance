"""
imbalance_methods.py — The 15 techniques under comparison.

Maps to: Chapter 3 sec 3.5, Chapter 5 sec 5.2.

=== THE ONE RULE ===
Every sampler returned here is applied INSIDE a cross-validation fold, to the
training portion only. The validation fold is never resampled, because at
deployment the model faces the true 1:578 distribution. Resampling before
splitting produces synthetic minority points derived from validation rows and
inflates AUC-PR dramatically. This is the single most common error in the
reviewed corpus (lit review sec 5.3).

Correct:
    for tr, va in ds.cv_splits():
        X_res, y_res = get_sampler("smote").fit_resample(X[tr], y[tr])
        model.fit(X_res, y_res)
        score(model, X[va], y[va])          # untouched

Wrong:
    X_res, y_res = smote.fit_resample(X, y)   # <-- leakage
    cross_val_score(model, X_res, y_res)

Data-level techniques (samplers):
  random_undersampling  imblearn RandomUnderSampler
  random_oversampling   imblearn RandomOverSampler
  smote                 imblearn SMOTE (Chawla et al. 2002), k_neighbors=5.
                        With ~260 training frauds per fold this is safe, but
                        any fold with n_minority <= k is logged.
  smote_tomek           imblearn SMOTETomek (Batista et al. 2004)
  smote_enn             imblearn SMOTEENN (Batista et al. 2004)
  adasyn                imblearn ADASYN (He et al. 2008). Can fail with "No
                        samples will be generated" when the minority is very
                        sparse; that is caught and logged, and the fold trains
                        on unresampled data rather than crashing the sweep.
  soa_s                 Selective oversampling (Gnip et al. 2021); see
                        SelectiveOversampler for the deviation from the paper.

Every sampler balances to 1:1 ("auto"), the convention in the reviewed
corpus.

Algorithm-level techniques modify the model, not the data:
  class_weight_balanced     sklearn class_weight="balanced", applied to the
                            classifiers that support it (LR, DT, RF).
                            XGBoost's equivalent is xgb_scale_pos_weight.
  balanced_random_forest    imblearn BalancedRandomForestClassifier
  rusboost                  imblearn RUSBoostClassifier (Seiffert et al. 2010)
  easy_ensemble             imblearn EasyEnsembleClassifier (Liu et al. 2009)
  xgb_scale_pos_weight      scale_pos_weight = n_neg / n_pos of the training
                            fold (~578), computed, not hard-coded
  lgbm_is_unbalance         LightGBM is_unbalance=True
  xgb_focal_loss            hand-rolled focal-loss objective (Lin et al. 2017)

The last six are classifiers in their own right, so they are not crossed with
the four base classifiers; each is compared within its own model family
(STANDALONE below). That gives 8 x 4 + 3 + 6 = 41 (technique, classifier)
pairs, i.e. 205 fitted models over 5 folds.

Tuning budget (rule 4, Gap 3): every technique and classifier runs with
library defaults, the same as the baselines, and nothing is searched. The
stub originally suggested tuning the focal-loss gamma over {1, 2, 5}. That
would give one technique three tries where every other gets one, voiding the
comparison, so gamma is fixed at Lin et al.'s recommended 2.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import (
    BalancedRandomForestClassifier,
    EasyEnsembleClassifier,
    RUSBoostClassifier,
)
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import (
    EditedNearestNeighbours,
    RandomUnderSampler,
    TomekLinks,
)
from lightgbm import LGBMClassifier
from sklearn.neighbors import LocalOutlierFactor
from xgboost import XGBClassifier

from . import config
from .baseline_models import get_baseline_models

logger = logging.getLogger(__name__)

# Algorithm-level techniques that are a whole classifier, mapped to the model
# family they are compared within.
STANDALONE = {
    "balanced_random_forest": "random_forest",
    "rusboost": "adaboost",
    "easy_ensemble": "adaboost",
    "xgb_scale_pos_weight": "xgboost",
    "lgbm_is_unbalance": "lightgbm",
    "xgb_focal_loss": "xgboost",
}

# Base classifiers that accept class_weight. XGBoost does not; its
# cost-sensitive counterpart is the separate xgb_scale_pos_weight technique.
CLASS_WEIGHT_CLASSIFIERS = ("logistic_regression", "decision_tree", "random_forest")

# Samplers built on k-nearest-neighbour interpolation; they need more than k
# minority points in the training fold.
_KNN_SAMPLERS = {"smote", "smote_tomek", "smote_enn", "adasyn", "soa_s"}


def sweep_pairs() -> list[tuple[str, str]]:
    """Every (technique, classifier) pair in the comparison, in a fixed order."""
    pairs = [
        (t, c) for t in config.RESAMPLING_TECHNIQUES for c in config.BASE_CLASSIFIERS
    ]
    pairs += [("class_weight_balanced", c) for c in CLASS_WEIGHT_CLASSIFIERS]
    pairs += list(STANDALONE.items())
    return pairs


def is_algorithm_level(technique: str) -> bool:
    """Algorithm-level techniques modify the model, not the data."""
    return technique in config.ALGORITHM_TECHNIQUES


# ---------------------------------------------------------------------------
# Data level
# ---------------------------------------------------------------------------
class SelectiveOversampler:
    """SOA-S: oversample only the representative core of the fraud class.

    After Gnip et al. (2021). Plain SMOTE interpolates between any two nearby
    frauds, including isolated outliers, so it can manufacture synthetic
    fraud deep inside legitimate territory. SOA-S first identifies minority
    outliers and generates synthetic points from the remaining core only.

    Deviation from the paper, to be stated in Chapter 3: there is no library
    implementation, so outliers are found with LocalOutlierFactor fitted on
    the fraud rows alone, and synthesis is SMOTE over the core. Outliers are
    kept in the training data as real observations; they are only excluded
    as seeds for synthesis. The class is balanced to 1:1 like the other
    samplers.
    """

    def __init__(
        self,
        n_neighbors: int = config.SOA_LOF_NEIGHBORS,
        k_neighbors: int = config.SMOTE_K_NEIGHBORS,
        random_state: int = config.RANDOM_STATE,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        n_new = len(neg) - len(pos)
        if n_new <= 0:
            return X, y

        lof = LocalOutlierFactor(n_neighbors=min(self.n_neighbors, len(pos) - 1))
        core = pos[lof.fit_predict(X[pos]) == 1]
        logger.info(
            "SOA-S: %d of %d frauds flagged as outliers, excluded as seeds",
            len(pos) - len(core), len(pos),
        )
        if len(core) <= self.k_neighbors:
            logger.warning(
                "SOA-S: only %d core frauds (k=%d); training on unresampled data",
                len(core), self.k_neighbors,
            )
            return X, y

        X_seed = np.vstack([X[neg], X[core]])
        y_seed = np.concatenate([y[neg], y[core]])
        smote = SMOTE(
            sampling_strategy={1: len(core) + n_new},
            k_neighbors=self.k_neighbors,
            random_state=self.random_state,
        )
        X_res, _ = smote.fit_resample(X_seed, y_seed)
        # imblearn appends synthetic rows after the originals.
        synthetic = X_res[len(X_seed):]
        y_synth = np.ones(len(synthetic), dtype=y.dtype)
        return np.vstack([X, synthetic]), np.concatenate([y, y_synth])


def get_sampler(name: str, random_state: int = config.RANDOM_STATE) -> Any:
    """Return an unfitted sampler for a data-level technique, or None for 'none'."""
    k = config.SMOTE_K_NEIGHBORS

    def smote() -> SMOTE:
        return SMOTE(k_neighbors=k, random_state=random_state)

    builders = {
        "none": lambda: None,
        "random_undersampling": lambda: RandomUnderSampler(random_state=random_state),
        "random_oversampling": lambda: RandomOverSampler(random_state=random_state),
        "smote": smote,
        "smote_tomek": lambda: SMOTETomek(
            smote=smote(), tomek=TomekLinks(n_jobs=-1), random_state=random_state
        ),
        "smote_enn": lambda: SMOTEENN(
            smote=smote(), enn=EditedNearestNeighbours(n_jobs=-1),
            random_state=random_state,
        ),
        "adasyn": lambda: ADASYN(n_neighbors=k, random_state=random_state),
        "soa_s": lambda: SelectiveOversampler(random_state=random_state),
    }
    if name not in builders:
        raise ValueError(f"{name!r} is not a data-level technique")
    return builders[name]()


def resample_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    technique: str,
    random_state: int = config.RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a data-level technique to ONE training fold.

    Returns (X_train, y_train) unchanged for 'none' and for algorithm-level
    techniques. Never pass validation or test data to this function.
    """
    if technique == "none" or is_algorithm_level(technique):
        return X_train, y_train

    n_minority = int(y_train.sum())
    if technique in _KNN_SAMPLERS and n_minority <= config.SMOTE_K_NEIGHBORS:
        logger.warning(
            "%s: only %d minority rows for k=%d neighbours",
            technique, n_minority, config.SMOTE_K_NEIGHBORS,
        )

    sampler = get_sampler(technique, random_state)
    try:
        return sampler.fit_resample(X_train, y_train)
    except RuntimeError as err:
        if technique != "adasyn":
            raise
        logger.warning(
            "ADASYN generated no samples (%s); this fold trains on the "
            "unresampled data", err,
        )
        return X_train, y_train


# ---------------------------------------------------------------------------
# Algorithm level
# ---------------------------------------------------------------------------
class FocalLoss:
    """Binary focal loss as an XGBoost objective (Lin et al. 2017).

        FL = -[y (1-p)^g log p + (1-y) p^g log(1-p)],   p = sigmoid(margin)

    The (1-p)^g factor shrinks the loss on examples the model already gets
    right, so the ~99.8% easy legitimate transactions stop dominating the
    gradient. g = 0 recovers ordinary log-loss, which the tests check.

    A class with __call__, not a closure, so the fitted model pickles and can
    be logged to MLflow.
    """

    def __init__(self, gamma: float = config.FOCAL_GAMMA) -> None:
        self.gamma = gamma

    def __call__(
        self, y_true: np.ndarray, margin: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        g = self.gamma
        p = np.clip(1.0 / (1.0 + np.exp(-margin)), 1e-12, 1 - 1e-12)
        q = 1.0 - p
        # Derivatives of the positive-class loss with respect to the margin.
        # The negative-class loss is the same function with p and q swapped
        # and the margin negated, which flips the gradient's sign only.
        grad_pos = g * p * q**g * np.log(p) - q ** (g + 1)
        hess_pos = (
            g * p * q ** (g + 1) * np.log(p)
            - g**2 * p**2 * q**g * np.log(p)
            + (2 * g + 1) * p * q ** (g + 1)
        )
        grad_neg = -(g * q * p**g * np.log(q) - p ** (g + 1))
        hess_neg = (
            g * q * p ** (g + 1) * np.log(q)
            - g**2 * q**2 * p**g * np.log(q)
            + (2 * g + 1) * q * p ** (g + 1)
        )
        pos = y_true == 1
        grad = np.where(pos, grad_pos, grad_neg)
        # Focal loss is not convex everywhere; a non-positive Hessian would
        # break XGBoost's Newton step.
        hess = np.maximum(np.where(pos, hess_pos, hess_neg), config.FOCAL_MIN_HESSIAN)
        return grad, hess


def _standalone_model(
    technique: str, random_state: int, y_train: np.ndarray | None
) -> Any:
    if technique == "balanced_random_forest":
        # Parameters given explicitly: imblearn changed these defaults in
        # 0.13, and the comparison must not depend on the installed version.
        return BalancedRandomForestClassifier(
            n_estimators=100, sampling_strategy="all", replacement=True,
            bootstrap=False, n_jobs=-1, random_state=random_state,
        )
    if technique == "rusboost":
        return RUSBoostClassifier(random_state=random_state)
    if technique == "easy_ensemble":
        return EasyEnsembleClassifier(n_jobs=-1, random_state=random_state)
    if technique == "lgbm_is_unbalance":
        return LGBMClassifier(
            is_unbalance=True, n_jobs=-1, random_state=random_state, verbose=-1
        )
    if technique == "xgb_scale_pos_weight":
        if y_train is None:
            raise ValueError("xgb_scale_pos_weight needs y_train for n_neg / n_pos")
        n_pos = int(np.sum(y_train))
        return XGBClassifier(
            eval_metric="aucpr", n_jobs=-1, random_state=random_state,
            scale_pos_weight=(len(y_train) - n_pos) / max(n_pos, 1),
        )
    if technique == "xgb_focal_loss":
        return XGBClassifier(
            objective=FocalLoss(), eval_metric="aucpr", n_jobs=-1,
            random_state=random_state,
        )
    raise ValueError(f"Unknown standalone technique {technique!r}")


def get_model(
    classifier: str,
    technique: str = "none",
    random_state: int = config.RANDOM_STATE,
    y_train: np.ndarray | None = None,
    **kwargs: Any,
) -> Any:
    """Build a classifier, applying algorithm-level imbalance handling.

    Data-level techniques get the plain baseline classifier from
    baseline_models.get_baseline_models(), so they differ from the untreated
    baseline ONLY in the data they see. y_train is the (resampled) training
    fold; only xgb_scale_pos_weight reads it.
    """
    if technique in STANDALONE:
        if classifier != STANDALONE[technique]:
            raise ValueError(
                f"{technique} belongs to the {STANDALONE[technique]!r} family, "
                f"not {classifier!r}"
            )
        model = _standalone_model(technique, random_state, y_train)
    else:
        if classifier not in config.BASE_CLASSIFIERS:
            raise ValueError(f"Unknown classifier {classifier!r}")
        model = get_baseline_models(random_state)[classifier]
        if technique == "class_weight_balanced":
            if classifier not in CLASS_WEIGHT_CLASSIFIERS:
                raise ValueError(f"{classifier} does not support class_weight")
            model.set_params(class_weight="balanced")
        elif technique not in config.RESAMPLING_TECHNIQUES:
            raise ValueError(f"Unknown technique {technique!r}")
    if kwargs:
        model.set_params(**kwargs)
    return model
