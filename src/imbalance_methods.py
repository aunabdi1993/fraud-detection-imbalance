"""
imbalance_methods.py — The 15 techniques under comparison.

Maps to: Chapter 3 sec 3.5, Chapter 5 sec 5.2.
STATUS: STUB — implement in Month 5-6 (Week 17-20).

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

Implementation notes per technique:
  random_undersampling  imblearn.under_sampling.RandomUnderSampler
  random_oversampling   imblearn.over_sampling.RandomOverSampler
  smote                 imblearn.over_sampling.SMOTE (Chawla et al. 2002)
                        k_neighbours=5 default; with 331 training frauds this
                        is safe, but log any fold where n_minority <= k.
  smote_tomek           imblearn.combine.SMOTETomek (Batista et al. 2004)
  smote_enn             imblearn.combine.SMOTEENN
  adasyn                imblearn.over_sampling.ADASYN (He et al. 2008)
                        Can fail with "No samples will be generated" when the
                        minority is very sparse — catch and log, do not crash
                        the sweep.
  soa_s                 Selective oversampling (Gnip et al. 2021). No library
                        implementation; identify minority outliers first
                        (e.g. isolation forest or LOF), exclude them, then
                        oversample the remaining core. Document the deviation
                        from the paper honestly in Chapter 3.

Algorithm-level techniques are model constructors, not samplers:
  class_weight_balanced     sklearn class_weight="balanced"
  balanced_random_forest    imblearn.ensemble.BalancedRandomForestClassifier
  rusboost                  imblearn.ensemble.RUSBoostClassifier (Seiffert 2010)
  easy_ensemble             imblearn.ensemble.EasyEnsembleClassifier (Liu 2009)
  xgb_scale_pos_weight      scale_pos_weight = n_neg / n_pos (~578)
  lgbm_is_unbalance         LightGBM is_unbalance=True
  xgb_focal_loss            custom objective; imbalance-xgboost or hand-rolled
                            grad/hess. Tune gamma in {1, 2, 5}.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import config


def get_sampler(name: str, random_state: int = config.RANDOM_STATE) -> Any:
    """Return a fitted-on-demand imblearn sampler, or None for 'none'.

    Raises NotImplementedError until Month 5-6.
    """
    raise NotImplementedError("Implement in Month 5-6 — see module docstring")


def get_model(
    classifier: str,
    technique: str = "none",
    random_state: int = config.RANDOM_STATE,
    **kwargs: Any,
) -> Any:
    """Build a classifier, applying algorithm-level imbalance handling."""
    raise NotImplementedError("Implement in Month 5-6 — see module docstring")


def is_algorithm_level(technique: str) -> bool:
    """Algorithm-level techniques modify the model, not the data."""
    return technique in config.ALGORITHM_TECHNIQUES


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
    raise NotImplementedError("Implement in Month 5-6 — see module docstring")
