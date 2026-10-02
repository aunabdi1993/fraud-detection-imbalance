"""
Tests for imbalance_methods.

Synthetic data only — these run without the CSV.
"""

from __future__ import annotations

import ast
import pickle
from pathlib import Path

import numpy as np
import pytest

from src import config
from src.baseline_models import get_baseline_models
from src.imbalance_methods import (
    STANDALONE,
    FocalLoss,
    SelectiveOversampler,
    get_model,
    resample_fold,
    sweep_pairs,
)

OVERSAMPLERS = ["random_oversampling", "smote", "adasyn", "soa_s"]


@pytest.fixture(scope="module")
def data():
    """~3% positive, overlapping classes so cleaning steps have work to do."""
    rng = np.random.default_rng(0)
    n = 3000
    y = (rng.random(n) < 0.03).astype(np.int8)
    X = rng.normal(size=(n, 5))
    X[y == 1] += 1.0
    return X, y


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------
def test_sweep_covers_every_configured_technique_once_per_pair():
    pairs = sweep_pairs()
    assert len(pairs) == len(set(pairs)) == 41
    assert {t for t, _ in pairs} == set(config.ALL_TECHNIQUES)


def test_data_loader_never_imports_imblearn():
    """CLAUDE.md rule 1: resampling cannot happen before the split if the
    module that makes the split cannot resample."""
    tree = ast.parse(Path("src/data_loader.py").read_text())
    imported = {
        alias.name for node in ast.walk(tree)
        if isinstance(node, ast.Import) for alias in node.names
    } | {
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any(name.startswith("imblearn") for name in imported)


# ---------------------------------------------------------------------------
# Data level
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "technique", ["none", "class_weight_balanced", "xgb_focal_loss"]
)
def test_non_resampling_techniques_return_data_untouched(data, technique):
    X, y = data
    X_out, y_out = resample_fold(X, y, technique)
    assert X_out is X and y_out is y


@pytest.mark.parametrize("technique", OVERSAMPLERS)
def test_oversamplers_balance_and_keep_every_original_row(data, technique):
    X, y = data
    X_out, y_out = resample_fold(X, y, technique)
    n_neg = int((y == 0).sum())
    assert int((y_out == 0).sum()) == n_neg
    # ADASYN targets balance approximately, by design.
    assert int(y_out.sum()) == pytest.approx(n_neg, rel=0.05)
    np.testing.assert_array_equal(X_out[: len(X)], X)


def test_undersampling_shrinks_majority_to_minority(data):
    X, y = data
    _, y_out = resample_fold(X, y, "random_undersampling")
    assert int((y_out == 0).sum()) == int(y_out.sum()) == int(y.sum())


@pytest.mark.parametrize("technique", ["smote_tomek", "smote_enn"])
def test_hybrid_samplers_clean_both_classes_after_oversampling(data, technique):
    """Batista et al. (2004) remove noisy points from BOTH classes. Checking
    only that some rows disappear missed a bug where the majority class was
    never touched (and, on the real data, nothing was removed at all)."""
    X, y = data
    _, y_smote = resample_fold(X, y, "smote")
    _, y_out = resample_fold(X, y, technique)
    for label in (0, 1):
        assert (y_out == label).sum() < (y_smote == label).sum()


def test_soa_s_never_synthesises_from_an_outlier(data):
    """The point of SOA-S: an isolated fraud is not used as a SMOTE seed."""
    X, y = data
    X = X.copy()
    outlier = np.full(X.shape[1], 25.0)
    X[np.flatnonzero(y == 1)[0]] = outlier
    X_out, _ = SelectiveOversampler().fit_resample(X, y)
    synthetic = X_out[len(X):]
    # SMOTE interpolates between neighbours, so a synthetic point anywhere
    # near the outlier could only have come from seeding on it.
    assert np.linalg.norm(synthetic - outlier, axis=1).min() > 10


def test_adasyn_failure_falls_back_instead_of_crashing(data, monkeypatch):
    from imblearn.over_sampling import ADASYN

    def no_samples(self, X, y):
        raise RuntimeError("No samples will be generated")

    monkeypatch.setattr(ADASYN, "fit_resample", no_samples)
    X, y = data
    X_out, y_out = resample_fold(X, y, "adasyn")
    assert X_out is X and y_out is y


def test_other_sampler_failures_are_not_swallowed(data, monkeypatch):
    from imblearn.over_sampling import SMOTE

    def broken(self, X, y):
        raise RuntimeError("boom")

    monkeypatch.setattr(SMOTE, "fit_resample", broken)
    with pytest.raises(RuntimeError, match="boom"):
        resample_fold(*data, "smote")


# ---------------------------------------------------------------------------
# Algorithm level
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("technique, classifier", sweep_pairs())
def test_every_pair_fits_and_scores(data, technique, classifier):
    X, y = data
    X_fit, y_fit = resample_fold(X, y, technique)
    model = get_model(classifier, technique, y_train=y_fit).fit(X_fit, y_fit)
    proba = model.predict_proba(X[:50])[:, 1]
    assert np.all((proba >= 0) & (proba <= 1))


def test_resampling_techniques_use_the_untreated_baseline_classifier():
    """A SMOTE model must differ from the baseline ONLY in its data."""
    for clf in config.BASE_CLASSIFIERS:
        assert (
            get_model(clf, "smote").get_params()
            == get_baseline_models()[clf].get_params()
        )


def test_class_weight_balanced_is_applied():
    model = get_model("logistic_regression", "class_weight_balanced")
    assert model.class_weight == "balanced"


@pytest.mark.parametrize(
    "classifier, technique",
    [("xgboost", "class_weight_balanced"), ("xgboost", "rusboost"),
     ("logistic_regression", "made_up")],
)
def test_invalid_combinations_are_rejected(classifier, technique):
    with pytest.raises(ValueError):
        get_model(classifier, technique)


def test_scale_pos_weight_is_computed_from_the_training_fold():
    y = np.array([0] * 90 + [1] * 10)
    model = get_model("xgboost", "xgb_scale_pos_weight", y_train=y)
    assert model.scale_pos_weight == pytest.approx(9.0)


def test_standalone_techniques_are_in_the_algorithm_list():
    assert set(STANDALONE) <= set(config.ALGORITHM_TECHNIQUES)


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------
def _focal(margin, y, gamma):
    p = 1 / (1 + np.exp(-margin))
    return -(y * (1 - p) ** gamma * np.log(p) + (1 - y) * p**gamma * np.log(1 - p))


def test_focal_loss_with_gamma_zero_is_log_loss():
    margin = np.linspace(-4, 4, 41)
    p = 1 / (1 + np.exp(-margin))
    for label in (0, 1):
        y = np.full_like(margin, label)
        grad, hess = FocalLoss(gamma=0.0)(y, margin)
        np.testing.assert_allclose(grad, p - y, atol=1e-9)
        np.testing.assert_allclose(hess, p * (1 - p), atol=1e-9)


@pytest.mark.parametrize("label", [0, 1])
def test_focal_loss_derivatives_match_finite_differences(label):
    gamma, h = 2.0, 1e-5
    margin = np.linspace(-4, 4, 41)
    y = np.full_like(margin, label)
    grad, hess = FocalLoss(gamma)(y, margin)
    num_grad = (_focal(margin + h, y, gamma) - _focal(margin - h, y, gamma)) / (2 * h)
    np.testing.assert_allclose(grad, num_grad, atol=1e-6)
    num_hess = (
        FocalLoss(gamma)(y, margin + h)[0] - FocalLoss(gamma)(y, margin - h)[0]
    ) / (2 * h)
    unclipped = num_hess > config.FOCAL_MIN_HESSIAN
    np.testing.assert_allclose(hess[unclipped], num_hess[unclipped], atol=1e-5)
    assert np.all(hess >= config.FOCAL_MIN_HESSIAN)


def test_focal_model_predict_proba_is_sigmoid_of_margin(data):
    """XGBoost applies the sigmoid to a custom objective's output itself
    (3.x). If a release stops doing so, scores become raw margins and every
    focal-loss metric is wrong without any error — this test catches it."""
    X, y = data
    model = get_model("xgboost", "xgb_focal_loss").fit(X, y)
    margin = model.predict(X[:20], output_margin=True)
    np.testing.assert_allclose(
        model.predict_proba(X[:20])[:, 1], 1 / (1 + np.exp(-margin)), rtol=1e-5
    )


def test_focal_model_pickles(data):
    """MLflow logs it with cloudpickle; plain pickle is the stricter test."""
    X, y = data
    model = get_model("xgboost", "xgb_focal_loss").fit(X, y)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(
        restored.predict_proba(X[:5]), model.predict_proba(X[:5])
    )
