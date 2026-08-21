"""
Tests for baseline_models.

Synthetic data only — these run without the CSV.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.baseline_models import get_baseline_models, train_baselines


@pytest.fixture
def imbalanced_split():
    """~1% positive rate, informative but imperfect features."""
    rng = np.random.default_rng(0)
    n_train, n_val, n_features = 2000, 500, 5

    def make(n):
        y = np.zeros(n, dtype=int)
        y[: max(1, n // 100)] = 1
        rng.shuffle(y)
        X = rng.normal(size=(n, n_features))
        X[y == 1] += 1.5  # weak signal separating the classes
        return X, y

    X_train, y_train = make(n_train)
    X_val, y_val = make(n_val)
    return X_train, y_train, X_val, y_val


def test_get_baseline_models_returns_all_five():
    models = get_baseline_models()
    assert set(models) == {
        "dummy_most_frequent",
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
    }


def test_get_baseline_models_is_seeded_deterministically():
    a = get_baseline_models(random_state=42)
    b = get_baseline_models(random_state=42)
    assert a["random_forest"].random_state == b["random_forest"].random_state == 42


def test_train_baselines_returns_one_row_per_model(imbalanced_split):
    rows = train_baselines(*imbalanced_split)
    assert {r["technique"] for r in rows} == set(get_baseline_models())


def test_dummy_classifier_has_zero_recall(imbalanced_split):
    """The core illustration of Chapter 1 sec 1.1: high accuracy, zero recall."""
    rows = train_baselines(*imbalanced_split)
    dummy = next(r for r in rows if r["technique"] == "dummy_most_frequent")
    assert dummy["recall"] == 0.0
    assert dummy["tp"] == 0


def test_non_dummy_baselines_beat_dummy_on_auc_pr(imbalanced_split):
    """Sanity check: models given a real (if weak) signal should separate
    classes better than a constant predictor, on this synthetic split."""
    rows = train_baselines(*imbalanced_split)
    by_name = {r["technique"]: r for r in rows}
    dummy_auc_pr = by_name["dummy_most_frequent"]["auc_pr"]
    for name in ("logistic_regression", "random_forest", "xgboost"):
        assert by_name[name]["auc_pr"] >= dummy_auc_pr


def test_threshold_chosen_on_validation_not_default(imbalanced_split):
    """Thresholds should not silently default to 0.5 (rule 3)."""
    rows = train_baselines(*imbalanced_split)
    thresholds = {r["technique"]: r["threshold"] for r in rows}
    assert thresholds["logistic_regression"] != 0.5
