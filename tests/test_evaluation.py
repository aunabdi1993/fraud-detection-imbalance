"""
Tests for evaluation.

Synthetic data only — these run without the CSV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    CostModel,
    auc_pr,
    bootstrap_ci,
    choose_threshold,
    evaluate,
    friedman_test,
    nemenyi_critical_difference,
    precision_at_k,
    recall_at_k,
    results_table,
)


@pytest.fixture
def imbalanced():
    """1000 records, 1% positive, informative but imperfect scores."""
    rng = np.random.default_rng(0)
    y = np.zeros(1000, dtype=int)
    y[:10] = 1
    score = np.where(y == 1, rng.beta(6, 2, 1000), rng.beta(2, 6, 1000))
    return y, score


def test_auc_pr_baseline_equals_positive_rate():
    """A random scorer should score approximately the positive rate."""
    rng = np.random.default_rng(1)
    y = np.zeros(10_000, dtype=int)
    y[:100] = 1
    rng.shuffle(y)
    assert auc_pr(y, rng.random(10_000)) == pytest.approx(0.01, abs=0.01)


def test_auc_pr_perfect_scorer():
    y = np.array([0, 0, 0, 1, 1])
    assert auc_pr(y, y.astype(float)) == pytest.approx(1.0)


def test_evaluate_returns_all_expected_keys(imbalanced):
    y, s = imbalanced
    row = evaluate(y, s, threshold=0.5, alert_budget=20)
    for k in ("auc_pr", "precision", "recall", "f1", "mcc", "tp", "fp",
              "fn", "tn", "total_cost", "precision_at_20"):
        assert k in row
    assert row["tp"] + row["fn"] == y.sum()
    assert row["tp"] + row["fp"] == row["n_alerts"]


def test_evaluate_handles_all_negative_predictions(imbalanced):
    """A threshold of 1.0 predicts nothing; metrics must be 0, not NaN."""
    y, s = imbalanced
    row = evaluate(y, s, threshold=1.0)
    assert row["recall"] == 0.0
    assert row["precision"] == 0.0
    assert not np.isnan(row["f1"])


def test_precision_and_recall_at_k(imbalanced):
    y, s = imbalanced
    assert 0.0 <= precision_at_k(y, s, 10) <= 1.0
    # recall@k is monotone non-decreasing in k
    assert recall_at_k(y, s, 100) >= recall_at_k(y, s, 10)


def test_precision_at_k_rejects_nonpositive_k(imbalanced):
    y, s = imbalanced
    with pytest.raises(ValueError):
        precision_at_k(y, s, 0)


def test_chosen_threshold_beats_default_f1(imbalanced):
    """The whole point of threshold tuning under imbalance."""
    y, s = imbalanced
    thr = choose_threshold(y, s, objective="f1")
    assert evaluate(y, s, thr)["f1"] >= evaluate(y, s, 0.5)["f1"]


def test_cost_objective_reduces_cost(imbalanced):
    y, s = imbalanced
    cm = CostModel(cost_fn=100, cost_fp=5)
    thr = choose_threshold(y, s, objective="cost", cost_model=cm)
    assert (
        evaluate(y, s, thr, cost_model=cm)["total_cost"]
        <= evaluate(y, s, 0.5, cost_model=cm)["total_cost"]
    )


def test_precision_objective_respects_constraint(imbalanced):
    y, s = imbalanced
    thr = choose_threshold(y, s, objective="precision", min_precision=0.8)
    assert evaluate(y, s, thr)["precision"] >= 0.79


def test_unknown_objective_raises(imbalanced):
    y, s = imbalanced
    with pytest.raises(ValueError):
        choose_threshold(y, s, objective="nonsense")


def test_bootstrap_ci_brackets_point_estimate(imbalanced):
    y, s = imbalanced
    point, lo, hi = bootstrap_ci(y, s, n_boot=200)
    assert lo <= point <= hi
    assert 0.0 <= lo and hi <= 1.0


def test_bootstrap_ci_is_reproducible(imbalanced):
    y, s = imbalanced
    a = bootstrap_ci(y, s, n_boot=100, random_state=42)
    b = bootstrap_ci(y, s, n_boot=100, random_state=42)
    assert a == b


def test_friedman_detects_a_real_difference():
    """Technique C is consistently best across folds."""
    scores = pd.DataFrame({
        "A": [0.70, 0.71, 0.69, 0.70, 0.72],
        "B": [0.75, 0.76, 0.74, 0.75, 0.77],
        "C": [0.82, 0.83, 0.81, 0.84, 0.82],
    })
    r = friedman_test(scores)
    assert r["significant_at_05"]
    assert min(r["mean_ranks"], key=r["mean_ranks"].get) == "C"


def test_friedman_requires_three_techniques():
    with pytest.raises(ValueError):
        friedman_test(pd.DataFrame({"A": [0.1, 0.2], "B": [0.3, 0.4]}))


def test_critical_difference_shrinks_with_more_folds():
    assert nemenyi_critical_difference(5, 10) < nemenyi_critical_difference(5, 5)


def test_results_table_formats_mean_and_std():
    rows = [
        {"technique": "smote", "auc_pr": 0.8, "precision": 0.7,
         "recall": 0.6, "f1": 0.65, "mcc": 0.6},
        {"technique": "smote", "auc_pr": 0.82, "precision": 0.72,
         "recall": 0.62, "f1": 0.67, "mcc": 0.62},
        {"technique": "none", "auc_pr": 0.70, "precision": 0.9,
         "recall": 0.3, "f1": 0.45, "mcc": 0.5},
        {"technique": "none", "auc_pr": 0.71, "precision": 0.91,
         "recall": 0.31, "f1": 0.46, "mcc": 0.51},
    ]
    t = results_table(rows)
    assert "±" in t.loc["smote", "auc_pr"]
    assert list(t.index)[0] == "smote"  # sorted by primary metric
