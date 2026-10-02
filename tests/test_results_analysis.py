"""
Tests for results_analysis.

Synthetic data only — these run without the CSV or a finished sweep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src import results_analysis as ra
from src.data_loader import Splits
from src.evaluation import nemenyi_critical_difference


def _results(rng=None) -> pd.DataFrame:
    """A fake sweep: 3 pairs x 5 folds with known mean AUC-PR."""
    rng = rng or np.random.default_rng(0)
    means = {("smote", "xgboost"): 0.80, ("none", "xgboost"): 0.70,
             ("none", "logistic_regression"): 0.60}
    rows = []
    for (t, c), m in means.items():
        for fold in range(config.N_FOLDS):
            rows.append({
                "technique": t, "classifier": c, "fold": fold,
                "auc_pr": m + rng.normal(0, 0.01), "auc_roc": 0.9,
                f"precision_at_{config.ALERT_BUDGET}": 0.5, "f1": 0.5,
                "mcc": 0.5, "fit_seconds": 1.0, "n_train": 100,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def splits():
    rng = np.random.default_rng(0)

    def make(n):
        y = (rng.random(n) < 0.03).astype(np.int8)
        X = rng.normal(size=(n, 4))
        X[y == 1] += 1.5
        return X, y

    (a, b), (c, d), (e, f) = make(3000), make(2000), make(2000)
    return Splits(a, b, c, d, e, f, feature_names=["f0", "f1", "f2", "f3"])


@pytest.fixture(scope="module")
def finals(splits):
    return ra.fit_final_models(
        [("none", "logistic_regression"), ("smote", "logistic_regression")], splits
    )


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------
def test_cv_summary_ranks_pairs_by_mean_auc_pr():
    table = ra.cv_summary(_results())
    assert list(table["rank"]) == [1, 2, 3]
    assert list(table["technique"]) == ["smote", "none", "none"]
    assert table["auc_pr"].is_monotonic_decreasing


def test_top_pairs_and_best_untreated():
    results = _results()
    assert ra.top_pairs(results, 2) == [("smote", "xgboost"), ("none", "xgboost")]
    assert ra.best_untreated(results) == ("none", "xgboost")


def test_technique_types():
    assert ra.technique_type("none") == "untreated"
    assert ra.technique_type("smote") == "data-level"
    assert ra.technique_type("rusboost") == "algorithm-level"


def test_rank_comparison_flags_only_large_rank_gaps():
    """One treatment always wins, one always loses; with 20 blocks the
    Nemenyi CD separates them but not the two in between."""
    rng = np.random.default_rng(1)
    n = 20
    scores = pd.DataFrame({
        "best": 0.9 + rng.normal(0, 0.001, n),
        "a": 0.5 + rng.normal(0, 0.05, n),
        "b": 0.5 + rng.normal(0, 0.05, n),
        "worst": 0.1 + rng.normal(0, 0.001, n),
    })
    result = ra.rank_comparison(scores, "toy", reference="a")
    flags = result.table.set_index("treatment")
    assert not flags.loc["best", "worse_than_best"]
    assert flags.loc["worst", "worse_than_best"]
    assert not flags.loc["b", "differs_from_reference"]
    assert result.p_value < 0.05


def test_nemenyi_beyond_the_tabulated_range():
    """41 pairs exceed Demsar's k <= 20 table; the computed value must
    continue it smoothly rather than raise."""
    cd20 = nemenyi_critical_difference(20, 5)
    cd21 = nemenyi_critical_difference(21, 5)
    assert cd21 > cd20
    assert nemenyi_critical_difference(41, 5) > cd21


def test_data_level_scores_block_by_classifier_and_fold():
    """Works on a partial sweep: only the techniques present, config order."""
    scores = ra.data_level_scores(_results())
    assert scores.index.names == ["classifier", "fold"]
    assert list(scores.columns) == ["none", "smote"]


def test_latency_table_joins_cv_auc_pr():
    results = _results()
    latency = pd.DataFrame({
        "technique": ["smote", "none"], "classifier": ["xgboost", "xgboost"],
        "batch_size": [1, 1], "p50_ms": [0.1, 0.1], "p95_ms": [0.2, 0.2],
        "p99_ms": [0.3, 150.0], "mean_ms": [0.1, 0.1],
        "model_size_mb": [1.0, 1.0], "cpu_utilisation": [1.0, 1.0],
    })
    table = ra.latency_table(latency, results).set_index("technique")
    assert table.loc["smote", "auc_pr"] == pytest.approx(
        results[results.technique == "smote"].auc_pr.mean())
    assert table.loc["smote", "within_budget"]
    assert not table.loc["none", "within_budget"]


# ---------------------------------------------------------------------------
# Stage 2: rule 3 guards
# ---------------------------------------------------------------------------
def test_thresholds_never_depend_on_test_labels(splits, finals):
    """Rule 3. Shuffling the test labels must not move a single threshold,
    in the final models or in any operating point."""
    shuffled = Splits(
        splits.X_train, splits.y_train, splits.X_val, splits.y_val,
        splits.X_test, np.random.default_rng(0).permutation(splits.y_test),
    )
    a = ra.operating_points(finals[0], splits)
    b = ra.operating_points(finals[0], shuffled)
    pd.testing.assert_series_equal(a["threshold"], b["threshold"])


def test_alert_budget_operating_point_hits_its_rate(splits, finals):
    points = ra.operating_points(finals[0], splits).set_index("rule")
    rate = config.ALERT_BUDGET / config.TRANSACTIONS_PER_DAY
    threshold = points.loc[f"{config.ALERT_BUDGET} alerts/day", "threshold"]
    # On validation, where it was chosen, the alert rate is as asked.
    assert np.mean(finals[0].p_val >= threshold) == pytest.approx(rate, abs=1e-3)


def test_paired_bootstrap_of_a_model_against_itself_is_zero(splits, finals):
    delta, lo, hi = ra.paired_bootstrap_delta(
        splits.y_test, finals[0].p_test, finals[0].p_test, n_boot=50)
    assert delta == lo == hi == 0


def test_paired_bootstrap_detects_a_clearly_worse_model(splits, finals):
    noise = np.random.default_rng(0).random(len(splits.y_test))
    delta, lo, hi = ra.paired_bootstrap_delta(
        splits.y_test, noise, finals[0].p_test, n_boot=200)
    assert hi < 0


def test_test_table_compares_against_the_cv_selected_model(splits, finals):
    table = ra.test_table(finals, splits)
    assert table.loc[0, "delta_vs_selected"] == 0
    assert not table.loc[0, "significant"]
    assert (table["auc_pr_lo"] <= table["auc_pr"]).all()
    assert (table["auc_pr"] <= table["auc_pr_hi"]).all()


def test_threshold_sweep_counts_are_consistent(splits, finals):
    sweep = ra.threshold_sweep(splits.y_test, finals[0].p_test)
    n_pos = int(splits.y_test.sum())
    assert (sweep["fn"] <= n_pos).all()
    # Raising the threshold can only trade false alarms for misses.
    assert sweep["fp"].is_monotonic_decreasing
    assert sweep["fn"].is_monotonic_increasing


def test_shap_importance_covers_every_feature(splits, finals):
    pytest.importorskip("shap")
    imp = ra.shap_importance(finals[0], splits, splits.feature_names)
    assert sorted(imp["feature"]) == sorted(splits.feature_names)
    assert imp["mean_abs_shap"].is_monotonic_decreasing


def test_figures_build_and_html_is_written_without_an_exporter(
    tmp_path, splits, finals, monkeypatch
):
    """PNG export needs Chrome; without it the HTML must still be written."""
    import plotly.graph_objects as go

    def no_chrome(self, *args, **kwargs):
        raise RuntimeError("no Chrome")

    monkeypatch.setattr(go.Figure, "write_image", no_chrome)
    tests = ra.test_table(finals, splits)
    fig = ra.fig_pr_curves(finals, splits.y_test, tests, reference=finals[1].label)
    ra.save_figure(fig, "pr", out_dir=tmp_path)
    assert (tmp_path / "pr.html").exists()
    ra.fig_confusion_matrices(finals, splits.y_test)
    points = ra.operating_points(finals[0], splits)
    sweep = ra.threshold_sweep(splits.y_test, finals[0].p_test)
    ra.fig_errors_by_threshold(sweep, points)
    ra.fig_cost_by_threshold(sweep, points)
