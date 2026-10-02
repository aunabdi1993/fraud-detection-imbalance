"""
Tests for experiment_runner.

Synthetic data only — these run without the CSV. The sweep test logs to a
throwaway SQLite MLflow store under tmp_path, then checks that
inference_profiler.py can read back what the sweep wrote.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_loader import FraudDataset
from src.experiment_runner import run_single, run_sweep
from src.inference_profiler import iter_mlflow_models


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 2000
    y = (rng.random(n) < 0.03).astype(np.int8)
    X = rng.normal(size=(n, 5))
    X[y == 1] += 1.5
    return X, y


@pytest.fixture(scope="module")
def cv(data):
    return list(FraudDataset().cv_splits(*data))


def test_run_single_returns_one_row_per_fold(data, cv):
    rows = run_single("none", "logistic_regression", *data, cv)
    assert [r["fold"] for r in rows] == list(range(len(cv)))
    for r in rows:
        assert r["fit_seconds"] > 0 and r["inference_ms_per_record"] > 0


@pytest.mark.parametrize("technique", ["smote", "random_undersampling"])
def test_validation_fold_is_never_resampled(data, cv, technique):
    """Rule 1. If resampling leaked into the validation fold its size or
    fraud count would change; the confusion matrix must cover exactly the
    original validation rows."""
    X, y = data
    for r, (tr, va) in zip(run_single(technique, "logistic_regression", X, y, cv), cv):
        assert r["tp"] + r["fn"] == int(y[va].sum())
        assert r["tp"] + r["fp"] + r["tn"] + r["fn"] == len(va)
        # ...while the training fold really was resampled.
        assert r["n_train"] != len(tr)


@pytest.fixture
def sweep_dirs(tmp_path, data, monkeypatch):
    """Processed-data folder plus an isolated MLflow store and artifact root."""
    pytest.importorskip("mlflow")
    X, y = data
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    np.savez_compressed(
        data_dir / "splits.npz",
        X_train=X, y_train=y, X_val=X, y_val=y, X_test=X, y_test=y,
    )
    (data_dir / "manifest.json").write_text(
        json.dumps({"feature_names": [f"f{i}" for i in range(X.shape[1])]})
    )
    # Keep model artifacts out of the repository's mlruns/.
    monkeypatch.setattr(
        config, "MLFLOW_ARTIFACT_LOCATION", (tmp_path / "artifacts").as_uri()
    )
    return {
        "data_dir": data_dir,
        "csv": tmp_path / "results_raw.csv",
        "uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        "artifacts": tmp_path / "artifacts",
    }


PAIRS = [("none", "logistic_regression"), ("smote", "decision_tree")]


def _sweep(d):
    return run_sweep(
        output_csv=d["csv"], data_dir=d["data_dir"], tracking_uri=d["uri"],
        experiment="test_sweep", pairs=PAIRS,
    )


def test_sweep_writes_csv_logs_mlflow_and_feeds_the_profiler(sweep_dirs):
    import mlflow

    results = _sweep(sweep_dirs)
    assert len(results) == len(PAIRS) * config.N_FOLDS
    assert set(zip(results.technique, results.classifier)) == set(PAIRS)

    runs = mlflow.search_runs(experiment_names=["test_sweep"])
    assert len(runs) == len(PAIRS) * config.N_FOLDS
    assert {"params.technique", "params.fold", "metrics.auc_pr",
            "metrics.fit_seconds", "tags.git_commit"} <= set(runs.columns)
    assert any(sweep_dirs["artifacts"].rglob("MLmodel"))

    # The contract between the two modules: the profiler finds one model per
    # pair, taken from fold 0, with AUC-PR averaged over all five folds.
    found = {
        (pm.technique, pm.classifier): pm
        for pm in iter_mlflow_models(sweep_dirs["uri"], "test_sweep")
    }
    assert set(found) == set(PAIRS)
    for (technique, classifier), pm in found.items():
        mine = results[(results.technique == technique)
                       & (results.classifier == classifier)]
        assert pm.auc_pr_mean == pytest.approx(mine.auc_pr.mean())
        assert pm.n_folds == config.N_FOLDS


def test_sweep_resumes_without_rerunning_finished_pairs(sweep_dirs):
    import mlflow

    _sweep(sweep_dirs)
    n_runs = len(mlflow.search_runs(experiment_names=["test_sweep"]))
    results = _sweep(sweep_dirs)
    assert len(results) == len(PAIRS) * config.N_FOLDS
    assert len(mlflow.search_runs(experiment_names=["test_sweep"])) == n_runs
    assert not results.duplicated(["technique", "classifier", "fold"]).any()
    header = pd.read_csv(sweep_dirs["csv"], nrows=0).columns
    assert list(header) == list(results.columns)


def test_rerun_recomputes_only_the_selected_pairs(sweep_dirs):
    import mlflow

    first = _sweep(sweep_dirs)
    n_runs = len(mlflow.search_runs(experiment_names=["test_sweep"]))
    redo = [PAIRS[1]]
    results = run_sweep(
        output_csv=sweep_dirs["csv"], data_dir=sweep_dirs["data_dir"],
        tracking_uri=sweep_dirs["uri"], experiment="test_sweep", pairs=redo,
        rerun=True,
    )
    assert len(results) == len(first)
    assert not results.duplicated(["technique", "classifier", "fold"]).any()
    # Only the rerun pair produced new MLflow runs.
    assert (
        len(mlflow.search_runs(experiment_names=["test_sweep"]))
        == n_runs + config.N_FOLDS
    )
