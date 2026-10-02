"""
Tests for inference_profiler.

Synthetic data and small models only — these run without the CSV. The MLflow
tests use a throwaway SQLite store under pytest's tmp_path.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import config
from src.inference_profiler import (
    ENVIRONMENT_JSON,
    RAW_CSV,
    RESULTS_CSV,
    LatencyResult,
    ProfiledModel,
    TimingProtocol,
    iter_mlflow_models,
    profile_all,
    profile_model,
    profile_models,
    serialised_size_mb,
    single_threaded,
    summarise_runs,
)

# Small enough that the whole module runs in seconds.
FAST = TimingProtocol(
    batch_sizes=(1, 50), n_trials=40, n_warmup=5, n_batch_trials=8,
    n_batch_warmup=2,
)


@pytest.fixture(scope="module")
def data():
    """~2% positive rate, weak signal, same shape conventions as the splits."""
    rng = np.random.default_rng(0)
    n, n_features = 1000, 6
    y = np.zeros(n, dtype=int)
    y[: n // 50] = 1
    rng.shuffle(y)
    X = rng.normal(size=(n, n_features))
    X[y == 1] += 1.5
    return X, y


@pytest.fixture(scope="module")
def forest(data):
    """n_jobs=-1, as in the baselines — the case thread pinning must handle."""
    X, y = data
    return RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=0).fit(X, y)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def test_profile_model_returns_one_result_per_batch_size(data, forest):
    X, _ = data
    results = profile_model(forest, X, FAST)
    assert [r.batch_size for r in results] == [1, 50]


def test_trial_counts_follow_protocol(data, forest):
    """Single records get n_trials; batches get the smaller n_batch_trials."""
    X, _ = data
    by_size = {r.batch_size: r for r in profile_model(forest, X, FAST)}
    assert by_size[1].timings_ms.size == FAST.n_trials
    assert by_size[50].timings_ms.size == FAST.n_batch_trials


def test_summary_percentiles_are_ordered(data, forest):
    X, _ = data
    for result in profile_model(forest, X, FAST):
        s = result.summary()
        ordered = [s[k] for k in ("min_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms")]
        assert 0 < ordered[0] and ordered == sorted(ordered)


def test_summary_statistics_on_known_timings():
    """1..100 ms at batch size 10: every derived column is checkable by hand."""
    result = LatencyResult(
        batch_size=10,
        timings_ms=np.arange(1, 101, dtype=float),
        cpu_utilisation=1.0,
        peak_predict_mb=0.0,
    )
    s = result.summary()
    assert s["mean_ms"] == pytest.approx(50.5)
    assert s["p50_ms"] == pytest.approx(50.5)
    assert s["p99_ms"] == pytest.approx(99.01)
    assert s["per_record_ms"] == pytest.approx(5.05)
    assert s["throughput_per_s"] == pytest.approx(1e3 / s["per_record_ms"])


def test_single_threaded_pins_then_restores_n_jobs(forest):
    with single_threaded(forest):
        assert forest.n_jobs == 1
    assert forest.n_jobs == -1


def test_threads_stay_pinned_after_multithreaded_work(data):
    """A fit immediately before profiling leaves BLAS workers spinning. The
    settle pause in single_threaded() must stop that CPU time being charged
    to the timed loop."""
    X, y = data
    model = LogisticRegression(max_iter=1000).fit(X, y)
    for result in profile_model(model, X, FAST):
        assert result.cpu_utilisation <= config.SINGLE_THREAD_CPU_TOLERANCE


def test_every_model_sees_the_same_rows(data, forest):
    """Re-seeding per model is what makes technique comparisons paired. A
    model that echoes its input exposes which rows it was given."""

    class Recorder:
        def __init__(self):
            self.seen = []

        def predict_proba(self, X):
            self.seen.append(X.copy())
            return np.zeros((len(X), 2))

    a, b = Recorder(), Recorder()
    profile_model(a, data[0], FAST)
    profile_model(b, data[0], FAST)
    assert len(a.seen) == len(b.seen)
    assert all(np.array_equal(x, y) for x, y in zip(a.seen, b.seen))


def test_serialised_size_grows_with_ensemble_size(data):
    X, y = data
    small = RandomForestClassifier(n_estimators=2, random_state=0).fit(X, y)
    large = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    assert serialised_size_mb(large) > serialised_size_mb(small)


def test_profile_models_builds_summary_and_raw_tables(data, forest):
    X, _ = data
    models = [
        ProfiledModel("none", "random_forest", forest, auc_pr_mean=0.7),
        ProfiledModel(
            "none", "logistic_regression",
            LogisticRegression(max_iter=1000).fit(*data), auc_pr_mean=0.6,
        ),
    ]
    summary, raw = profile_models(models, X, FAST)
    assert len(summary) == len(models) * len(FAST.batch_sizes)
    assert len(raw) == len(models) * (FAST.n_trials + FAST.n_batch_trials)
    assert set(summary["auc_pr_mean"]) == {0.6, 0.7}
    assert (summary["model_size_mb"] > 0).all()


# ---------------------------------------------------------------------------
# Run selection (pure pandas, no MLflow needed)
# ---------------------------------------------------------------------------
def _runs(rows):
    """Mimic the columns mlflow.search_runs() returns."""
    return pd.DataFrame(
        rows,
        columns=["run_id", "status", "start_time", "params.technique",
                 "params.classifier", "params.fold", "metrics.auc_pr",
                 "metrics.fit_seconds"],
    )


def test_summarise_runs_averages_metrics_over_folds():
    table = summarise_runs(_runs([
        ("a", "FINISHED", 1, "smote", "xgboost", "0", 0.70, 10.0),
        ("b", "FINISHED", 2, "smote", "xgboost", "1", 0.80, 20.0),
        ("c", "FINISHED", 3, "none", "xgboost", "0", 0.50, 5.0),
    ]))
    row = table.set_index(["technique", "classifier"]).loc[("smote", "xgboost")]
    assert row["auc_pr_mean"] == pytest.approx(0.75)
    assert row["fit_seconds_mean"] == pytest.approx(15.0)
    assert row["n_folds"] == 2
    assert len(table) == 2


def test_summarise_runs_orders_candidates_by_fold():
    """Fold "10" must sort after fold "2" — numerically, not as strings."""
    table = summarise_runs(_runs([
        ("f10", "FINISHED", 1, "smote", "xgboost", "10", 0.7, 1.0),
        ("f2", "FINISHED", 2, "smote", "xgboost", "2", 0.7, 1.0),
        ("f0", "FINISHED", 3, "smote", "xgboost", "0", 0.7, 1.0),
    ]))
    assert table.loc[0, "run_ids"] == ["f0", "f2", "f10"]


def test_summarise_runs_drops_unfinished_and_superseded_runs():
    table = summarise_runs(_runs([
        ("old", "FINISHED", 1, "smote", "xgboost", "0", 0.10, 1.0),
        ("new", "FINISHED", 2, "smote", "xgboost", "0", 0.90, 1.0),
        ("dead", "FAILED", 3, "smote", "xgboost", "1", 0.99, 1.0),
    ]))
    assert table.loc[0, "run_ids"] == ["new"]
    assert table.loc[0, "auc_pr_mean"] == pytest.approx(0.90)


def test_summarise_runs_handles_empty_experiment():
    assert summarise_runs(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# MLflow round trip
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sweep(tmp_path_factory, data):
    """A miniature sweep logged the way experiment_runner.py must log it.

    smote/logistic_regression logs a model on fold 1 only, so the profiler
    has to fall through fold 0 to find it.
    """
    pytest.importorskip("mlflow")
    import mlflow.sklearn

    root = tmp_path_factory.mktemp("sweep")
    uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    # Explicit artifact location, or MLflow writes models into ./mlruns.
    mlflow.create_experiment(
        config.MLFLOW_EXPERIMENT, artifact_location=(root / "artifacts").as_uri()
    )
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    X, y = data
    model_name_kw = (
        "name" if int(mlflow.__version__.split(".")[0]) >= 3 else "artifact_path"
    )
    plan = {
        ("none", "random_forest"): {0: 0.60, 1: 0.70},
        ("smote", "logistic_regression"): {0: 0.40, 1: 0.50},
    }
    expected_run = {}
    for (technique, classifier), folds in plan.items():
        model_fold = 1 if technique == "smote" else 0
        for fold, auc in folds.items():
            with mlflow.start_run() as run:
                mlflow.log_params(
                    {"technique": technique, "classifier": classifier, "fold": fold}
                )
                mlflow.log_metrics({"auc_pr": auc, "fit_seconds": 1.0 + fold})
                if fold == model_fold:
                    model = (
                        RandomForestClassifier(n_estimators=5, random_state=0)
                        if classifier == "random_forest"
                        else LogisticRegression(max_iter=1000)
                    ).fit(X, y)
                    mlflow.sklearn.log_model(
                        model,
                        serialization_format=config.MLFLOW_SERIALIZATION_FORMAT,
                        **{model_name_kw: config.MLFLOW_MODEL_ARTIFACT},
                    )
                    expected_run[(technique, classifier)] = run.info.run_id

    data_dir = root / "processed"
    data_dir.mkdir()
    np.savez_compressed(
        data_dir / "splits.npz",
        X_train=X, y_train=y, X_val=X, y_val=y, X_test=X, y_test=y,
    )
    (data_dir / "manifest.json").write_text(json.dumps({
        "feature_names": [f"f{i}" for i in range(X.shape[1])],
        "validation_report": {"file_sha256": "synthetic"},
    }))
    return {"uri": uri, "data_dir": data_dir, "root": root,
            "expected_run": expected_run}


def test_iter_mlflow_models_finds_a_model_per_pair(sweep):
    found = {
        (pm.technique, pm.classifier): pm
        for pm in iter_mlflow_models(sweep["uri"], config.MLFLOW_EXPERIMENT)
    }
    assert set(found) == set(sweep["expected_run"])
    for key, pm in found.items():
        assert pm.run_id == sweep["expected_run"][key]
        assert hasattr(pm.model, "predict_proba")
    assert found[("none", "random_forest")].auc_pr_mean == pytest.approx(0.65)
    assert found[("smote", "logistic_regression")].n_folds == 2


def test_unloadable_model_raises_instead_of_being_skipped(sweep, monkeypatch):
    """A fold with no model is skipped; a model that exists but will not load
    must stop the run, or a technique silently drops out of the comparison."""
    import mlflow.sklearn

    def corrupt(uri):
        raise ValueError("corrupt artifact")

    monkeypatch.setattr(mlflow.sklearn, "load_model", corrupt)
    with pytest.raises(RuntimeError, match="could not load"):
        list(iter_mlflow_models(sweep["uri"], config.MLFLOW_EXPERIMENT))


def test_iter_mlflow_models_rejects_unknown_experiment(sweep):
    with pytest.raises(LookupError):
        next(iter_mlflow_models(sweep["uri"], "no_such_experiment"))


def test_profile_all_writes_results_raw_and_environment(sweep):
    out = sweep["root"] / "out"
    summary = profile_all(
        tracking_uri=sweep["uri"],
        experiment=config.MLFLOW_EXPERIMENT,
        data_dir=sweep["data_dir"],
        out_dir=out,
        protocol=FAST,
    )
    assert len(summary) == 2 * len(FAST.batch_sizes)

    on_disk = pd.read_csv(out / RESULTS_CSV)
    for col in ("technique", "classifier", "auc_pr_mean", "fit_seconds_mean",
                "model_size_mb", "batch_size", "mean_ms", "std_ms", "p50_ms",
                "p95_ms", "p99_ms", "throughput_per_s", "cpu_utilisation",
                "peak_predict_mb"):
        assert col in on_disk.columns

    raw = pd.read_csv(out / RAW_CSV)
    assert len(raw) == 2 * (FAST.n_trials + FAST.n_batch_trials)

    env = json.loads((out / ENVIRONMENT_JSON).read_text())
    assert env["protocol"]["batch_sizes"] == list(FAST.batch_sizes)
    assert env["input_pool"]["split"] == "validation"
    assert env["input_pool"]["dataset_sha256"] == "synthetic"
    assert env["packages"]["scikit-learn"]
