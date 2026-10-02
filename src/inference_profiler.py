"""
inference_profiler.py — Latency characterisation.

Maps to: Chapter 5 sec 5.4. THIS IS CONTRIBUTION 2 (lit review Gap 1).

Only 2 of 21 reviewed sources report inference latency at all, and none measure
how the choice of imbalance-handling technique affects it. That is the gap.

The expected finding, worth stating as a hypothesis in Chapter 3: resampling
techniques change only the TRAINING data, so a SMOTE-trained logistic
regression has identical inference cost to an untreated one. Ensemble methods
(EasyEnsemble fits n_estimators independent models) do carry a real inference
penalty. If that holds, the practical conclusion is sharp: resampling is
effectively free at inference time, ensembles are not — so the accuracy gains
of ensembles must clear a higher bar to justify deployment.

The hypothesis is cleanest for logistic regression, whose size is fixed by the
feature count. A tree's size depends on the data it was grown on, so a forest
fitted to SMOTE-balanced data can be much larger than one fitted to
undersampled data. model_size_mb is recorded so that effect is visible rather
than assumed away.

What is measured, and why
-------------------------
Per (technique, classifier) model, at each batch size in
config.LATENCY_BATCH_SIZES:

  latency      wall-clock ms per predict_proba call: mean, std, min, p50, p95,
               p99, max. The median and tail percentiles are the headline,
               because latency is right-skewed: a mean is pulled up by rare
               stalls and hides the tail an SLA is written against. Mean and
               std are still reported because the two sources that measure
               latency at all report averages (Albalawi & Dardouri 2025;
               Darwish et al. 2025), and a like-for-like comparison needs the
               same statistic.
  throughput   records per second, and its reciprocal per_record_ms. Batch
               size 1 is the real-time case: a fraud API scores one
               transaction at a time. 100 / 1,000 / 10,000 show how far
               vectorisation amortises per-call overhead in bulk re-scoring,
               and give a figure comparable to Darwish et al.'s 495 tx/s.
  CPU          cpu_utilisation = process CPU time / wall time over the timed
               loop. Timing is single-threaded, so this should sit just under
               1.0. Above config.SINGLE_THREAD_CPU_TOLERANCE a thread pool
               escaped the pinning and a warning is logged; well below 1.0
               another process was competing for the core and the latencies
               are inflated. This makes "threads were pinned" a check rather
               than an assumption.
  memory       model_size_mb: pickled size of the fitted model, a proxy for
               on-disk and resident footprint.
               peak_predict_mb: peak Python-heap allocation (tracemalloc)
               during one predict_proba call. numpy buffers are tracked;
               allocations inside XGBoost's and LightGBM's C++ cores are not,
               so compare this within a library rather than across them.

AUC-PR and fit time are not re-measured here. They come from the sweep's
MLflow runs, averaged over folds, so Figure 8 plots the same AUC-PR as
Table 2.

Measurement protocol
--------------------
  - warm up with config.N_LATENCY_WARMUP discarded calls before timing (lazy
    allocation, first-call caches)
  - time.perf_counter() around predict_proba only; selecting the input rows
    happens before the clock starts
  - every model is timed on the same seeded sequence of validation records,
    so differences between techniques are paired rather than confounded by
    which rows each model saw. A fresh row per call stops one cached record
    flattering the result. The test split is never touched.
  - single-threaded (see single_threaded()). Threading speeds up large
    batches but adds dispatch overhead to single records, and how much it
    helps depends on the core count of the machine that ran it. Pinning is
    what makes rows comparable. It is OMP_NUM_THREADS=1 applied at runtime,
    extended to cover joblib and the boosting libraries' own thread pools.
  - garbage collector disabled during the timed loop (as timeit does), so a
    collection triggered by an earlier model's garbage is not charged to the
    current one
  - one model in memory at a time
  - hardware, OS, Python and library versions, default thread-pool state,
    dataset hash and git commit are written alongside the results

Not measured here: feature scaling, request parsing and network time. These
are identical for every technique. api/main.py measures the end-to-end
request for Chapter 4.

MLflow contract
---------------
experiment_runner.py logs to config.MLFLOW_EXPERIMENT at
config.MLFLOW_TRACKING_URI, one run per (technique, classifier, fold):
  params   technique, classifier, fold
  metrics  auc_pr, fit_seconds
  model    mlflow.sklearn.log_model(
               model, name=config.MLFLOW_MODEL_ARTIFACT,
               serialization_format=config.MLFLOW_SERIALIZATION_FORMAT)
Only one model per (technique, classifier) is profiled, so not every fold
needs to log one; see summarise_runs() for which is chosen.

Outputs (config.EXPERIMENTS_DIR)
--------------------------------
  latency_results.csv       one row per (technique, classifier, batch_size)
  latency_raw.csv           every timed call, so any percentile can be
                            recomputed, or plotted as a distribution, later
  latency_environment.json  machine, versions, protocol, dataset hash, commit

Deliverable: Figure 8, accuracy (AUC-PR) vs p99 latency scatter, with the
100 ms budget drawn as a vertical line. Techniques in the upper-left quadrant
are the deployable ones. Plot it from latency_results.csv where
batch_size == 1 (auc_pr_mean against p99_ms).

Running it
----------
    python -m src.inference_profiler        # or: make profile

Close other applications, and keep a laptop on mains power: CPU frequency
scaling and background load move the tail percentiles more than most
differences between techniques do.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import pickle
import platform
import subprocess
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_info, threadpool_limits

from . import config
from .data_loader import FraudDataset

logger = logging.getLogger(__name__)

RESULTS_CSV = "latency_results.csv"
RAW_CSV = "latency_raw.csv"
ENVIRONMENT_JSON = "latency_environment.json"

# Packages whose version can move a latency number; recorded with the results.
VERSIONED_PACKAGES = (
    "numpy", "scipy", "pandas", "scikit-learn", "imbalanced-learn",
    "xgboost", "lightgbm", "mlflow", "threadpoolctl",
)

RUN_SUMMARY_COLUMNS = [
    "technique", "classifier", "auc_pr_mean", "fit_seconds_mean", "n_folds",
    "run_ids",
]


# ---------------------------------------------------------------------------
# Protocol and result containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TimingProtocol:
    """How many calls to time, and to discard, at each batch size.

    Held in one frozen object so the exact protocol is written into
    latency_environment.json next to the numbers it produced. Defaults come
    from config; tests pass a much smaller protocol.
    """

    batch_sizes: tuple[int, ...] = config.LATENCY_BATCH_SIZES
    n_trials: int = config.N_LATENCY_TRIALS
    n_warmup: int = config.N_LATENCY_WARMUP
    n_batch_trials: int = config.N_BATCH_TRIALS
    n_batch_warmup: int = config.N_BATCH_WARMUP
    random_state: int = config.RANDOM_STATE

    def calls_for(self, batch_size: int) -> tuple[int, int]:
        """(timed, warm-up) call counts for one batch size.

        Single records get the full 1,000 trials a p99 needs. A 10,000-row
        call to a large forest takes around a tenth of a second, so 1,000
        trials at every batch size across every model would add hours without
        changing a conclusion. Batches get fewer trials, which means their p99
        is close to their max and should be read as such.
        """
        if batch_size == 1:
            return self.n_trials, self.n_warmup
        return self.n_batch_trials, self.n_batch_warmup


@dataclass
class LatencyResult:
    """Raw measurements for one model at one batch size."""

    batch_size: int
    timings_ms: np.ndarray  # one entry per timed call
    cpu_utilisation: float
    peak_predict_mb: float

    def summary(self) -> dict:
        t = self.timings_ms
        row = {
            "batch_size": self.batch_size,
            "n_trials": int(t.size),
            "mean_ms": float(t.mean()),
            "std_ms": float(t.std(ddof=1)) if t.size > 1 else 0.0,
            "min_ms": float(t.min()),
        }
        for q in config.LATENCY_PERCENTILES:
            row[f"p{q}_ms"] = float(np.percentile(t, q))
        row.update(
            {
                "max_ms": float(t.max()),
                # per_record_ms == 1000 / throughput_per_s. Both are kept:
                # Chapter 5 quotes ms per transaction, Darwish et al. (2025)
                # quote transactions per second.
                "per_record_ms": float(t.mean() / self.batch_size),
                "throughput_per_s": float(1e3 * self.batch_size / t.mean()),
                "cpu_utilisation": self.cpu_utilisation,
                "peak_predict_mb": self.peak_predict_mb,
            }
        )
        return row


@dataclass
class ProfiledModel:
    """A fitted model plus the sweep metadata Figure 8 plots beside it."""

    technique: str
    classifier: str
    model: Any
    run_id: str | None = None
    auc_pr_mean: float = float("nan")
    fit_seconds_mean: float = float("nan")
    n_folds: int = 0


# ---------------------------------------------------------------------------
# Measuring one model
# ---------------------------------------------------------------------------
@contextmanager
def single_threaded(model: Any) -> Iterator[None]:
    """Run the enclosed block with every thread pool the model uses set to 1.

    No single mechanism covers every library, so two are combined:
      - n_jobs=1 on the estimator and on every nested estimator. This governs
        joblib parallelism in RandomForest, BalancedRandomForest and
        EasyEnsemble, and XGBoost and LightGBM read it as their native thread
        count at predict time.
      - threadpoolctl limits BLAS (LogisticRegression's matrix product) and
        any OpenMP runtime to one thread.
    Limiting a pool does not stop threads it already started. OpenBLAS
    workers spin-wait for a while after each multi-threaded call, so without
    the settle pause, CPU time from earlier work (a fit, say) is charged to
    the timed loop. That was observed during development: a logistic
    regression profiled straight after fitting showed CPU/wall = 3.6 on four
    cores, and 1.0 after the pause.

    The original n_jobs values are restored on exit, leaving the caller's
    model as it was found.
    """
    params = model.get_params(deep=True) if hasattr(model, "get_params") else {}
    original = {k: v for k, v in params.items() if k.split("__")[-1] == "n_jobs"}
    try:
        if original:
            model.set_params(**{k: 1 for k in original})
        with threadpool_limits(limits=1):
            time.sleep(config.THREAD_SETTLE_SECONDS)
            yield
    finally:
        if original:
            model.set_params(**original)


def time_predictions(
    model: Any,
    X_pool: np.ndarray,
    batch_size: int,
    n_trials: int,
    n_warmup: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Time n_trials calls to model.predict_proba, each on fresh rows.

    Returns (wall-clock ms per call, CPU utilisation over the timed loop).
    Rows are drawn with replacement, so any batch size works with any pool.
    CPU time is taken over the whole loop rather than per call, because
    per-call process_time() is too coarse on some platforms (about 16 ms on
    Windows) to resolve a sub-millisecond prediction.
    """
    predict = model.predict_proba
    n = len(X_pool)
    for idx in rng.integers(0, n, size=(n_warmup, batch_size)):
        predict(X_pool[idx])

    draws = rng.integers(0, n, size=(n_trials, batch_size))
    timings = np.empty(n_trials)
    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        cpu_start, wall_start = time.process_time(), time.perf_counter()
        for i, idx in enumerate(draws):
            X = X_pool[idx]
            t0 = time.perf_counter()
            predict(X)
            timings[i] = time.perf_counter() - t0
        cpu_seconds = time.process_time() - cpu_start
        wall_seconds = time.perf_counter() - wall_start
    finally:
        if gc_was_enabled:
            gc.enable()
    return timings * 1e3, cpu_seconds / wall_seconds


def peak_predict_mb(model: Any, X: np.ndarray) -> float:
    """Peak Python-heap allocation during one predict_proba call, in MB.

    Run outside the timed loop because tracing slows every allocation.
    """
    tracemalloc.start()
    try:
        model.predict_proba(X)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 2**20


def serialised_size_mb(model: Any) -> float:
    """Pickled size of the fitted model, in MB.

    Measured on the in-memory object rather than read from the artifact store,
    so it does not depend on how MLflow happened to save or compress it.
    """
    return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) / 2**20


def profile_model(
    model: Any, X_pool: np.ndarray, protocol: TimingProtocol | None = None
) -> list[LatencyResult]:
    """Time one fitted model at every batch size in the protocol.

    Batch sizes run in ascending order, each with its own warm-up, since a
    10,000-row call exercises different allocation paths from a 1-row call.
    The random number generator is re-seeded for every model, so every model
    is timed on the same rows.
    """
    protocol = protocol or TimingProtocol()
    rng = np.random.default_rng(protocol.random_state)
    results = []
    with single_threaded(model):
        for batch_size in sorted(protocol.batch_sizes):
            n_trials, n_warmup = protocol.calls_for(batch_size)
            timings_ms, cpu_util = time_predictions(
                model, X_pool, batch_size, n_trials, n_warmup, rng
            )
            if cpu_util > config.SINGLE_THREAD_CPU_TOLERANCE:
                logger.warning(
                    "%s at batch %d: CPU/wall = %.2f, so a thread pool escaped "
                    "single-thread pinning. This row is not comparable.",
                    type(model).__name__, batch_size, cpu_util,
                )
            sample = X_pool[rng.integers(0, len(X_pool), size=batch_size)]
            results.append(
                LatencyResult(
                    batch_size=batch_size,
                    timings_ms=timings_ms,
                    cpu_utilisation=cpu_util,
                    peak_predict_mb=peak_predict_mb(model, sample),
                )
            )
    return results


def profile_models(
    models: Iterable[ProfiledModel],
    X_pool: np.ndarray,
    protocol: TimingProtocol | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profile each model in turn. Returns (summary table, raw timings).

    The summary has one row per (technique, classifier, batch_size); the raw
    table has one row per timed call.
    """
    summary_rows: list[dict] = []
    raw_frames: list[pd.DataFrame] = []
    for pm in models:
        logger.info("Profiling %s / %s", pm.technique, pm.classifier)
        meta = {
            "technique": pm.technique,
            "classifier": pm.classifier,
            "run_id": pm.run_id,
            "auc_pr_mean": pm.auc_pr_mean,
            "fit_seconds_mean": pm.fit_seconds_mean,
            "n_folds": pm.n_folds,
            "model_size_mb": serialised_size_mb(pm.model),
        }
        for result in profile_model(pm.model, X_pool, protocol):
            summary_rows.append({**meta, **result.summary()})
            raw_frames.append(
                pd.DataFrame(
                    {
                        "technique": pm.technique,
                        "classifier": pm.classifier,
                        "batch_size": result.batch_size,
                        "trial": np.arange(result.timings_ms.size),
                        "latency_ms": result.timings_ms,
                    }
                )
            )
        del pm  # release before the next model is loaded

    if not summary_rows:
        return pd.DataFrame(), pd.DataFrame()
    return pd.DataFrame(summary_rows), pd.concat(raw_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Loading models from MLflow
# ---------------------------------------------------------------------------
def summarise_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Reduce the sweep's run table to one row per (technique, classifier).

    `runs` is the DataFrame mlflow.search_runs() returns. The sweep logs one
    run per fold. Latency depends on the fitted model's structure, which
    barely varies between folds, so one model per pair is profiled, not five.
    AUC-PR and fit time are averaged over all folds so they match Table 2.

    run_ids lists the pair's runs in the order a model is looked for: lowest
    fold first. Unfinished runs are dropped, and a re-run of a fold replaces
    the earlier run.
    """
    if runs.empty:
        return pd.DataFrame(columns=RUN_SUMMARY_COLUMNS)

    runs = runs.copy()
    for col in ("params.technique", "params.classifier", "params.fold",
                "metrics.auc_pr", "metrics.fit_seconds"):
        if col not in runs:
            runs[col] = np.nan
    runs = runs[
        (runs["status"] == "FINISHED")
        & runs["params.technique"].notna()
        & runs["params.classifier"].notna()
    ]
    runs = runs.sort_values("start_time", ascending=False).drop_duplicates(
        ["params.technique", "params.classifier", "params.fold"]
    )
    runs["fold_num"] = pd.to_numeric(runs["params.fold"], errors="coerce")
    runs = runs.sort_values("fold_num", kind="stable")

    rows = [
        {
            "technique": technique,
            "classifier": classifier,
            "auc_pr_mean": g["metrics.auc_pr"].mean(),
            "fit_seconds_mean": g["metrics.fit_seconds"].mean(),
            "n_folds": len(g),
            "run_ids": g["run_id"].tolist(),
        }
        for (technique, classifier), g in runs.groupby(
            ["params.technique", "params.classifier"]
        )
    ]
    return pd.DataFrame(rows, columns=RUN_SUMMARY_COLUMNS)


def iter_mlflow_models(
    tracking_uri: str = config.MLFLOW_TRACKING_URI,
    experiment: str = config.MLFLOW_EXPERIMENT,
) -> Iterator[ProfiledModel]:
    """Yield one fitted model per (technique, classifier) from the sweep.

    A generator, so only one model is in memory at a time. A forest fitted on
    SMOTE-balanced data can run to hundreds of MB; holding every model at
    once would exhaust a laptop's RAM and perturb timings through memory
    pressure.

    Models are loaded with the native sklearn flavour, not mlflow.pyfunc. The
    pyfunc wrapper adds DataFrame conversion and schema checks whose cost has
    nothing to do with the imbalance technique, and the API calls
    predict_proba on the native object anyway.
    """
    import mlflow
    import mlflow.sklearn
    from mlflow.exceptions import MlflowException
    from mlflow.models import get_model_info

    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        raise LookupError(
            f"No MLflow experiment {experiment!r} at {tracking_uri}. The "
            f"profiler measures models the sweep has logged, so run "
            f"src/experiment_runner.py first."
        )
    table = summarise_runs(mlflow.search_runs(experiment_ids=[exp.experiment_id]))

    missing = sorted(set(config.ALL_TECHNIQUES) - set(table["technique"]))
    if missing:
        logger.warning(
            "No finished runs for %d configured technique(s): %s",
            len(missing), ", ".join(missing),
        )
    logger.info("Found %d (technique, classifier) pairs", len(table))

    for pair in table.itertuples(index=False):
        for run_id in pair.run_ids:
            uri = f"runs:/{run_id}/{config.MLFLOW_MODEL_ARTIFACT}"
            # Look for the model's metadata before loading it. A fold that
            # logged metrics only is expected and skipped. A model that exists
            # but will not load is a real fault and must stop the run, not
            # quietly drop a technique from the comparison.
            try:
                get_model_info(uri)
            except (MlflowException, OSError):
                continue
            try:
                model = mlflow.sklearn.load_model(uri)
            except Exception as err:
                raise RuntimeError(
                    f"{pair.technique} / {pair.classifier}: run {run_id} has "
                    f"a model that the sklearn flavour could not load"
                ) from err
            yield ProfiledModel(
                technique=pair.technique,
                classifier=pair.classifier,
                model=model,
                run_id=run_id,
                auc_pr_mean=pair.auc_pr_mean,
                fit_seconds_mean=pair.fit_seconds_mean,
                n_folds=pair.n_folds,
            )
            del model
            break
        else:
            logger.warning(
                "%s / %s: none of its %d runs has a model logged under %r; "
                "skipped",
                pair.technique, pair.classifier, len(pair.run_ids),
                config.MLFLOW_MODEL_ARTIFACT,
            )


# ---------------------------------------------------------------------------
# Environment record
# ---------------------------------------------------------------------------
def _git(args: list[str]) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=config.ROOT, capture_output=True, text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def git_state() -> dict:
    """Commit the results came from, and whether the working tree was clean.

    A dirty tree means the numbers came from code that is not in any commit,
    which breaks "every reported number comes from a committed script".
    """
    status = _git(["status", "--porcelain"])
    return {
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": None if status is None else bool(status),
    }


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _cpu_model() -> str:
    """platform.processor() is empty on most Linux builds, so ask the OS."""
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=True,
            )
            return out.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def environment_info() -> dict:
    """The machine and software that produced the numbers, for Chapter 5.

    Thread pools are recorded as found, outside single_threaded(). That
    answers "was BLAS threading active?"; the timing itself ran pinned to one
    thread, which each row's cpu_utilisation confirms.
    """
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **git_state(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "cores_logical": psutil.cpu_count(logical=True),
        "cores_physical": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "packages": {name: _package_version(name) for name in VERSIONED_PACKAGES},
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
        "thread_pools": [
            {k: pool.get(k) for k in ("user_api", "internal_api", "version",
                                      "num_threads")}
            for pool in threadpool_info()
        ],
        "timer": "time.perf_counter",
        "timer_resolution_s": time.get_clock_info("perf_counter").resolution,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def profile_all(
    tracking_uri: str = config.MLFLOW_TRACKING_URI,
    experiment: str = config.MLFLOW_EXPERIMENT,
    data_dir: str | Path = config.DATA_PROCESSED,
    out_dir: str | Path = config.EXPERIMENTS_DIR,
    protocol: TimingProtocol | None = None,
) -> pd.DataFrame:
    """Profile every model the sweep logged and write the three output files.

    Input rows come from the VALIDATION split. As everywhere else in this
    project, the test split is reserved for the final evaluation.
    Returns the summary table.
    """
    protocol = protocol or TimingProtocol()
    splits, manifest = FraudDataset.load_splits(data_dir)
    X_pool = splits.X_val

    summary, raw = profile_models(
        iter_mlflow_models(tracking_uri, experiment), X_pool, protocol
    )
    if summary.empty:
        raise RuntimeError(
            f"No models could be loaded from experiment {experiment!r} at "
            f"{tracking_uri}. Check that the sweep logs models as described "
            f"under 'MLflow contract' in this module's docstring."
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / RESULTS_CSV, index=False)
    raw.to_csv(out / RAW_CSV, index=False)

    env = environment_info()
    env.update(
        {
            "protocol": asdict(protocol),
            "threads_during_timing": 1,
            "gc_disabled_during_timing": True,
            "input_pool": {
                "split": "validation",
                "n_rows": int(X_pool.shape[0]),
                "n_features": int(X_pool.shape[1]),
                "dataset_sha256": manifest.get("validation_report", {}).get(
                    "file_sha256"
                ),
            },
            "mlflow": {"tracking_uri": tracking_uri, "experiment": experiment},
        }
    )
    (out / ENVIRONMENT_JSON).write_text(json.dumps(env, indent=2))
    logger.info(
        "Wrote %s, %s and %s to %s", RESULTS_CSV, RAW_CSV, ENVIRONMENT_JSON, out
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tracking-uri", default=config.MLFLOW_TRACKING_URI)
    parser.add_argument("--experiment", default=config.MLFLOW_EXPERIMENT)
    parser.add_argument("--data", default=str(config.DATA_PROCESSED))
    parser.add_argument("--out", default=str(config.EXPERIMENTS_DIR))
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+",
        default=list(config.LATENCY_BATCH_SIZES),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
    )
    # Per-model download progress bars would bury the log.
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

    summary = profile_all(
        tracking_uri=args.tracking_uri,
        experiment=args.experiment,
        data_dir=args.data,
        out_dir=args.out,
        protocol=TimingProtocol(batch_sizes=tuple(args.batch_sizes)),
    )

    over_budget = summary[
        (summary["batch_size"] == 1)
        & (summary["p99_ms"] > config.LATENCY_BUDGET_MS)
    ]
    for row in over_budget.itertuples(index=False):
        logger.warning(
            "%s / %s: single-record p99 %.1f ms exceeds the %.0f ms budget",
            row.technique, row.classifier, row.p99_ms, config.LATENCY_BUDGET_MS,
        )

    headline = summary[summary["batch_size"] == summary["batch_size"].min()]
    columns = [
        "technique", "classifier", "batch_size", "auc_pr_mean", "p50_ms",
        "p95_ms", "p99_ms", "model_size_mb", "cpu_utilisation",
    ]
    print("\n" + headline[columns].sort_values("p99_ms").to_string(index=False))


if __name__ == "__main__":
    main()
