"""
experiment_runner.py — The full technique x classifier sweep.

Maps to: Chapter 5 sec 5.1-5.3.

Grid: the 41 (technique, classifier) pairs from
imbalance_methods.sweep_pairs() x 5 folds = 205 fitted models. Runtime is
dominated by the SMOTE family on random forests (several hundred thousand
synthetic rows per fold) and by SMOTE-ENN's nearest-neighbour cleaning.
Budget a few hours. Results are checkpointed after every (technique,
classifier) pair, so an interrupted run resumes where it stopped instead of
losing everything.

Per-fold procedure:
    1. split the TRAINING split into fold-train / fold-val (stratified,
       FraudDataset.cv_splits, seed 42); val and test splits are not used
    2. resample fold-train ONLY
    3. fit model on resampled fold-train
    4. predict probabilities on untouched fold-val
    5. choose threshold on fold-val, record evaluate() row
    6. measure fit time and per-record inference latency (Gap 1)

Step 5 picks the threshold on the same fold it scores, so the CV
threshold-dependent metrics (F1, MCC, precision, recall) are optimistic.
AUC-PR, the primary metric, is threshold-free and unaffected. Final
threshold-dependent figures come from applying a validation-chosen threshold
to the test split (rule 3), not from this table.

Every run is logged to MLflow with: technique, classifier, fold, seed, the
resampled training-set size, all metrics, fit_seconds,
inference_ms_per_record, and git commit. The commit hash matters — when a
number in Chapter 5 is queried six months from now, you need to recover the
exact code that produced it.

The fold-0 model of every pair is also logged, in the format
inference_profiler.py loads (see 'MLflow contract' in its docstring). Only
one model per pair is needed for latency, and logging all five would multiply
disk use: a forest grown on SMOTE data can be hundreds of MB.

Output: experiments/results_raw.csv, one row per (technique, classifier, fold).
Everything in Chapter 5 is a groupby over that single file.

Run:
    python -m src.experiment_runner                 # or: make sweep
    python -m src.experiment_runner --techniques none smote   # a subset
    python -m src.experiment_runner --techniques smote_enn --rerun
                                     # discard and recompute a subset
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import config
from .data_loader import FraudDataset
from .evaluation import CostModel, choose_threshold, evaluate
from .imbalance_methods import get_model, resample_fold, sweep_pairs
from .inference_profiler import git_state, single_threaded, time_predictions

logger = logging.getLogger(__name__)

# Logged as MLflow params rather than metrics.
PARAM_KEYS = (
    "technique", "classifier", "fold", "seed", "n_train", "n_train_fraud",
)


def _quick_latency_ms(model: Any, X_pool: np.ndarray) -> float:
    """Median single-record predict_proba latency, a sanity figure per run.

    Same method as inference_profiler.py with far fewer trials; Chapter 5
    sec 5.4 quotes the profiler, not this.
    """
    rng = np.random.default_rng(config.RANDOM_STATE)
    with single_threaded(model):
        timings_ms, _ = time_predictions(
            model, X_pool, 1, config.SWEEP_LATENCY_TRIALS,
            config.SWEEP_LATENCY_WARMUP, rng,
        )
    return float(np.median(timings_ms))


def run_single(
    technique: str,
    classifier: str,
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    log_to_mlflow: bool = False,
) -> list[dict]:
    """Run one (technique, classifier) pair over every fold.

    Returns one evaluate() row per fold, extended with the run metadata.
    With log_to_mlflow, each fold is also logged as an MLflow run in the
    active experiment, and the fold-0 model is logged for profiling.
    """
    cost_model = CostModel(cost_fn=config.COST_FN, cost_fp=config.COST_FP)
    rows = []
    for fold, (tr, va) in enumerate(cv_splits):
        X_fit, y_fit = resample_fold(X[tr], y[tr], technique)
        model = get_model(classifier, technique, y_train=y_fit)

        start = time.perf_counter()
        model.fit(X_fit, y_fit)
        fit_seconds = time.perf_counter() - start

        X_va, y_va = X[va], y[va]   # never resampled
        p_val = model.predict_proba(X_va)[:, 1]
        # A constant score has no PR curve to search (see baseline_models).
        threshold = (
            choose_threshold(y_va, p_val, objective="f1")
            if np.unique(p_val).size > 1 else 0.5
        )
        row = evaluate(
            y_va, p_val, threshold=threshold, cost_model=cost_model,
            alert_budget=config.ALERT_BUDGET,
        )
        row.update(
            {
                "technique": technique,
                "classifier": classifier,
                "fold": fold,
                "seed": config.RANDOM_STATE,
                # Training-set size after resampling: undersampling leaves a
                # few hundred rows, SMOTE several hundred thousand. Explains
                # fit time and, for trees, model size.
                "n_train": int(len(y_fit)),
                "n_train_fraud": int(y_fit.sum()),
                "fit_seconds": fit_seconds,
                "inference_ms_per_record": _quick_latency_ms(model, X_va),
            }
        )
        logger.info(
            "  fold %d: AUC-PR %.4f, fit %.1fs, n_train %d",
            fold, row["auc_pr"], fit_seconds, row["n_train"],
        )
        if log_to_mlflow:
            _log_run(row, model if fold == 0 else None)
        rows.append(row)
    return rows


def _log_run(row: dict, model: Any | None) -> None:
    """Log one fold as an MLflow run; log the model too if one is given."""
    import mlflow
    import mlflow.sklearn

    name = f"{row['technique']}/{row['classifier']}/fold{row['fold']}"
    with mlflow.start_run(run_name=name):
        mlflow.log_params({k: row[k] for k in PARAM_KEYS})
        mlflow.log_metrics(
            {
                k: float(v) for k, v in row.items()
                if k not in PARAM_KEYS and isinstance(v, (int, float, np.number))
            }
        )
        mlflow.set_tags({k: str(v) for k, v in git_state().items()})
        if model is not None:
            # MLflow 3 renamed artifact_path to name.
            name_kw = (
                "name" if int(mlflow.__version__.split(".")[0]) >= 3
                else "artifact_path"
            )
            mlflow.sklearn.log_model(
                model,
                serialization_format=config.MLFLOW_SERIALIZATION_FORMAT,
                **{name_kw: config.MLFLOW_MODEL_ARTIFACT},
            )


def _completed_pairs(output_csv: Path, n_folds: int) -> set[tuple[str, str]]:
    """Pairs already in the checkpoint file with every fold present."""
    if not output_csv.exists():
        return set()
    done = pd.read_csv(output_csv, usecols=["technique", "classifier", "fold"])
    counts = done.groupby(["technique", "classifier"])["fold"].nunique()
    return set(counts[counts == n_folds].index)


def _set_experiment(tracking_uri: str, experiment: str) -> None:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(
            experiment, artifact_location=config.MLFLOW_ARTIFACT_LOCATION
        )
    mlflow.set_experiment(experiment)


def run_sweep(
    output_csv: str | Path = config.RESULTS_RAW_CSV,
    data_dir: str | Path = config.DATA_PROCESSED,
    tracking_uri: str = config.MLFLOW_TRACKING_URI,
    experiment: str = config.MLFLOW_EXPERIMENT,
    pairs: Sequence[tuple[str, str]] | None = None,
    rerun: bool = False,
) -> pd.DataFrame:
    """Run every pair, checkpointing to output_csv; return the full table.

    Pairs already complete in output_csv are skipped, so re-running after an
    interruption resumes. A pair interrupted mid-way is re-run from fold 0;
    its earlier MLflow runs are superseded, because the profiler keeps the
    most recent run per fold.

    rerun=True first discards the checkpointed rows for `pairs`, so they
    are recomputed: for use after a fix to a technique. Their new MLflow
    runs supersede the old ones in the same way.
    """
    state = git_state()
    if state["git_dirty"]:
        logger.warning(
            "Working tree has uncommitted changes: these results will not be "
            "traceable to commit %s. Commit before a run you will report.",
            state["git_commit"],
        )

    splits, _ = FraudDataset.load_splits(data_dir)
    X, y = splits.X_train, splits.y_train
    cv_splits = list(FraudDataset().cv_splits(X, y))

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs = list(pairs or sweep_pairs())
    if rerun and out.exists():
        kept = pd.read_csv(out)
        stale = pd.Series(list(zip(kept.technique, kept.classifier))).isin(pairs)
        logger.info("Discarding %d checkpointed rows to rerun", int(stale.sum()))
        kept[~stale.to_numpy()].to_csv(out, index=False)
    done = _completed_pairs(out, len(cv_splits))
    _set_experiment(tracking_uri, experiment)

    for i, (technique, classifier) in enumerate(pairs, 1):
        if (technique, classifier) in done:
            logger.info("[%d/%d] %s / %s: done, skipping",
                        i, len(pairs), technique, classifier)
            continue
        logger.info("[%d/%d] %s / %s", i, len(pairs), technique, classifier)
        rows = pd.DataFrame(
            run_single(technique, classifier, X, y, cv_splits, log_to_mlflow=True)
        )
        if out.exists():
            # Keep the existing column order so the file stays rectangular.
            rows = rows.reindex(columns=pd.read_csv(out, nrows=0).columns)
        rows.to_csv(out, mode="a", header=not out.exists(), index=False)

    return pd.read_csv(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default=str(config.DATA_PROCESSED))
    parser.add_argument("--out", default=str(config.RESULTS_RAW_CSV))
    parser.add_argument("--tracking-uri", default=config.MLFLOW_TRACKING_URI)
    parser.add_argument("--experiment", default=config.MLFLOW_EXPERIMENT)
    parser.add_argument("--techniques", nargs="+", help="run only these")
    parser.add_argument("--classifiers", nargs="+", help="run only these")
    parser.add_argument(
        "--rerun", action="store_true",
        help="recompute the selected pairs even if already checkpointed",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
    )

    pairs = [
        (t, c) for t, c in sweep_pairs()
        if (not args.techniques or t in args.techniques)
        and (not args.classifiers or c in args.classifiers)
    ]
    if not pairs:
        parser.error("no (technique, classifier) pairs match the filters")

    results = run_sweep(
        output_csv=args.out, data_dir=args.data,
        tracking_uri=args.tracking_uri, experiment=args.experiment, pairs=pairs,
        rerun=args.rerun,
    )
    table = (
        results.groupby(["technique", "classifier"])[["auc_pr", "fit_seconds"]]
        .mean()
        .sort_values("auc_pr", ascending=False)
    )
    print("\n" + table.to_string())


if __name__ == "__main__":
    main()
