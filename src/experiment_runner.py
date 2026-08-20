"""
experiment_runner.py — The full technique x classifier sweep.

Maps to: Chapter 5 sec 5.1-5.3.
STATUS: STUB — implement Month 5-6.

Grid: 15 techniques x 4 classifiers x 5 folds = 300 fitted models.
Runtime is dominated by EasyEnsemble and RUSBoost. Budget several hours and
checkpoint after every (technique, classifier) pair so an interrupted run does
not lose everything.

Per-fold procedure:
    1. split training data into fold-train / fold-val (stratified)
    2. resample fold-train ONLY
    3. fit model on resampled fold-train
    4. predict probabilities on untouched fold-val
    5. choose threshold on fold-val, record evaluate() row
    6. measure fit time and per-record inference latency (Gap 1)

Log every run to MLflow with: technique, classifier, fold, seed, all metrics,
fit_seconds, inference_ms_per_record, git commit hash. The commit hash matters
— when a number in Chapter 5 is queried six months from now, you need to
recover the exact code that produced it.

Output: experiments/results_raw.csv, one row per (technique, classifier, fold).
Everything in Chapter 5 is a groupby over that single file.
"""

from __future__ import annotations

raise_msg = "Implement in Month 5-6 — see module docstring"


def run_single(technique: str, classifier: str, X, y, cv_splits) -> "list[dict]":
    raise NotImplementedError(raise_msg)


def run_sweep(output_csv: str = "experiments/results_raw.csv") -> None:
    raise NotImplementedError(raise_msg)
