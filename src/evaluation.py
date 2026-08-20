"""
evaluation.py — Metrics, threshold selection and statistical testing.

Maps to: Chapter 3 §3.4 (Evaluation Metrics), Chapter 5 (all result tables).

Why AUC-PR is the primary metric (lit review §5, Gap 2)
-------------------------------------------------------
At an imbalance ratio of 1:578, ROC-AUC is misleading: the false-positive rate
denominator is dominated by ~284k negatives, so a large absolute number of false
alarms barely moves the curve. Precision-Recall focuses on the positive class and
its baseline equals the positive rate (0.0017), making improvements interpretable.
Accuracy is never reported as a headline figure — a classifier predicting
"legitimate" for everything scores 99.83%.

Only 11.4% of the reviewed corpus reports AUC-PR despite broad agreement that it
is the appropriate metric. Reporting it here addresses Gap 2 directly.

Threshold policy
----------------
A default threshold of 0.5 is arbitrary under extreme imbalance. Every
threshold-dependent metric in this project is reported at a threshold selected on
the VALIDATION split and then applied unchanged to the test split. Selecting the
threshold on test data is a leakage error that inflates F1 substantially.

Usage:
    from evaluation import evaluate, choose_threshold, bootstrap_ci

    thr = choose_threshold(y_val, p_val, objective="f1")
    row = evaluate(y_test, p_test, threshold=thr)
    lo, hi = bootstrap_ci(y_test, p_test, metric="auc_pr")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

PRIMARY_METRIC = "auc_pr"

# Asymmetric misclassification costs (Chapter 3 §3.4, Chapter 6 §6.3).
# A missed fraud costs the chargeback; a false alarm costs review labour and
# customer friction. These are indicative defaults — state the assumption
# explicitly in the dissertation and run a sensitivity analysis.
DEFAULT_COST_FN = 100.0   # missed fraud
DEFAULT_COST_FP = 5.0     # false alarm


@dataclass
class CostModel:
    """Business cost of errors. Used for cost-sensitive threshold selection."""

    cost_fn: float = DEFAULT_COST_FN
    cost_fp: float = DEFAULT_COST_FP

    def total(self, fp: int, fn: int) -> float:
        return fp * self.cost_fp + fn * self.cost_fn


# ---------------------------------------------------------------------------
# Threshold-free metrics
# ---------------------------------------------------------------------------
def auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the Precision-Recall curve (average precision).

    PRIMARY METRIC. Baseline for a random classifier equals the positive rate.
    """
    return float(average_precision_score(y_true, y_score))


def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Reported for comparability with prior work only — not the headline."""
    return float(roc_auc_score(y_true, y_score))


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Precision within the k highest-scored transactions.

    This is the metric a fraud operations team actually cares about: if analysts
    can review k alerts per day, what fraction are real? Absent from most of the
    reviewed corpus.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, len(y_score))
    top = np.argsort(y_score)[::-1][:k]
    return float(np.mean(y_true[top]))


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Fraction of all fraud captured within a k-alert review budget."""
    k = min(max(k, 1), len(y_score))
    top = np.argsort(y_score)[::-1][:k]
    n_pos = int(np.sum(y_true))
    return float(np.sum(y_true[top]) / n_pos) if n_pos else 0.0


# ---------------------------------------------------------------------------
# Threshold-dependent metrics
# ---------------------------------------------------------------------------
def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    cost_model: CostModel | None = None,
    alert_budget: int | None = None,
) -> dict:
    """Compute the full metric row for one model at one threshold.

    Returns a flat dict, so a list of these becomes a Chapter 5 results table.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cm = cost_model or CostModel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0

    row = {
        "auc_pr": auc_pr(y_true, y_score),
        "auc_roc": auc_roc(y_true, y_score),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        # MCC is robust under imbalance because it uses all four cells of the
        # confusion matrix — unlike F1, which ignores true negatives.
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "g_mean": float(np.sqrt(sensitivity * specificity)),
        "specificity": float(specificity),
        # Calibration matters if the score feeds a cost-based decision rule.
        "brier": float(brier_score_loss(y_true, y_score)),
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_alerts": int(tp + fp),
        "total_cost": cm.total(int(fp), int(fn)),
    }

    if alert_budget:
        row[f"precision_at_{alert_budget}"] = precision_at_k(
            y_true, y_score, alert_budget
        )
        row[f"recall_at_{alert_budget}"] = recall_at_k(
            y_true, y_score, alert_budget
        )
    return row


def choose_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    objective: str = "f1",
    cost_model: CostModel | None = None,
    min_precision: float | None = None,
) -> float:
    """Select an operating threshold on the VALIDATION split.

    objective:
      "f1"        — maximise F1 (balanced default)
      "cost"      — minimise expected business cost (recommended for fraud)
      "mcc"       — maximise MCC
      "precision" — maximise recall subject to precision >= min_precision,
                    i.e. "catch as much fraud as possible without drowning
                    the review team in false alarms"

    Never call this on the test split.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns len(thr) == len(prec) - 1
    prec, rec = prec[:-1], rec[:-1]
    if len(thr) == 0:
        return 0.5

    if objective == "f1":
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = np.nan_to_num(2 * prec * rec / (prec + rec))
        return float(thr[int(np.argmax(scores))])

    if objective == "precision":
        if min_precision is None:
            raise ValueError("objective='precision' requires min_precision")
        ok = prec >= min_precision
        if not ok.any():
            logger.warning(
                "No threshold reaches precision >= %.3f; falling back to F1",
                min_precision,
            )
            return choose_threshold(y_true, y_score, "f1")
        return float(thr[int(np.argmax(np.where(ok, rec, -np.inf)))])

    if objective in {"cost", "mcc"}:
        # Evaluate on a grid of candidate thresholds; the PR-curve thresholds
        # are the only points where predictions change, so this is exact.
        cands = thr if len(thr) <= 2000 else thr[:: max(1, len(thr) // 2000)]
        cm = cost_model or CostModel()
        best, best_val = 0.5, np.inf if objective == "cost" else -np.inf
        for t in cands:
            pred = (y_score >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(
                y_true, pred, labels=[0, 1]
            ).ravel()
            if objective == "cost":
                val = cm.total(int(fp), int(fn))
                if val < best_val:
                    best, best_val = float(t), val
            else:
                val = matthews_corrcoef(y_true, pred)
                if val > best_val:
                    best, best_val = float(t), val
        return best

    raise ValueError(f"Unknown objective {objective!r}")


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
_METRIC_FNS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "auc_pr": auc_pr,
    "auc_roc": auc_roc,
}


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "auc_pr",
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
    stratified: bool = True,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI. Returns (point_estimate, lower, upper).

    Stratified resampling keeps the number of positives fixed across replicates.
    With only ~71 fraud cases in the test split, unstratified resampling
    produces replicates with wildly varying positive counts and an
    uninterpretably wide interval.

    Every headline figure in Chapter 5 should carry one of these intervals —
    differences between techniques are often smaller than the interval width,
    which is itself a finding (lit review §6: no consistent winner).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    fn = _METRIC_FNS.get(metric)
    if fn is None:
        raise ValueError(f"metric must be one of {sorted(_METRIC_FNS)}")

    rng = np.random.default_rng(random_state)
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    n = len(y_true)

    vals = []
    for _ in range(n_boot):
        if stratified:
            idx = np.concatenate([
                rng.choice(pos, size=len(pos), replace=True),
                rng.choice(neg, size=len(neg), replace=True),
            ])
        else:
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y_true[idx])) < 2:
                continue
        vals.append(fn(y_true[idx], y_score[idx]))

    vals = np.asarray(vals)
    return (
        float(fn(y_true, y_score)),
        float(np.percentile(vals, 100 * alpha / 2)),
        float(np.percentile(vals, 100 * (1 - alpha / 2))),
    )


# ---------------------------------------------------------------------------
# Statistical comparison across techniques
# ---------------------------------------------------------------------------
# Studentised range statistic q_alpha at p=0.05, indexed by number of techniques.
# Source: Demsar (2006), Table 5 — the standard reference for comparing
# classifiers over multiple datasets/folds.
_Q_ALPHA_05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
    9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
    15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544,
}


def friedman_test(scores: pd.DataFrame) -> dict:
    """Friedman test across techniques.

    scores: DataFrame with one row per fold and one column per technique.
    Returns the statistic, p-value and mean ranks (rank 1 = best).

    The Friedman test is non-parametric and does not assume normality, which
    matters because AUC-PR across folds is bounded and typically skewed.
    """
    if scores.shape[1] < 3:
        raise ValueError("Friedman requires at least 3 techniques")
    stat, p = stats.friedmanchisquare(*[scores[c].values for c in scores.columns])
    # Higher score is better, so rank the negated values.
    ranks = scores.rank(axis=1, ascending=False).mean().sort_values()
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "mean_ranks": ranks.to_dict(),
        "n_folds": int(scores.shape[0]),
        "n_techniques": int(scores.shape[1]),
        "significant_at_05": bool(p < 0.05),
    }


def nemenyi_critical_difference(n_techniques: int, n_folds: int) -> float:
    """Critical difference for the Nemenyi post-hoc test at p=0.05.

    Two techniques differ significantly if their mean ranks differ by more
    than this value. Report it as the CD bar on the rank plot (Chapter 5).
    """
    if n_techniques not in _Q_ALPHA_05:
        raise ValueError(f"No tabulated q_alpha for k={n_techniques}")
    q = _Q_ALPHA_05[n_techniques]
    k, n = n_techniques, n_folds
    return float(q * np.sqrt(k * (k + 1) / (6 * n)))


def nemenyi_posthoc(scores: pd.DataFrame) -> pd.DataFrame:
    """Pairwise significance matrix (True = significantly different)."""
    ranks = scores.rank(axis=1, ascending=False).mean()
    cd = nemenyi_critical_difference(scores.shape[1], scores.shape[0])
    cols = list(scores.columns)
    mat = pd.DataFrame(False, index=cols, columns=cols)
    for a in cols:
        for b in cols:
            if a != b:
                mat.loc[a, b] = abs(ranks[a] - ranks[b]) > cd
    return mat


def results_table(
    rows: Sequence[dict],
    metrics: Sequence[str] = ("auc_pr", "precision", "recall", "f1", "mcc"),
    group_by: str = "technique",
) -> pd.DataFrame:
    """Aggregate per-fold metric rows into a mean ± std table for Chapter 5."""
    df = pd.DataFrame(list(rows))
    agg = df.groupby(group_by)[list(metrics)].agg(["mean", "std"])
    out = pd.DataFrame(index=agg.index)
    for m in metrics:
        out[m] = [
            f"{mu:.4f} ± {sd:.4f}" if pd.notna(sd) else f"{mu:.4f}"
            for mu, sd in zip(agg[(m, "mean")], agg[(m, "std")])
        ]
    return out.sort_values(metrics[0], ascending=False)
