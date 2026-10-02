"""
results_analysis.py — Chapter 5 tables and figures.

Maps to: Chapter 5 sec 5.1-5.5. Driven by notebooks/03_results_analysis.ipynb.

Every number and figure in Chapter 5 is computed here rather than in notebook
cells, so it comes from a committed, testable script (CLAUDE.md). The
notebook calls these functions in order, displays the output, and saves it
to dissertation/tables and dissertation/figures.

Inputs:
  experiments/results_raw.csv       the sweep, one row per (pair, fold)
  experiments/latency_results.csv   the profiler
  data/processed/                   the splits, for the final models

Two stages, kept strictly apart (rule 3):
  1. Cross-validation results on the TRAINING split rank all 41 pairs
     (Table 2), feed the Friedman-Nemenyi tests (Table 4) and Figure 8.
  2. The top config.N_TOP_TECHNIQUES pairs by CV AUC-PR, plus the best
     untreated baseline, are refitted on the full training split. Their
     thresholds and operating points are chosen on VALIDATION and applied
     unchanged to TEST, which is used exactly once, here. Nothing is selected
     by looking at test results.

Statistical caveat for Chapter 5 sec 5.3: Demsar's (2006) Friedman-Nemenyi
procedure assumes independent datasets as blocks. Here the blocks are CV
folds (or classifier x fold), which share training data, so the tests are
somewhat liberal (Dietterich 1998). With only 5 folds they are also low
powered, so "no significant difference" is weak evidence of equivalence.
Report both caveats; they support the literature's "no consistent winner"
rather than undermine the comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config
from .data_loader import Splits
from .evaluation import (
    CostModel,
    auc_pr,
    bootstrap_ci,
    choose_threshold,
    evaluate,
    friedman_test,
    nemenyi_critical_difference,
)
from .imbalance_methods import get_model, resample_fold

logger = logging.getLogger(__name__)

LATENCY_CSV = config.EXPERIMENTS_DIR / "latency_results.csv"

# Chart styling: the validated light-mode reference palette from the dataviz
# guidance. Categorical slots are used in this fixed order, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]
TYPE_COLOURS = {  # three slots: validated for scatter (all pairs compared)
    "untreated": SERIES[0], "data-level": SERIES[1], "algorithm-level": SERIES[2],
}
CLASSIFIER_SYMBOLS = {
    "logistic_regression": "circle", "decision_tree": "square",
    "random_forest": "diamond", "xgboost": "triangle-up",
    "adaboost": "star", "lightgbm": "hexagon",
}


def pair_label(technique: str, classifier: str) -> str:
    return f"{technique} · {classifier}"


def technique_type(technique: str) -> str:
    if technique == "none":
        return "untreated"
    if technique in config.RESAMPLING_TECHNIQUES:
        return "data-level"
    return "algorithm-level"


# ---------------------------------------------------------------------------
# Stage 1: cross-validation
# ---------------------------------------------------------------------------
def load_results(path: str | Path = config.RESULTS_RAW_CSV) -> pd.DataFrame:
    results = pd.read_csv(path)
    counts = results.groupby(["technique", "classifier"])["fold"].nunique()
    incomplete = counts[counts != config.N_FOLDS]
    if len(incomplete):
        logger.warning("Pairs without all %d folds: %s",
                       config.N_FOLDS, list(incomplete.index))
    return results


def cv_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Table 2: every pair's CV metrics, mean (and std for AUC-PR), by AUC-PR.

    Threshold-dependent columns use a threshold picked on the same fold, so
    they are optimistic (see experiment_runner.py); AUC-PR is not affected.
    """
    k = config.ALERT_BUDGET
    table = (
        results.groupby(["technique", "classifier"])
        .agg(
            auc_pr=("auc_pr", "mean"), auc_pr_std=("auc_pr", "std"),
            auc_roc=("auc_roc", "mean"),
            **{f"precision_at_{k}": (f"precision_at_{k}", "mean")},
            f1=("f1", "mean"), mcc=("mcc", "mean"),
            fit_seconds=("fit_seconds", "mean"), n_train=("n_train", "mean"),
        )
        .reset_index()
        .sort_values(["auc_pr", "auc_pr_std"], ascending=[False, True])
        .reset_index(drop=True)
    )
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table.insert(3, "type", table["technique"].map(technique_type))
    return table


def top_pairs(results: pd.DataFrame, n: int = config.N_TOP_TECHNIQUES
              ) -> list[tuple[str, str]]:
    """The n best pairs by mean CV AUC-PR — chosen before test is touched."""
    head = cv_summary(results).head(n)
    return list(zip(head["technique"], head["classifier"]))


def best_untreated(results: pd.DataFrame) -> tuple[str, str]:
    """The strongest no-treatment baseline: the reference every technique
    has to beat to be worth its complexity."""
    table = cv_summary(results)
    row = table[table["technique"] == "none"].iloc[0]
    return row["technique"], row["classifier"]


def load_latency(path: str | Path = LATENCY_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def latency_table(latency: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Table 3: single-record latency per pair beside its CV AUC-PR.

    AUC-PR is joined from the sweep (with its std, for Figure 8's error
    bars) rather than taken from the profiler's copy, so both tables quote
    the same number.
    """
    cv = cv_summary(results)[["technique", "classifier", "auc_pr", "auc_pr_std"]]
    single = latency[latency["batch_size"] == 1]
    table = single[
        ["technique", "classifier", "p50_ms", "p95_ms", "p99_ms", "mean_ms",
         "model_size_mb", "cpu_utilisation"]
    ].merge(cv, on=["technique", "classifier"], how="left")
    table.insert(2, "type", table["technique"].map(technique_type))
    table["within_budget"] = table["p99_ms"] <= config.LATENCY_BUDGET_MS
    return table.sort_values("p99_ms").reset_index(drop=True)


def throughput_table(latency: pd.DataFrame) -> pd.DataFrame:
    """Per-record cost by batch size: how much batching amortises."""
    return latency.pivot_table(
        index=["technique", "classifier"], columns="batch_size",
        values="per_record_ms",
    ).sort_values(1)


@dataclass
class RankResult:
    """Friedman test plus Nemenyi post-hoc for one comparison."""

    name: str
    table: pd.DataFrame          # one row per treatment, best first
    statistic: float
    p_value: float
    n_blocks: int
    critical_difference: float
    reference: str | None


def rank_comparison(
    scores: pd.DataFrame, name: str, reference: str | None = None
) -> RankResult:
    """Friedman-Nemenyi over `scores` (rows = blocks, columns = treatments).

    Each treatment is flagged if its mean rank is more than one critical
    difference from the best treatment, and from `reference` (the untreated
    baseline) when given. Those flags are the significance annotations.
    """
    scores = scores.dropna(axis=1, how="any")
    friedman = friedman_test(scores)
    cd = nemenyi_critical_difference(scores.shape[1], scores.shape[0])
    ranks = pd.Series(friedman["mean_ranks"])
    best = ranks.idxmin()
    table = pd.DataFrame({
        "treatment": ranks.index,
        "mean_rank": ranks.to_numpy(),
        "mean_auc_pr": scores.mean()[ranks.index].to_numpy(),
        "worse_than_best": (ranks - ranks[best] > cd).to_numpy(),
    })
    if reference is not None and reference in ranks:
        table["differs_from_reference"] = (
            (ranks - ranks[reference]).abs() > cd
        ).to_numpy()
    return RankResult(
        name=name, table=table, statistic=friedman["statistic"],
        p_value=friedman["p_value"], n_blocks=scores.shape[0],
        critical_difference=cd, reference=reference,
    )


def data_level_scores(results: pd.DataFrame) -> pd.DataFrame:
    """AUC-PR of the 8 data-level treatments, blocked by classifier x fold.

    Crossing every sampler with all four base classifiers is what allows
    20 blocks rather than 5, and asks the clean question "does this
    resampling technique help, whatever the classifier?".
    """
    data = results[results["technique"].isin(config.RESAMPLING_TECHNIQUES)
                   & results["classifier"].isin(config.BASE_CLASSIFIERS)]
    wide = data.pivot_table(index=["classifier", "fold"], columns="technique",
                            values="auc_pr")
    # Config order; techniques not (yet) in the sweep are simply absent.
    return wide[[t for t in config.RESAMPLING_TECHNIQUES if t in wide.columns]]


def family_scores(results: pd.DataFrame, classifier: str) -> pd.DataFrame:
    """AUC-PR of every technique within one classifier family, by fold.

    The family includes its algorithm-level variants (e.g. random_forest
    includes balanced_random_forest), so data- and algorithm-level fixes are
    compared on equal terms against the family's untreated model.
    """
    family = results[results["classifier"] == classifier]
    return family.pivot_table(index="fold", columns="technique", values="auc_pr")


def family_rankings(results: pd.DataFrame) -> dict[str, RankResult]:
    """One comparison per base classifier with an untreated reference."""
    return {
        clf: rank_comparison(family_scores(results, clf), clf, reference="none")
        for clf in config.BASE_CLASSIFIERS
    }


# ---------------------------------------------------------------------------
# Stage 2: final models on the test split
# ---------------------------------------------------------------------------
@dataclass
class FinalModel:
    """A pair refitted on the full training split and scored once."""

    technique: str
    classifier: str
    model: Any
    p_val: np.ndarray
    p_test: np.ndarray
    threshold: float   # max-F1 on validation, applied unchanged to test

    @property
    def label(self) -> str:
        return pair_label(self.technique, self.classifier)


def fit_final_models(
    pairs: Sequence[tuple[str, str]], splits: Splits
) -> list[FinalModel]:
    """Refit each pair on the whole training split, then score val and test.

    Same pipeline as the sweep: resample the training data only, fit with
    the same defaults and seed. The validation split picks the threshold.
    """
    finals = []
    for technique, classifier in pairs:
        logger.info("Refitting %s on the full training split",
                    pair_label(technique, classifier))
        X_fit, y_fit = resample_fold(splits.X_train, splits.y_train, technique)
        model = get_model(classifier, technique, y_train=y_fit).fit(X_fit, y_fit)
        p_val = model.predict_proba(splits.X_val)[:, 1]
        finals.append(FinalModel(
            technique=technique, classifier=classifier, model=model,
            p_val=p_val, p_test=model.predict_proba(splits.X_test)[:, 1],
            threshold=choose_threshold(splits.y_val, p_val, objective="f1"),
        ))
    return finals


def paired_bootstrap_delta(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_boot: int = config.BOOTSTRAP_N,
    alpha: float = config.ALPHA,
    random_state: int = config.RANDOM_STATE,
) -> tuple[float, float, float]:
    """AUC-PR(a) - AUC-PR(b) with a paired, stratified percentile CI.

    Paired: both models are scored on the SAME resampled rows each time, so
    the interval reflects the difference between them rather than the much
    larger shared noise of a 71-fraud test set. Two overlapping individual
    CIs do not imply no difference; this interval answers that directly.
    """
    y_true = np.asarray(y_true).astype(int)
    rng = np.random.default_rng(random_state)
    pos, neg = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        y_b = y_true[idx]
        deltas[i] = auc_pr(y_b, score_a[idx]) - auc_pr(y_b, score_b[idx])
    return (
        auc_pr(y_true, score_a) - auc_pr(y_true, score_b),
        float(np.percentile(deltas, 100 * alpha / 2)),
        float(np.percentile(deltas, 100 * (1 - alpha / 2))),
    )


def test_table(finals: Sequence[FinalModel], splits: Splits) -> pd.DataFrame:
    """Table 5: test-split results of the final models.

    AUC-PR carries a stratified bootstrap 95% CI. delta_vs_selected compares
    each model with the CV-selected best (finals[0]) on the same bootstrap
    rows; significant means that interval excludes zero. Threshold-dependent
    metrics use each model's validation-chosen threshold.
    """
    cost = CostModel(cost_fn=config.COST_FN, cost_fp=config.COST_FP)
    selected = finals[0]
    rows = []
    for fm in finals:
        point, lo, hi = bootstrap_ci(
            splits.y_test, fm.p_test, n_boot=config.BOOTSTRAP_N,
            alpha=config.ALPHA, random_state=config.RANDOM_STATE,
        )
        delta, d_lo, d_hi = paired_bootstrap_delta(
            splits.y_test, fm.p_test, selected.p_test
        )
        metrics = evaluate(splits.y_test, fm.p_test, threshold=fm.threshold,
                           cost_model=cost, alert_budget=config.ALERT_BUDGET)
        rows.append({
            "pair": fm.label, "type": technique_type(fm.technique),
            "auc_pr": point, "auc_pr_lo": lo, "auc_pr_hi": hi,
            "delta_vs_selected": delta, "delta_lo": d_lo, "delta_hi": d_hi,
            "significant": fm is not selected and (d_lo > 0 or d_hi < 0),
            **{k: metrics[k] for k in (
                "threshold", "precision", "recall", "f1", "mcc",
                f"precision_at_{config.ALERT_BUDGET}", "tp", "fp", "fn", "tn",
                "total_cost")},
        })
    return pd.DataFrame(rows)


def operating_points(fm: FinalModel, splits: Splits) -> pd.DataFrame:
    """Table 6: candidate operating thresholds for one model.

    Each threshold is chosen on VALIDATION by a different business rule,
    then applied to TEST, so the table shows what operations would actually
    get. The threshold is a deployment parameter, not a model property
    (api/main.py point 2): this is the menu operations choose from.
    """
    cost = CostModel(cost_fn=config.COST_FN, cost_fp=config.COST_FP)
    alert_rate = config.ALERT_BUDGET / config.TRANSACTIONS_PER_DAY
    rules = {
        "max F1": choose_threshold(splits.y_val, fm.p_val, "f1"),
        "min expected cost": choose_threshold(
            splits.y_val, fm.p_val, "cost", cost_model=cost),
        f"precision >= {config.OPERATING_MIN_PRECISION:.0%}": choose_threshold(
            splits.y_val, fm.p_val, "precision",
            min_precision=config.OPERATING_MIN_PRECISION),
        f"{config.ALERT_BUDGET} alerts/day": float(
            np.quantile(fm.p_val, 1 - alert_rate)),
    }
    rows = []
    for rule, threshold in rules.items():
        m = evaluate(splits.y_test, fm.p_test, threshold=threshold, cost_model=cost)
        rows.append({
            "rule": rule, "threshold": threshold,
            "alerts_per_day": m["n_alerts"] / len(splits.y_test)
            * config.TRANSACTIONS_PER_DAY,
            **{k: m[k] for k in ("precision", "recall", "tp", "fp", "fn",
                                 "total_cost")},
        })
    return pd.DataFrame(rows)


def threshold_sweep(
    y_true: np.ndarray, y_score: np.ndarray, n_points: int = 400
) -> pd.DataFrame:
    """False positives, false negatives and cost across thresholds.

    Thresholds combine score quantiles (dense where most scores are, near 0)
    with a log-spaced grid up to 1. Quantiles alone leave the sparse upper
    range empty, and a line drawn across the gap would show data that is
    not there.
    """
    y_true = np.asarray(y_true).astype(int)
    floor = max(float(np.min(y_score[y_score > 0], initial=1.0)), 1e-6)
    thresholds = np.unique(np.concatenate([
        np.quantile(y_score, np.linspace(0, 1, n_points)),
        np.logspace(np.log10(floor), 0, n_points),
    ]))
    thresholds = thresholds[thresholds > 0]
    flagged = y_score[None, :] >= thresholds[:, None]
    fp = (flagged & (y_true == 0)).sum(axis=1)
    fn = (~flagged & (y_true == 1)).sum(axis=1)
    return pd.DataFrame({
        "threshold": thresholds, "fp": fp, "fn": fn,
        "total_cost": fp * config.COST_FP + fn * config.COST_FN,
    })


def shap_importance(
    fm: FinalModel, splits: Splits, feature_names: Sequence[str]
) -> pd.DataFrame:
    """Mean |SHAP| per feature for one final model.

    Computed on VALIDATION rows (every fraud plus config.SHAP_N_LEGIT
    legitimate rows), keeping test for performance only. Interpretation is
    limited: V1-V28 are anonymised PCA components, so "V14 matters" says
    nothing about which real transaction attribute matters (Chapter 6
    sec 6.4).
    """
    import shap

    rng = np.random.default_rng(config.RANDOM_STATE)
    y = splits.y_val
    legit = rng.choice(np.flatnonzero(y == 0),
                       size=min(config.SHAP_N_LEGIT, int((y == 0).sum())),
                       replace=False)
    X = splits.X_val[np.concatenate([np.flatnonzero(y == 1), legit])]
    if hasattr(fm.model, "coef_"):
        explainer = shap.LinearExplainer(fm.model, splits.X_train)
    else:
        explainer = shap.TreeExplainer(fm.model)
    values = np.asarray(explainer.shap_values(X))
    if values.ndim == 3:   # (rows, features, classes): keep the fraud class
        values = values[..., 1]
    return (
        pd.DataFrame({"feature": list(feature_names),
                      "mean_abs_shap": np.abs(values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style(fig: go.Figure, title: str, height: int = 480) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "xref": "paper",
               "font": {"size": 16, "color": INK}},
        font={"family": "system-ui, -apple-system, Segoe UI, sans-serif",
              "size": 12, "color": INK_2},
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
        margin={"l": 64, "r": 24, "t": 64, "b": 56},
        hoverlabel={"bgcolor": "white", "font": {"color": INK}},
        legend={"bgcolor": "rgba(0,0,0,0)"},
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False,
                     ticks="outside", tickcolor=AXIS)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False,
                     ticks="outside", tickcolor=AXIS)
    return fig


def save_figure(fig: go.Figure, name: str,
                out_dir: str | Path = config.FIGURES_DIR) -> None:
    """Write interactive HTML always, and a static PNG for the dissertation
    when plotly's exporter (kaleido + Chrome) is available."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.write_html(out / f"{name}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(out / f"{name}.png", scale=2)
    except Exception as err:  # exporter missing; HTML still written
        logger.warning("PNG export of %s skipped (%s). Run `plotly_get_chrome` "
                       "once to enable it.", name, type(err).__name__)


def fig_auc_pr_vs_latency(table: pd.DataFrame) -> go.Figure:
    """Figure 8: CV AUC-PR against single-record p99 latency.

    Upper-left is deployable and accurate; anything right of the budget line
    fails the real-time requirement whatever its AUC-PR. Colour is the kind
    of treatment (3 hues, the most a scatter can separate reliably); marker
    shape is the classifier; hover gives the exact pair and numbers.
    """
    fig = go.Figure()
    for kind, colour in TYPE_COLOURS.items():
        sub = table[table["type"] == kind]
        fig.add_trace(go.Scatter(
            x=sub["p99_ms"], y=sub["auc_pr"], mode="markers", name=kind,
            legendgroup=kind, showlegend=False,
            error_y={"type": "data", "array": sub["auc_pr_std"],
                     "color": colour, "thickness": 1, "width": 0},
            marker={"color": colour, "size": 11,
                    "symbol": sub["classifier"].map(CLASSIFIER_SYMBOLS),
                    "line": {"color": SURFACE, "width": 2}},
            customdata=np.stack([sub["technique"], sub["classifier"],
                                 sub["p50_ms"], sub["model_size_mb"]], axis=1),
            hovertemplate=("<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                           "AUC-PR %{y:.3f}<br>p99 %{x:.3f} ms · p50 "
                           "%{customdata[2]:.3f} ms<br>model %{customdata[3]:.2f}"
                           " MB<extra></extra>"),
        ))
    # Legends drawn with dummy traces, so the colour key shows neutral dots
    # and the shape key neutral grey, each encoding read on its own.
    for kind, colour in TYPE_COLOURS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=kind, legendgroup=kind,
            legendgrouptitle_text="treatment" if kind == "untreated" else None,
            marker={"color": colour, "size": 10, "symbol": "circle"},
        ))
    for clf in table["classifier"].unique():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=clf,
            legendgroup="classifier", legendgrouptitle_text="classifier",
            marker={"color": MUTED, "size": 10,
                    "symbol": CLASSIFIER_SYMBOLS.get(clf, "circle")},
        ))
    fig.add_vline(x=config.LATENCY_BUDGET_MS, line={"color": INK_2, "dash": "dash",
                                                    "width": 1.5})
    fig.add_annotation(x=np.log10(config.LATENCY_BUDGET_MS), y=1, yref="paper",
                       text=f"{config.LATENCY_BUDGET_MS:.0f} ms p99 budget",
                       showarrow=False, xanchor="right", yanchor="top",
                       font={"color": INK_2})
    fig.update_xaxes(
        type="log", title="single-record p99 latency (ms, log scale)",
        range=[np.log10(table["p99_ms"].min() / 1.5),
               np.log10(config.LATENCY_BUDGET_MS * 2.5)],
        tickvals=[0.01, 0.1, 1, 10, 100, 1000],
        ticktext=["0.01", "0.1", "1", "10", "100", "1000"],
    )
    fig.update_yaxes(title="CV AUC-PR (mean ± std over 5 folds)")
    return _style(fig, "Figure 8 — Detection performance vs inference latency",
                  height=560)


def fig_pr_curves(finals: Sequence[FinalModel], y_test: np.ndarray,
                  tests: pd.DataFrame, reference: str | None = None) -> go.Figure:
    """Test-split PR curves. The untreated reference is drawn muted and
    dashed so the top techniques read as the subject. The flat line at the
    fraud rate is a no-skill classifier's PR curve."""
    from sklearn.metrics import precision_recall_curve

    fig = go.Figure()
    ci = tests.set_index("pair")
    colours = iter(SERIES)
    for fm in finals:
        prec, rec, _ = precision_recall_curve(y_test, fm.p_test)
        row = ci.loc[fm.label]
        is_ref = fm.label == reference
        fig.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines",
            name=(f"{fm.label}: {row.auc_pr:.3f} "
                  f"[{row.auc_pr_lo:.3f}, {row.auc_pr_hi:.3f}]"
                  + (" (untreated reference)" if is_ref else "")),
            line={"color": MUTED if is_ref else next(colours), "width": 2,
                  "dash": "dash" if is_ref else "solid", "shape": "hv"},
            hovertemplate=f"{fm.label}<br>recall %{{x:.3f}}<br>precision "
                          "%{y:.3f}<extra></extra>",
        ))
    rate = float(np.mean(y_test))
    fig.add_hline(y=rate, line={"color": AXIS, "width": 1})
    fig.add_annotation(x=0, y=rate, text=f"no skill ({rate:.4f})", showarrow=False,
                       xanchor="left", yanchor="bottom", font={"color": MUTED})
    fig.update_xaxes(title="recall", range=[0, 1.01])
    fig.update_yaxes(title="precision", range=[0, 1.02])
    fig.update_layout(legend={"title": "test AUC-PR [95% bootstrap CI]",
                              "yanchor": "bottom", "y": 0.02,
                              "xanchor": "left", "x": 0.02})
    return _style(fig, "Precision-recall on the test split", height=560)


def fig_confusion_matrices(finals: Sequence[FinalModel],
                           y_test: np.ndarray) -> go.Figure:
    """Test confusion matrices at each model's validation-chosen threshold.

    Colour is the row rate (share of each true class), because raw counts
    span 70 frauds to 41,000 legitimate rows and would leave the fraud row
    blank. The cell text gives the counts.
    """
    from sklearn.metrics import confusion_matrix

    n_cols = 3
    n_rows = -(-len(finals) // n_cols)
    fig = make_subplots(
        rows=n_rows, cols=n_cols, horizontal_spacing=0.08, vertical_spacing=0.2,
        subplot_titles=[f"{f.technique}<br>{f.classifier}" for f in finals],
    )
    labels = ["legitimate", "fraud"]
    for i, fm in enumerate(finals, 1):
        r, c = (i - 1) // n_cols + 1, (i - 1) % n_cols + 1
        cm = confusion_matrix(y_test, (fm.p_test >= fm.threshold).astype(int),
                              labels=[0, 1])
        rate = cm / cm.sum(axis=1, keepdims=True)
        fig.add_trace(go.Heatmap(
            z=rate, x=labels, y=labels, zmin=0, zmax=1,
            colorscale=[[i / (len(SEQUENTIAL) - 1), c]
                        for i, c in enumerate(SEQUENTIAL)],
            showscale=i == len(finals), text=cm, texttemplate="%{text:,}",
            textfont={"size": 13},
            colorbar={"title": "row rate", "thickness": 10},
            hovertemplate="true %{y}<br>predicted %{x}<br>%{text:,} rows "
                          "(%{z:.1%} of true class)<extra></extra>",
        ), row=r, col=c)
        fig.update_xaxes(title="predicted" if r == n_rows else None,
                         row=r, col=c)
        fig.update_yaxes(autorange="reversed", title="true" if c == 1 else None,
                         showticklabels=c == 1, row=r, col=c)
    fig.update_annotations(font={"size": 12, "color": INK_2})
    fig = _style(fig, "Confusion matrices on test (threshold chosen on "
                      "validation)", height=330 * n_rows + 100)
    fig.update_layout(margin={"t": 120})
    return fig


def _mark_thresholds(fig: go.Figure, points: pd.DataFrame) -> None:
    """Dotted line and label per operating point (threshold axes are log)."""
    for i, row in enumerate(points.itertuples(index=False)):
        if row.threshold <= 0:
            continue
        fig.add_vline(x=row.threshold, line={"color": MUTED, "dash": "dot",
                                             "width": 1})
        fig.add_annotation(x=np.log10(row.threshold), y=1 - 0.07 * i,
                           yref="paper",
                           text=row.rule, showarrow=False,
                           # Labels near the right edge sit left of the line.
                           xanchor="right" if row.threshold > 0.5 else "left",
                           font={"size": 10, "color": INK_2})


def fig_errors_by_threshold(sweep: pd.DataFrame,
                            points: pd.DataFrame) -> go.Figure:
    """False positives and false negatives on test as the threshold moves.

    Log scale: false positives run to tens of thousands, false negatives to
    at most ~70, and both need to be readable. Dotted lines are the
    operating points, all fixed on validation beforehand; nothing here is
    chosen from the test curve.
    """
    fig = go.Figure()
    for col, name, colour in (("fp", "false positives (false alarms)", SERIES[0]),
                              ("fn", "false negatives (missed fraud)", SERIES[1])):
        fig.add_trace(go.Scatter(
            x=sweep["threshold"], y=sweep[col] + 1, mode="lines", name=name,
            line={"color": colour, "width": 2, "shape": "hv"},
            customdata=sweep[col],
            hovertemplate="threshold %{x:.4f}<br>%{customdata:,}<extra>"
                          + name + "</extra>",
        ))
    _mark_thresholds(fig, points)
    fig.update_xaxes(type="log", title="decision threshold (log scale)",
                     exponentformat="power",
                     range=[np.log10(sweep["threshold"].min()), 0.05])
    fig.update_yaxes(type="log", title="count on test (+1, log scale)",
                     tickvals=[1, 10, 100, 1_000, 10_000, 100_000],
                     ticktext=["1", "10", "100", "1k", "10k", "100k"])
    return _style(fig, "Errors by threshold (test split)")


def fig_cost_by_threshold(sweep: pd.DataFrame, points: pd.DataFrame) -> go.Figure:
    """Expected cost on test, a separate chart rather than a second y-axis."""
    fig = go.Figure(go.Scatter(
        x=sweep["threshold"], y=sweep["total_cost"], mode="lines",
        name="total cost", line={"color": SERIES[0], "width": 2, "shape": "hv"},
        hovertemplate="threshold %{x:.4f}<br>cost %{y:,.0f}<extra></extra>",
    ))
    _mark_thresholds(fig, points)
    fig.update_xaxes(type="log", title="decision threshold (log scale)",
                     exponentformat="power",
                     range=[np.log10(sweep["threshold"].min()), 0.05])
    fig.update_yaxes(type="log", title=f"cost, log scale ({config.COST_FN:.0f} "
                     f"per missed fraud, {config.COST_FP:.0f} per false alarm)",
                     tickvals=[100, 1_000, 10_000, 100_000, 1_000_000],
                     ticktext=["100", "1k", "10k", "100k", "1M"])
    return _style(fig, "Expected cost by threshold (test split)")


def fig_mean_ranks(result: RankResult) -> go.Figure:
    """Mean Friedman ranks with the Nemenyi critical difference.

    The shaded band spans one CD from the best rank: treatments inside it
    are not significantly worse than the best. Treatments outside it are
    drawn muted.
    """
    t = result.table.sort_values("mean_rank", ascending=False)
    best = t["mean_rank"].min()
    fig = go.Figure()
    fig.add_vrect(x0=best, x1=best + result.critical_difference,
                  fillcolor=SEQUENTIAL[0], opacity=0.6, line_width=0,
                  layer="below")
    for flag, colour, name in ((False, SERIES[0], "not significantly worse "
                                "than best"),
                               (True, MUTED, "significantly worse than best")):
        sub = t[t["worse_than_best"] == flag]
        fig.add_trace(go.Scatter(
            x=sub["mean_rank"], y=sub["treatment"], mode="markers", name=name,
            marker={"color": colour, "size": 10, "line": {"color": SURFACE,
                                                          "width": 2}},
            customdata=sub["mean_auc_pr"],
            hovertemplate="%{y}<br>mean rank %{x:.2f}<br>mean AUC-PR "
                          "%{customdata:.3f}<extra></extra>",
        ))
    fig.update_xaxes(title="mean rank (1 = best)")
    fig.update_yaxes(categoryorder="array", categoryarray=list(t["treatment"]))
    return _style(
        fig,
        f"{result.name}<br><sup>Friedman p = {result.p_value:.2g} · Nemenyi "
        f"CD = {result.critical_difference:.2f} over {result.n_blocks} blocks · "
        f"band = within one CD of the best</sup>",
        height=150 + 28 * len(t),
    )


def fig_shap(importance: pd.DataFrame, label: str, top: int = 15) -> go.Figure:
    head = importance.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=head["mean_abs_shap"], y=head["feature"], orientation="h",
        marker={"color": SERIES[0], "cornerradius": 4},
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_xaxes(title="mean |SHAP value| (validation, all frauds + sample)")
    return _style(fig, f"Feature attribution — {label}", height=100 + 26 * len(head))


def save_table(table: pd.DataFrame, name: str,
               out_dir: str | Path = config.TABLES_DIR) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.csv"
    table.round(config.TABLE_PRECISION).to_csv(path, index=False)
    return path
