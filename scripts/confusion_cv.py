"""Pooled cross-validation confusion matrices for Chapter 5 sec 5.5.

Sums tp/fp/fn/tn over the 5 training-split folds in experiments/results_raw.csv
for a few illustrative (technique, classifier) pairs, so the cells always add
to the 192,963 training rows. Thresholds were chosen per fold on the scored
fold (F1), so these matrices are OPTIMISTIC; the test-split matrices come from
notebooks/03_results_analysis.ipynb (Table 5/6) once the data is available.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# best untreated, best AUC-PR, lowest cost, most conservative threshold,
# and two failure modes (undersampling flood, LightGBM defaults).
PAIRS = [
    ("none", "random_forest"),
    ("random_oversampling", "xgboost"),
    ("xgb_focal_loss", "xgboost"),
    ("smote", "xgboost"),
    ("random_undersampling", "decision_tree"),
    ("lgbm_is_unbalance", "lightgbm"),
]


def pooled(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tech, clf in PAIRS:
        sub = results[(results.technique == tech) & (results.classifier == clf)]
        assert sub.fold.nunique() == config.N_FOLDS, (tech, clf)
        c = sub[["tp", "fp", "fn", "tn"]].sum()
        rows.append({
            "technique": tech, "classifier": clf,
            "auc_pr": sub.auc_pr.mean(),
            "thr_mean": sub.threshold.mean(),
            "thr_min": sub.threshold.min(), "thr_max": sub.threshold.max(),
            **c.astype(int).to_dict(),
            "total_cost": config.COST_FN * c.fn + config.COST_FP * c.fp,
        })
    return pd.DataFrame(rows)


def plot(cells: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4))
    for ax, r in zip(axes.ravel(), cells.itertuples()):
        ax.imshow([[0, 1], [1, 0]], cmap="Greys", vmin=0, vmax=3)
        for i, row in enumerate([[r.tn, r.fp], [r.fn, r.tp]]):
            for j, v in enumerate(row):
                ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=11,
                        color="black")
        ax.set_xticks([0, 1], ["Legit", "Fraud"])
        ax.set_yticks([0, 1], ["Legit", "Fraud"])
        ax.set_title(f"{r.technique} / {r.classifier}", fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("Predicted")
    for ax in axes[:, 0]:
        ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cells = pooled(pd.read_csv(config.RESULTS_RAW_CSV))
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cells.to_csv(config.TABLES_DIR / "table7_confusion_cv.csv", index=False)
    plot(cells, config.FIGURES_DIR / "fig9_confusion_cv.png")
    print(cells.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
