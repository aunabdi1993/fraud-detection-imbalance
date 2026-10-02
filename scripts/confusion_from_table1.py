"""Recover validation confusion matrices from Table 1 and plot them.

Why this exists: Table 1 stores precision, recall and the validation threshold
but not the raw cells. With the validation split size (N=41,350, 71 frauds;
data/processed/manifest.json) the cells are fully determined:
TP = recall * P, FP = TP / precision - TP, FN = P - TP, TN = N - TP - FP - FN.
Integer recovery is asserted, so a changed table fails loudly.

Output feeds Chapter 5 sec 5.5. Matrices are on the VALIDATION split (the test
split is not yet used; CLAUDE.md rule 3), so they are optimistic.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

N_VAL, P_VAL = 41350, 71  # manifest.json split_summary, val


def recover_cells(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in table.itertuples():
        tp = round(r.recall * P_VAL)
        assert abs(tp - r.recall * P_VAL) < 1e-6, r.technique
        if tp:
            fp = round(tp / r.precision - tp)
            assert abs(tp / r.precision - tp - fp) < 1e-6, r.technique
        else:
            fp = 0  # zero precision with zero alerts (dummy)
        fn = P_VAL - tp
        tn = N_VAL - tp - fp - fn
        rows.append({
            "technique": r.technique, "threshold": r.threshold,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_alerts": tp + fp,
            "total_cost": config.COST_FN * fn + config.COST_FP * fp,
        })
    return pd.DataFrame(rows)


def plot(cells: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, len(cells), figsize=(3.0 * len(cells), 3.4))
    for ax, r in zip(axes, cells.itertuples()):
        m = [[r.tn, r.fp], [r.fn, r.tp]]
        ax.imshow([[0, 1], [1, 0]], cmap="Greys", vmin=0, vmax=3)  # error cells shaded
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{m[i][j]:,}", ha="center", va="center", fontsize=11)
        ax.set_xticks([0, 1], ["Legit", "Fraud"])
        ax.set_yticks([0, 1], ["Legit", "Fraud"])
        ax.set_xlabel("Predicted")
        ax.set_title(f"{r.technique}\nthr={r.threshold:.3g}", fontsize=9)
    axes[0].set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    table = pd.read_csv(config.TABLES_DIR / "table1_baselines.csv")
    cells = recover_cells(table)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cells.to_csv(config.TABLES_DIR / "table_confusion_baselines.csv", index=False)
    plot(cells, config.FIGURES_DIR / "fig_confusion_baselines.png")
    print(cells.to_string(index=False))


if __name__ == "__main__":
    main()
