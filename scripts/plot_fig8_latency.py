"""Figure 8: AUC-PR vs p99 single-record latency, 100 ms budget marked.

Reads experiments/latency_results.csv (columns: technique, auc_pr, p99_ms),
which src/inference_profiler.py will write in Month 7-8. Deployable techniques
sit in the upper left (lit review Gap 1). Fails if the file is missing rather
than plotting invented values.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config


def main() -> None:
    df = pd.read_csv(config.EXPERIMENTS_DIR / "latency_results.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(df["p99_ms"], df["auc_pr"], s=36)
    for r in df.itertuples():
        ax.annotate(r.technique, (r.p99_ms, r.auc_pr), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.axvline(config.LATENCY_BUDGET_MS, ls="--", color="grey")
    ax.text(config.LATENCY_BUDGET_MS, ax.get_ylim()[0], " 100 ms budget",
            fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("p99 single-record latency (ms, log scale)")
    ax.set_ylabel("AUC-PR (validation)")
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / "fig8_latency_vs_auc_pr.png", dpi=200)


if __name__ == "__main__":
    main()
