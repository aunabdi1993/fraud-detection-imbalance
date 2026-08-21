"""
baseline_models.py — Untreated baselines.

Maps to: Chapter 3 sec 3.4, Chapter 5 sec 5.1.
STATUS: Week 15-16 implementation.

These models receive NO imbalance handling. They are the reference point every
technique is measured against, and they demonstrate the core problem: expect
near-perfect accuracy alongside poor recall on the fraud class.

Include a DummyClassifier(strategy="most_frequent") in the results table. It
will score ~99.83% accuracy and 0.0 recall — the single clearest illustration
of why accuracy is excluded as a headline metric (Chapter 1 sec 1.1).

Models: DummyClassifier, LogisticRegression(max_iter=1000),
DecisionTreeClassifier, RandomForestClassifier(n_estimators=100, n_jobs=-1),
XGBClassifier(eval_metric="aucpr").

Hyperparameter budget must be IDENTICAL across every technique (Gap 3) —
same search space, same number of trials, same CV. Record the budget in
Chapter 3 so the comparison is defensible. Baselines use each model's library
defaults (no search) precisely because they are the reference point, not a
tuned contender; every subsequent technique in imbalance_methods.py must be
tuned with the same budget as its peers, not as these baselines.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config
from .data_loader import FraudDataset
from .evaluation import CostModel, choose_threshold, evaluate

logger = logging.getLogger(__name__)

DISPLAY_COLUMNS = [
    "technique", "auc_pr", "auc_roc", "precision", "recall", "f1", "mcc",
    "threshold", f"precision_at_{config.ALERT_BUDGET}",
    f"recall_at_{config.ALERT_BUDGET}",
]


def get_baseline_models(random_state: int = config.RANDOM_STATE) -> dict:
    """Untreated reference models, library defaults only.

    No resampling, no class weighting, no threshold tuning at fit time — that
    is the point. Every deviation from these defaults belongs to a named
    technique in imbalance_methods.py, not here.
    """
    return {
        "dummy_most_frequent": DummyClassifier(
            strategy="most_frequent", random_state=random_state
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=random_state
        ),
        "xgboost": XGBClassifier(
            eval_metric="aucpr", random_state=random_state, n_jobs=-1
        ),
    }


def train_baselines(X_train, y_train, X_val, y_val) -> list[dict]:
    """Fit each untreated baseline and score it on validation.

    Threshold is chosen on validation via choose_threshold() (rule 3) — never
    on a held-out test split, which does not exist at this stage. Returns one
    metric row per model, ready for evaluation.results_table().
    """
    cost_model = CostModel(cost_fn=config.COST_FN, cost_fp=config.COST_FP)
    rows = []
    for name, model in get_baseline_models().items():
        logger.info("Fitting baseline: %s", name)
        model.fit(X_train, y_train)

        p_val = model.predict_proba(X_val)[:, 1]
        if np.unique(p_val).size > 1:
            threshold = choose_threshold(y_val, p_val, objective="f1")
        else:
            # DummyClassifier(strategy="most_frequent") emits a constant score,
            # so there is no PR curve to search: every threshold below it
            # predicts "all positive" and every threshold above predicts "all
            # negative". choose_threshold's F1 search would pick the former
            # (nonzero recall beats zero precision under F1), which hides the
            # exact accuracy-paradox behaviour this baseline exists to show.
            # Fall back to the canonical 0.5 cut instead.
            threshold = 0.5

        row = evaluate(
            y_val,
            p_val,
            threshold=threshold,
            cost_model=cost_model,
            alert_budget=config.ALERT_BUDGET,
        )
        row["technique"] = name
        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(config.DATA_PROCESSED))
    parser.add_argument("--out", default=str(config.TABLES_DIR))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
    )

    splits, _ = FraudDataset.load_splits(args.data)
    rows = train_baselines(
        splits.X_train, splits.y_train, splits.X_val, splits.y_val
    )

    table = pd.DataFrame(rows)[DISPLAY_COLUMNS].sort_values(
        "auc_pr", ascending=False
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "table1_baselines.csv"
    table.to_csv(out_path, index=False)
    logger.info("Saved baseline results table to %s", out_path)

    print("\n" + table.to_string(index=False))


if __name__ == "__main__":
    main()
