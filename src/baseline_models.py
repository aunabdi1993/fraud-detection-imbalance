"""
baseline_models.py — Untreated baselines.

Maps to: Chapter 3 sec 3.4, Chapter 5 sec 5.1.
STATUS: STUB — implement Week 15-16.

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
Chapter 3 so the comparison is defensible.
"""

from __future__ import annotations

raise_msg = "Implement in Week 15-16 — see module docstring"


def get_baseline_models(random_state: int = 42) -> dict:
    raise NotImplementedError(raise_msg)


def train_baselines(X_train, y_train, X_val, y_val) -> "list[dict]":
    raise NotImplementedError(raise_msg)
