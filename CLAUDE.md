# CLAUDE.md — Project context for Claude Code

Read this before writing code in this repository.

## What this is

MSc dissertation. Research question:

> How do different class imbalance handling techniques compare in performance
> and practical deployability for financial fraud detection?

This is **not** a novel-algorithm project. The contribution is a rigorous,
controlled comparison of existing techniques, plus latency characterisation and
a working deployed API. The literature review concluded that no technique
consistently wins (Brandt & Lanzén 2020; Gnip et al. 2021) - that finding is the
*justification* for a careful comparison, not a weakness. Do not suggest
inventing a new sampling method.

## Dataset

ULB European credit-card fraud (Kaggle, `mlg-ulb/creditcardfraud`).
Not in git — 150 MB, over GitHub's limit. Run `./scripts/get_data.sh`.

| Property | Value |
|---|---|
| Transactions | 284,807 |
| Fraud | 492 (0.1727%) |
| Imbalance ratio | 1:578 |
| Features | V1–V28 (PCA), Time, Amount |
| Missing values | 0 |
| Collection window | 48 hours, September 2013 |

After feature-space deduplication: 275,663 rows, 473 fraud.

## The four rules

**1. Never resample before splitting.**
Resampling belongs inside the CV fold, applied to the training portion only.
The validation fold keeps the true 1:578 distribution, because that is what
deployment looks like. Resampling first produces synthetic points derived from
validation rows and inflates AUC-PR dramatically. This is the most common error
in the reviewed literature. `src/data_loader.py` deliberately does not import
`imblearn` — keep it that way.

**2. AUC-PR is the primary metric. Accuracy is never a headline number.**
At 0.17% positives, a model predicting "legitimate" always scores 99.83%.
ROC-AUC is also misleading here because the FPR denominator is dominated by
284k negatives. Report AUC-PR first, then precision/recall/F1/MCC, then
precision@k for a realistic alert budget.

**3. Thresholds are chosen on validation, never on test.**
0.5 is arbitrary under extreme imbalance. Use `choose_threshold()` on the
validation split, then apply that fixed number to test.

**4. Fixed seed (42), identical pipeline, identical tuning budget everywhere.**
This directly addresses lit review Gap 3 (cross-study comparison is impossible
when hyperparameter budgets differ). If a technique gets more tuning trials
than another, the comparison is void.

## Layout

```
src/data_loader.py         DONE   load, validate, dedup, split, scale
src/evaluation.py          DONE   metrics, thresholds, bootstrap, Friedman
src/config.py              DONE   all experimental constants
src/preprocessing.py       STUB   Week 13-14  feature engineering
src/baseline_models.py     DONE   Week 15-16  untreated baselines
src/imbalance_methods.py   DONE   Month 5-6   the 15 techniques
src/experiment_runner.py   DONE   Month 5-6   the full sweep
src/inference_profiler.py  DONE   Month 7-8   latency (Contribution 2)
api/main.py                STUB   Month 7-8   FastAPI service (Contribution 3)
```

Every stub has a detailed docstring specifying what to build and why. Read it
before implementing — the design decisions are already made and tied to
specific literature.

## Chapter mapping

| Module | Chapter |
|---|---|
| `data_loader.py` | Ch.3 §3.1–3.3 |
| `preprocessing.py` | Ch.3 §3.3 |
| `evaluation.py` | Ch.3 §3.4, all of Ch.5 |
| `imbalance_methods.py` | Ch.3 §3.5, Ch.5 §5.2 |
| `experiment_runner.py` | Ch.5 §5.1–5.3 |
| `inference_profiler.py` | Ch.5 §5.4 |
| `api/main.py` | Ch.4 |

## Conventions

- Python 3.10+, type hints, `from __future__ import annotations`
- Docstrings explain **why**, with a literature reference where one applies.
  This code is an appendix to an examined document; a marker will read it.
- No magic numbers outside `src/config.py`
- `pytest tests/ -v` must pass before any commit
- Log with `logging`, not `print`
- Numbers that appear in the dissertation come from a committed script, never
  from a REPL session that cannot be reproduced

## Testing

The leakage guards in `tests/test_data_loader.py` are load-bearing:
`test_scaler_fitted_on_training_only` and
`test_no_duplicate_rows_across_train_and_test`. A leak does not raise an
exception — it just makes results wrong in a direction that looks like success.
If you change the pipeline and one of these fails, the pipeline is wrong, not
the test.

## Known methodological points

**Duplicates.** 1,081 rows are exact duplicates across all 31 columns, but
9,144 are duplicates once `Time` is excluded — and `Time` is dropped as a
feature, so those rows are indistinguishable to the model. No duplicate group
carries conflicting labels, so this is redundancy, not label noise. Default
`dedup_scope="features"`. `"none"` reproduces the literature convention; run it
as a sensitivity check.

**Temporal drift.** Fraud rate by chronological third: 0.184% / 0.129% /
0.122%. A `--split temporal` option exists. The window is only 48 hours, so
this is a weak drift test — report it as indicative, not conclusive.

**PCA features.** V1–V28 are anonymised components, so real feature engineering
and meaningful SHAP interpretation are both limited. Say so plainly in
Ch.6 §6.4 rather than over-claiming explainability.

## Do not

- Add deep learning. Out of scope, and the dissertation does not need it.
- Expand feature engineering. The comparison is the contribution.
- Report accuracy as a headline metric.
- Tune on the test split.
- Build elaborate infrastructure. A small working API beats a half-finished
  microservice architecture.
