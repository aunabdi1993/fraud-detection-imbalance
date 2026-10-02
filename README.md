# Class Imbalance Handling for Financial Fraud Detection

MSc dissertation implementation.

> **Research question:** How do different class imbalance handling techniques
> compare in performance and practical deployability for financial fraud detection?

This project does not propose a new algorithm. It provides a controlled
comparison of 15 established techniques on a single dataset with a fixed
pipeline, statistical significance testing, latency measurement, and a
deployable scoring API.

## Quick start

```bash
git clone https://github.com/aunabdi1993/fraud-detection-imbalance.git && cd fraud-detection-imbalance
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

./scripts/get_data.sh                  # dataset is not in git (150 MB)
python src/data_loader.py              # validate, dedup, split, scale
pytest tests/ -v                       # 54 tests
```

## Dataset

ULB European credit-card fraud ([Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)),
284,807 transactions over 48 hours in September 2013, 492 fraudulent
(**0.1727%**, imbalance ratio **1:578**). Features V1–V28 are PCA components;
Time and Amount are raw. No missing values.

The CSV exceeds GitHub's 100 MB file limit and is gitignored. Its SHA256 is
recorded in `data/processed/manifest.json`, so any reported result is traceable
to an exact copy of the input.

## Status

| Module | Purpose | Status |
|---|---|---|
| `src/data_loader.py` | Load, validate, deduplicate, split, scale | Complete |
| `src/evaluation.py` | Metrics, thresholds, bootstrap CIs, Friedman–Nemenyi | Complete |
| `src/config.py` | Experimental constants | Complete |
| `src/preprocessing.py` | Feature engineering | Stub |
| `src/baseline_models.py` | Untreated baselines | Complete |
| `src/imbalance_methods.py` | The 15 techniques | Stub |
| `src/experiment_runner.py` | Full sweep, 300 fitted models | Stub |
| `src/inference_profiler.py` | Latency measurement | Complete |
| `api/main.py` | FastAPI scoring service | Stub |

Each stub carries a detailed specification in its module docstring.

## Methodological commitments

**Resampling happens inside the CV fold, never before the split.** The
validation fold retains the true 1:578 distribution because that is what
deployment looks like. `data_loader.py` has no `imblearn` import by design.

**AUC-PR is primary; accuracy is never a headline figure.** A constant
"legitimate" predictor scores 99.83% here. Only 11.4% of the reviewed corpus
reports AUC-PR despite broad agreement that it is appropriate under imbalance.

**Thresholds are selected on validation and applied unchanged to test.**

**Fixed seed, identical pipeline and tuning budget across every technique** -
addressing the cross-study comparability gap identified in the review.

Two findings from data validation are documented in `dissertation/chapters/ch3_methodology.md`:
9,144 records are duplicates in the feature space actually used (not the 1,081
exact duplicates usually cited), and the fraud rate declines across the 48-hour
window (0.184% / 0.129% / 0.122% by chronological third).

## Layout

```
src/           analysis modules, one per dissertation section
api/           FastAPI scoring service
tests/         pytest suite, including leakage guards
notebooks/     exploratory analysis
scripts/       data acquisition
experiments/   run outputs (gitignored)
dissertation/  chapter drafts, tables, figures
docs/reference/ planning documents and literature review
```

## Commands

```bash
make prepare    # build data/processed from the raw CSV
make test       # run the suite
make profile    # latency of every model the sweep logged to MLflow
make api        # serve the scoring endpoint locally
```

