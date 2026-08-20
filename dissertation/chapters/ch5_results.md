# Chapter 5: Results

## 5.1 Baseline Performance (no imbalance handling)
Table 1: including DummyClassifier to demonstrate the accuracy paradox
| Model | Accuracy | AUC-PR | Precision | Recall | F1 | MCC |

## 5.2 Technique Comparison
Table 2: mean ± std across 5 folds, sorted by AUC-PR
Figure 2: PR curves, top 3 techniques

## 5.3 Statistical Significance
Friedman statistic, p-value, mean ranks; Nemenyi critical difference diagram.
Report where differences fall inside the bootstrap CI — that is itself a finding.

## 5.4 Latency Characterisation  [CONTRIBUTION 2]
Table 3: fit time, p50/p95/p99 single-record inference, model size
Figure 8: AUC-PR vs p99 latency, with the 100 ms budget marked
Hypothesis: resampling is free at inference; ensembles are not.

## 5.5 Explainability (scope-limited)
SHAP on the selected model. Note the PCA constraint on interpretation.

## 5.6 Sensitivity Analyses
Duplicates retained vs removed; temporal vs stratified split.
