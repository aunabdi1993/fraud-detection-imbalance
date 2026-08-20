# Chapter 4: System Architecture

## 4.1 Requirements
Real-time scoring, p99 under 100 ms, operator-tunable threshold.

## 4.2 Design
Model + scaler loaded once at startup. Threshold as deployment parameter.
Discussion of train/serve skew.

## 4.3 Experiment Tracking
MLflow: parameters, metrics, git commit per run.

## 4.4 Testing and Validation
Leakage guards, API contract tests.
