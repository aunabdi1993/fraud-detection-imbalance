"""
inference_profiler.py — Latency characterisation.

Maps to: Chapter 5 sec 5.4. THIS IS CONTRIBUTION 2 (lit review Gap 1).
STATUS: STUB — implement Month 7-8.

Only 2 of 21 reviewed sources report inference latency at all, and none measure
how the choice of imbalance-handling technique affects it. That is the gap.

The expected finding, worth stating as a hypothesis in Chapter 3: resampling
techniques change only the TRAINING data, so a SMOTE-trained logistic
regression has identical inference cost to an untreated one. Ensemble methods
(EasyEnsemble fits n_estimators independent models) do carry a real inference
penalty. If that holds, the practical conclusion is sharp: resampling is
effectively free at inference time, ensembles are not — so the accuracy gains
of ensembles must clear a higher bar to justify deployment.

Measurement protocol (rigour matters here, it is your contribution):
  - warm up with >=100 discarded predictions (JIT, cache, lazy allocation)
  - measure single-record latency, not batch throughput — a real fraud API
    scores one transaction at a time
  - time.perf_counter(), 1000 trials
  - report p50, p95, p99 and MEDIAN not mean (latency is right-skewed)
  - record hardware, Python version, library versions, and whether any BLAS
    threading was active — pin OMP_NUM_THREADS=1 for comparability
  - also report model size on disk and fit time

Deliverable: Figure 8, accuracy (AUC-PR) vs p99 latency scatter, with the
100 ms budget drawn as a vertical line. Techniques in the upper-left quadrant
are the deployable ones.
"""

from __future__ import annotations

raise_msg = "Implement in Month 7-8 — see module docstring"


def profile_model(model, X_sample, n_trials: int = 1000) -> dict:
    raise NotImplementedError(raise_msg)


def profile_all(models_dir: str) -> "list[dict]":
    raise NotImplementedError(raise_msg)
