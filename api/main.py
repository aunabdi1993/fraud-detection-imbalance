"""
main.py — Production fraud scoring API.

Maps to: Chapter 4 (System Architecture). CONTRIBUTION 3.
STATUS: STUB — implement Month 7-8.

Run:  uvicorn api.main:app --reload

Endpoints:
  GET  /health   liveness, returns loaded model id and version
  POST /score    score one transaction -> {fraud_probability, decision, latency_ms}
  GET  /metrics  request count, latency histogram, alert rate

Design points to defend in Chapter 4:
  1. Load the model and scaler ONCE at startup, not per request. The scaler
     must be the exact object fitted in data_loader — re-fitting at serve time
     silently changes the feature distribution the model was trained on. This
     train/serve skew is a classic production failure and worth a paragraph.
  2. The decision threshold is a deployment parameter, not a model property.
     Load it from the manifest. Operations should be able to retune the
     alert rate without retraining.
  3. Return the probability alongside the binary decision, so downstream
     systems can apply their own risk appetite.
  4. Log p50/p95/p99 latency per endpoint — this feeds Chapter 5 sec 5.4 and
     lets you claim measured, not theoretical, deployability.
  5. Validate input with pydantic: 29 or 30 float features, reject anything
     else with 422 rather than returning a garbage score.

Keep it genuinely small. A working, well-documented service beats a
half-finished microservice architecture, and the marks are for the
dissertation, not the infrastructure.
"""

from __future__ import annotations

raise NotImplementedError("Implement in Month 7-8 — see module docstring")
