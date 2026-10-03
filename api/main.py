"""
main.py — Fraud scoring API.

Maps to: Chapter 4 (System Architecture). CONTRIBUTION 3.

Run:  uvicorn api.main:app --reload
Needs a model bundle first:  python -m src.export_model --model xgboost

Endpoints:
  GET  /health         model status and measured inference latency (p50/p95/p99)
  POST /predict        one transaction -> probability + decision
  POST /batch-predict  CSV upload -> one prediction per row
  GET  /metrics        validation-set metrics at the current threshold
  GET  /threshold      current decision threshold
  POST /threshold      change the threshold; metrics are recomputed

Design points (Chapter 4):
  1. The model and scaler are loaded ONCE at startup from a bundle written by
     src/export_model.py. The scaler is the object fitted on the training
     split; re-fitting at serve time would cause train/serve skew. Raw
     `Amount` is log1p-transformed here exactly as data_loader did.
  2. The threshold is a deployment parameter held in the bundle and adjustable
     via POST /threshold, so operations can retune the alert rate without
     retraining.
  3. The probability is always returned next to the decision, so downstream
     systems can apply their own risk appetite.
  4. Latency is measured, not assumed: per-endpoint request latency and
     model-only inference latency are kept in a rolling window and reported
     as p50/p95/p99 (feeds Chapter 5 sec 5.4 against config.LATENCY_BUDGET_MS).
  5. Inputs are validated by pydantic / explicit checks; malformed requests
     get 422 (or 413 / 400 for uploads), never a garbage score.

/metrics and POST /threshold use the VALIDATION split only. The test split is
not stored in the bundle, so interactive threshold tuning cannot leak into the
numbers reported in the dissertation (rule 3).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, create_model

from src import config
from src.evaluation import CostModel, evaluate

logger = logging.getLogger("fraud_api")


# ---------------------------------------------------------------------------
# JSON logging
# ---------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    """One JSON object per line, so logs can be parsed for Chapter 5 sec 5.4."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging() -> None:
    if any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def log_event(message: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, message, extra={"fields": fields})


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
_feature_fields: dict[str, Any] = {
    name: (float, Field(allow_inf_nan=False, description=f"Raw feature {name}"))
    for name in config.RAW_FEATURES
}
_feature_fields["Amount"] = (
    float,
    Field(ge=0, allow_inf_nan=False, description="Raw transaction amount"),
)

# Built from config.RAW_FEATURES so the schema cannot drift from the pipeline.
# extra="forbid": an unexpected key is far more likely a typo than a feature.
TransactionInput = create_model(
    "TransactionInput",
    __config__=ConfigDict(extra="forbid"),
    __doc__="One transaction: V1-V28 (PCA components) and raw Amount.",
    **_feature_fields,
)


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float
    model_id: str
    latency_ms: float


class BatchPrediction(BaseModel):
    row: int
    fraud_probability: float
    is_fraud: bool


class BatchResponse(BaseModel):
    n_rows: int
    n_flagged: int
    threshold: float
    model_id: str
    latency_ms: float
    predictions: list[BatchPrediction]


class ThresholdRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    threshold: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


# ---------------------------------------------------------------------------
# Latency tracking
# ---------------------------------------------------------------------------
class LatencyTracker:
    """Rolling window of recent latencies per key, with p50/p95/p99."""

    def __init__(self, window: int = config.API_LATENCY_WINDOW) -> None:
        self._window = window
        self._data: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def record(self, key: str, ms: float) -> None:
        with self._lock:
            self._data.setdefault(key, deque(maxlen=self._window)).append(ms)

    def summary(self, key: str) -> dict[str, float | int | None]:
        with self._lock:
            vals = np.array(self._data.get(key, ()), dtype=float)
        if vals.size == 0:
            return {"n": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None}
        p50, p95, p99 = np.percentile(vals, [50, 95, 99])
        return {
            "n": int(vals.size),
            "p50_ms": round(float(p50), 3),
            "p95_ms": round(float(p95), 3),
            "p99_ms": round(float(p99), 3),
        }

    def keys(self) -> list[str]:
        with self._lock:
            return sorted(self._data)


# ---------------------------------------------------------------------------
# Model service
# ---------------------------------------------------------------------------
class ModelNotLoaded(RuntimeError):
    """Raised when a scoring endpoint is called without a usable bundle."""


class ModelService:
    """Holds the loaded bundle and does preprocessing + scoring."""

    def __init__(self, bundle: dict, latency: LatencyTracker) -> None:
        required = {"model", "scaler", "threshold", "feature_names", "val_y", "val_p"}
        missing = required - set(bundle)
        if missing:
            raise ValueError(f"model bundle is missing keys: {sorted(missing)}")
        if list(bundle["feature_names"]) != config.RAW_FEATURES:
            raise ValueError("bundle feature_names do not match config.RAW_FEATURES")
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.log_amount = bool(bundle.get("log_amount", True))
        self.model_id = str(bundle.get("model_id", "unknown"))
        self.version = str(bundle.get("version", "unknown"))
        self.val_y = np.asarray(bundle["val_y"], dtype=int)
        self.val_p = np.asarray(bundle["val_p"], dtype=float)
        self._threshold = float(bundle["threshold"])
        self._lock = threading.Lock()
        self._latency = latency

    @property
    def threshold(self) -> float:
        return self._threshold

    def set_threshold(self, value: float) -> float:
        with self._lock:
            previous, self._threshold = self._threshold, float(value)
        return previous

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Raw features -> fraud probabilities, replicating training prep."""
        X = frame[config.RAW_FEATURES].to_numpy(dtype=np.float64, copy=True)
        if self.log_amount:
            X[:, -1] = np.log1p(X[:, -1])  # Amount is the last column
        t0 = time.perf_counter()
        proba = self.model.predict_proba(self.scaler.transform(X))[:, 1]
        self._latency.record("inference", (time.perf_counter() - t0) * 1000)
        return proba

    def validation_metrics(self, threshold: float | None = None) -> dict:
        t = self._threshold if threshold is None else threshold
        row = evaluate(
            self.val_y,
            self.val_p,
            threshold=t,
            cost_model=CostModel(config.COST_FN, config.COST_FP),
            alert_budget=config.ALERT_BUDGET,
        )
        keys = [
            "auc_pr", "precision", "recall", "f1", "mcc", "threshold",
            "tp", "fp", "fn", "tn", "n_alerts",
            f"precision_at_{config.ALERT_BUDGET}",
        ]
        out = {k: row[k] for k in keys}
        out["split"] = "validation"
        out["n_samples"] = int(self.val_y.size)
        out["n_fraud"] = int(self.val_y.sum())
        return out


def load_service(path: Path, latency: LatencyTracker) -> ModelService:
    bundle = joblib.load(path)
    service = ModelService(bundle, latency)
    # Warm-up: the first predict_proba call pays one-off initialisation cost
    # that would otherwise inflate the first real request's latency.
    service.score(pd.DataFrame([dict.fromkeys(config.RAW_FEATURES, 0.0)]))
    return service


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app(
    bundle_path: str | Path | None = None,
    timeout_s: float | None = None,
) -> FastAPI:
    configure_logging()
    path = Path(
        bundle_path or os.environ.get("FRAUD_MODEL_PATH") or config.MODEL_BUNDLE_PATH
    )
    timeout = config.API_REQUEST_TIMEOUT_S if timeout_s is None else timeout_s
    latency = LatencyTracker()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = None
        app.state.load_error = None
        app.state.started = time.time()
        app.state.counts = {}
        try:
            app.state.service = load_service(path, latency)
            log_event("model loaded", path=str(path),
                      model_id=app.state.service.model_id)
        except Exception as exc:  # degraded mode: /health reports it
            app.state.load_error = f"{type(exc).__name__}: {exc}"
            log_event("model load failed", logging.ERROR,
                      path=str(path), error=app.state.load_error)
        yield

    app = FastAPI(
        title="Fraud scoring API",
        description="Serves a class-imbalance-handled fraud model (MSc dissertation).",
        version="1.0.0",
        lifespan=lifespan,
    )

    def get_service(request: Request) -> ModelService:
        service = request.app.state.service
        if service is None:
            raise ModelNotLoaded(request.app.state.load_error or "model not loaded")
        return service

    # -- middleware: request id, timeout, JSON access log, latency ----------
    @app.middleware("http")
    async def observe(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        try:
            # Note: wait_for cancels the awaiting task, but a sync endpoint
            # already running in the threadpool finishes in the background.
            # Acceptable here because requests are bounded by the row cap.
            response = await asyncio.wait_for(call_next(request), timeout=timeout)
        except asyncio.TimeoutError:
            response = JSONResponse(
                {"detail": f"request exceeded {timeout:g}s timeout"}, status_code=504
            )
        except Exception:
            log_event("unhandled error", logging.ERROR, request_id=request_id,
                      path=request.url.path)
            logger.exception("unhandled error")
            response = JSONResponse({"detail": "internal server error"},
                                    status_code=500)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        route = request.scope.get("route")
        endpoint = getattr(route, "path", request.url.path)
        latency.record(f"{request.method} {endpoint}", elapsed_ms)
        counts = request.app.state.counts
        counts[response.status_code] = counts.get(response.status_code, 0) + 1
        response.headers["x-request-id"] = request_id
        log_event(
            "request", request_id=request_id, method=request.method,
            path=request.url.path, status=response.status_code,
            latency_ms=round(elapsed_ms, 3),
        )
        return response

    @app.exception_handler(ModelNotLoaded)
    async def model_not_loaded(_: Request, exc: ModelNotLoaded):
        return JSONResponse({"detail": f"model unavailable: {exc}"}, status_code=503)

    # -- endpoints -----------------------------------------------------------
    @app.get("/health")
    def health(request: Request):
        service: ModelService | None = request.app.state.service
        inference = latency.summary("inference")
        body: dict[str, Any] = {
            "status": "ok" if service else "degraded",
            "model_loaded": service is not None,
            "uptime_s": round(time.time() - request.app.state.started, 1),
            "inference_latency": inference,
            "latency_budget_ms": config.LATENCY_BUDGET_MS,
            "endpoint_latency": {k: latency.summary(k) for k in latency.keys()
                                 if k != "inference"},
            "responses_by_status": dict(request.app.state.counts),
        }
        if service:
            body.update(model_id=service.model_id, version=service.version,
                        threshold=service.threshold)
        else:
            body["error"] = request.app.state.load_error
        return JSONResponse(body, status_code=200 if service else 503)

    @app.post("/predict", response_model=PredictionResponse)
    def predict(tx: TransactionInput, request: Request):  # type: ignore[valid-type]
        service = get_service(request)
        t0 = time.perf_counter()
        p = float(service.score(pd.DataFrame([tx.model_dump()]))[0])
        return PredictionResponse(
            fraud_probability=p,
            is_fraud=p >= service.threshold,
            threshold=service.threshold,
            model_id=service.model_id,
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        )

    @app.post("/batch-predict", response_model=BatchResponse)
    async def batch_predict(request: Request, file: UploadFile = File(...)):
        service = get_service(request)
        raw = await file.read(config.API_MAX_UPLOAD_BYTES + 1)
        if len(raw) > config.API_MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"file exceeds {config.API_MAX_UPLOAD_BYTES} bytes")
        try:
            frame = pd.read_csv(io.BytesIO(raw))
        except Exception:
            raise HTTPException(400, "could not parse upload as CSV")
        if frame.empty:
            raise HTTPException(400, "CSV contains no rows")
        if len(frame) > config.API_MAX_BATCH_ROWS:
            raise HTTPException(413, f"more than {config.API_MAX_BATCH_ROWS} rows")
        missing = [c for c in config.RAW_FEATURES if c not in frame.columns]
        if missing:
            raise HTTPException(422, f"missing columns: {missing}")

        # Extra columns (Time, Class) are ignored: Time is dropped in training
        # and labels are never needed to score.
        features = frame[config.RAW_FEATURES].apply(pd.to_numeric, errors="coerce")
        bad = ~np.isfinite(features.to_numpy(dtype=float)).all(axis=1)
        if bad.any():
            rows = (np.flatnonzero(bad)[:10] + 1).tolist()
            raise HTTPException(422, f"non-numeric or non-finite values in rows {rows}")
        if (features["Amount"] < 0).any():
            raise HTTPException(422, "Amount must be non-negative")

        # Offloaded so a large batch does not block the event loop.
        t0 = time.perf_counter()
        proba = await asyncio.to_thread(service.score, features)
        elapsed = (time.perf_counter() - t0) * 1000
        flagged = proba >= service.threshold
        return BatchResponse(
            n_rows=len(proba),
            n_flagged=int(flagged.sum()),
            threshold=service.threshold,
            model_id=service.model_id,
            latency_ms=round(elapsed, 3),
            predictions=[
                BatchPrediction(row=i, fraud_probability=float(p), is_fraud=bool(f))
                for i, (p, f) in enumerate(zip(proba, flagged), start=1)
            ],
        )

    @app.get("/metrics")
    def metrics(request: Request):
        return get_service(request).validation_metrics()

    @app.get("/threshold")
    def get_threshold(request: Request):
        return {"threshold": get_service(request).threshold}

    @app.post("/threshold")
    def set_threshold(body: ThresholdRequest, request: Request):
        service = get_service(request)
        previous = service.set_threshold(body.threshold)
        log_event("threshold changed", previous=previous, new=body.threshold)
        return {"previous_threshold": previous,
                "metrics": service.validation_metrics()}

    return app


app = create_app()
