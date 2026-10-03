"""
Tests for api/main.py.

Synthetic data only — these run without the CSV or a real trained model.
"""

from __future__ import annotations

import io
import time

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from api.main import create_app
from src import config


def _raw(rng, n, fraud=False):
    X = rng.normal(size=(n, len(config.RAW_FEATURES)))
    X[:, -1] = rng.uniform(0, 500, size=n)  # raw Amount
    if fraud:
        X[:, :5] += 3.0
    return X


@pytest.fixture(scope="module")
def bundle_path(tmp_path_factory):
    rng = np.random.default_rng(0)
    X = np.vstack([_raw(rng, 1500), _raw(rng, 60, fraud=True)])
    y = np.r_[np.zeros(1500), np.ones(60)].astype(int)
    Xp = X.copy()
    Xp[:, -1] = np.log1p(Xp[:, -1])  # mirrors data_loader._engineer
    scaler = StandardScaler().fit(Xp)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(Xp), y)
    p = model.predict_proba(scaler.transform(Xp))[:, 1]
    path = tmp_path_factory.mktemp("bundle") / "bundle.joblib"
    joblib.dump(
        {
            "model": model, "scaler": scaler, "threshold": 0.5,
            "feature_names": config.RAW_FEATURES, "log_amount": True,
            "val_y": y, "val_p": p, "model_id": "test-model", "version": "t",
        },
        path,
    )
    return path


@pytest.fixture
def client(bundle_path):
    with TestClient(create_app(bundle_path)) as c:
        yield c


def _tx(rng, fraud=False):
    return dict(zip(config.RAW_FEATURES, _raw(rng, 1, fraud)[0].tolist()))


def _csv(rows, columns=config.RAW_FEATURES):
    lines = [",".join(columns)] + [",".join(str(v) for v in r) for r in rows]
    return io.BytesIO("\n".join(lines).encode())


def test_health_reports_model_and_latency(client):
    client.post("/predict", json=_tx(np.random.default_rng(1)))
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_id"] == "test-model"
    assert body["inference_latency"]["n"] >= 1
    assert body["inference_latency"]["p99_ms"] is not None


def test_predict_returns_probability_and_decision(client):
    rng = np.random.default_rng(2)
    r = client.post("/predict", json=_tx(rng, fraud=True))
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["is_fraud"] == (body["fraud_probability"] >= body["threshold"])
    assert r.headers["x-request-id"]


def test_predict_separates_fraud_from_legit(client):
    rng = np.random.default_rng(3)
    p_fraud = client.post("/predict", json=_tx(rng, True)).json()["fraud_probability"]
    p_legit = client.post("/predict", json=_tx(rng, False)).json()["fraud_probability"]
    assert p_fraud > p_legit


def test_predict_rejects_missing_feature(client):
    tx = _tx(np.random.default_rng(4))
    del tx["V7"]
    assert client.post("/predict", json=tx).status_code == 422


def test_predict_rejects_extra_and_non_numeric_fields(client):
    tx = _tx(np.random.default_rng(5))
    assert client.post("/predict", json={**tx, "Class": 1}).status_code == 422
    assert client.post("/predict", json={**tx, "V1": "abc"}).status_code == 422


def test_predict_rejects_negative_amount(client):
    tx = _tx(np.random.default_rng(6))
    tx["Amount"] = -1.0
    assert client.post("/predict", json=tx).status_code == 422


def test_batch_predict_matches_single_predict(client):
    rng = np.random.default_rng(7)
    rows = _raw(rng, 5)
    r = client.post("/batch-predict", files={"file": ("t.csv", _csv(rows), "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["n_rows"] == 5 and len(body["predictions"]) == 5
    single = client.post(
        "/predict", json=dict(zip(config.RAW_FEATURES, rows[0].tolist()))
    ).json()["fraud_probability"]
    assert body["predictions"][0]["fraud_probability"] == pytest.approx(single)


def test_batch_predict_ignores_extra_columns(client):
    rng = np.random.default_rng(8)
    rows = [list(r) + [0, 0] for r in _raw(rng, 3)]
    cols = config.RAW_FEATURES + ["Time", "Class"]
    r = client.post("/batch-predict", files={"file": ("t.csv", _csv(rows, cols))})
    assert r.status_code == 200 and r.json()["n_rows"] == 3


def test_batch_predict_rejects_bad_uploads(client, monkeypatch):
    rng = np.random.default_rng(9)
    rows = _raw(rng, 2)
    missing = client.post(
        "/batch-predict", files={"file": ("t.csv", _csv(rows[:, :-1], config.RAW_FEATURES[:-1]))}
    )
    assert missing.status_code == 422
    nan_rows = rows.copy()
    nan_rows[1, 3] = np.nan
    assert client.post(
        "/batch-predict", files={"file": ("t.csv", _csv(nan_rows))}
    ).status_code == 422
    empty = client.post("/batch-predict", files={"file": ("t.csv", _csv([]))})
    assert empty.status_code == 400
    monkeypatch.setattr(config, "API_MAX_BATCH_ROWS", 1)
    assert client.post(
        "/batch-predict", files={"file": ("t.csv", _csv(rows))}
    ).status_code == 413


def test_metrics_are_validation_metrics(client):
    body = client.get("/metrics").json()
    for key in ("auc_pr", "precision", "recall", "f1"):
        assert 0.0 <= body[key] <= 1.0
    assert body["split"] == "validation"
    assert body["tp"] + body["fn"] == body["n_fraud"]


def test_threshold_update_changes_decisions_and_metrics(client):
    before = client.get("/metrics").json()
    r = client.post("/threshold", json={"threshold": 0.99})
    assert r.status_code == 200 and r.json()["previous_threshold"] == 0.5
    assert client.get("/threshold").json()["threshold"] == 0.99
    after = client.get("/metrics").json()
    assert after["threshold"] == 0.99
    assert after["n_alerts"] <= before["n_alerts"]
    assert after["auc_pr"] == before["auc_pr"]  # threshold-free metric unchanged
    tx = _tx(np.random.default_rng(10))
    assert client.post("/predict", json=tx).json()["threshold"] == 0.99


@pytest.mark.parametrize("bad", [-0.1, 1.5, "x"])
def test_threshold_rejects_out_of_range(client, bad):
    assert client.post("/threshold", json={"threshold": bad}).status_code == 422


def test_missing_model_degrades_gracefully(tmp_path):
    with TestClient(create_app(tmp_path / "nope.joblib")) as c:
        health = c.get("/health")
        assert health.status_code == 503 and health.json()["status"] == "degraded"
        assert c.post("/predict", json=_tx(np.random.default_rng(11))).status_code == 503
        assert c.get("/metrics").status_code == 503


def test_corrupt_bundle_degrades_gracefully(tmp_path):
    bad = tmp_path / "bad.joblib"
    joblib.dump({"model": None}, bad)
    with TestClient(create_app(bad)) as c:
        assert c.get("/health").json()["model_loaded"] is False


def test_request_timeout_returns_504(bundle_path, monkeypatch):
    from api import main

    original = main.ModelService.score

    def slow(self, frame):
        time.sleep(0.5)
        return original(self, frame)

    monkeypatch.setattr(main.ModelService, "score", slow)
    app = create_app(bundle_path, timeout_s=0.1)
    with TestClient(app) as c:
        # warm-up in lifespan already ran with the slow patch; that is fine
        r = c.post("/predict", json=_tx(np.random.default_rng(12)))
        assert r.status_code == 504
