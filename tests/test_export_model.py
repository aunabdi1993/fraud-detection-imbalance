"""Tests for export_model — synthetic splits, no CSV needed."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src import config
from src.data_loader import Splits
from src.export_model import build_bundle


@pytest.fixture
def splits_and_scaler():
    rng = np.random.default_rng(0)

    def make(n):
        y = np.zeros(n, dtype=int)
        y[: n // 50] = 1
        X = rng.normal(size=(n, len(config.RAW_FEATURES)))
        X[y == 1] += 2.0
        return X, y

    Xtr, ytr = make(2000)
    Xva, yva = make(600)
    scaler = StandardScaler().fit(Xtr)
    s = Splits(scaler.transform(Xtr), ytr, scaler.transform(Xva), yva,
               Xva, yva, list(config.RAW_FEATURES))
    manifest = {"config": {"log_amount": True}, "feature_names": s.feature_names}
    return s, manifest, scaler


def test_bundle_has_validation_scores_and_no_test_data(splits_and_scaler):
    s, manifest, scaler = splits_and_scaler
    b = build_bundle(s, manifest, scaler, "logistic_regression")
    assert len(b["val_p"]) == len(s.y_val) and 0 < b["threshold"] < 1
    assert b["scaler"] is scaler  # the fitted object, never re-fitted
    assert not any(k.endswith("test") or "test" in k for k in b)


def test_dummy_model_is_not_exportable(splits_and_scaler):
    s, manifest, scaler = splits_and_scaler
    with pytest.raises(ValueError):
        build_bundle(s, manifest, scaler, "dummy_most_frequent")
