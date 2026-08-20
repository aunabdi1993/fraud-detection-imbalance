"""
Tests for data_loader.

The tests that matter here are the leakage guards. A subtle leak does not
raise an exception — it just makes every number in Chapter 5 wrong in a
direction that looks like success. These tests are the tripwire.

Run: pytest tests/ -v
Tests needing the real CSV skip automatically if it is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data_loader import DataConfig, DataValidationError, FraudDataset

CSV = "data/raw/creditcard.csv"


def _have_csv() -> bool:
    from pathlib import Path
    return Path(CSV).exists()


requires_csv = pytest.mark.skipif(
    not _have_csv(), reason="creditcard.csv not present; run scripts/get_data.sh"
)


# --- Config ----------------------------------------------------------------
def test_config_rejects_bad_strategy():
    with pytest.raises(ValueError):
        DataConfig(split_strategy="random")


def test_config_rejects_bad_sizes():
    with pytest.raises(ValueError):
        DataConfig(test_size=1.5)


# --- Validation ------------------------------------------------------------
@requires_csv
def test_validation_matches_published_benchmarks():
    r = FraudDataset(DataConfig(csv_path=CSV)).validate()
    assert r["validation_passed"]
    assert r["n_rows"] == 284_807
    assert r["n_fraud"] == 492
    assert r["imbalance_ratio"] == pytest.approx(577.9, abs=0.1)


@requires_csv
def test_validation_detects_duplicates():
    r = FraudDataset(DataConfig(csv_path=CSV)).validate()
    assert r["n_duplicate_rows"] == 1081
    assert r["n_duplicate_fraud_rows"] == 19


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        FraudDataset(DataConfig(csv_path="data/raw/nope.csv")).load()


# --- Splitting -------------------------------------------------------------
@requires_csv
def test_splits_are_disjoint():
    """No record may appear in more than one split."""
    ds = FraudDataset(DataConfig(csv_path=CSV))
    s = ds.prepare()
    total = len(s.y_train) + len(s.y_val) + len(s.y_test)
    assert total == ds.report["n_rows_after_dedup"]


@requires_csv
def test_stratification_preserves_fraud_rate():
    """Test split must reflect deployment prevalence, within tolerance."""
    s = FraudDataset(DataConfig(csv_path=CSV)).prepare()
    rates = [y.mean() for y in (s.y_train, s.y_val, s.y_test)]
    assert max(rates) - min(rates) < 1e-4, f"stratification drifted: {rates}"


@requires_csv
def test_every_split_contains_fraud():
    s = FraudDataset(DataConfig(csv_path=CSV)).prepare()
    for name in ("train", "val", "test"):
        assert getattr(s, f"y_{name}").sum() > 0


@requires_csv
def test_split_is_deterministic():
    """Same seed must give byte-identical splits, or results are unreproducible."""
    a = FraudDataset(DataConfig(csv_path=CSV, random_state=42)).prepare()
    b = FraudDataset(DataConfig(csv_path=CSV, random_state=42)).prepare()
    assert np.array_equal(a.y_test, b.y_test)
    assert np.allclose(a.X_test, b.X_test)


@requires_csv
def test_different_seed_gives_different_split():
    a = FraudDataset(DataConfig(csv_path=CSV, random_state=42)).prepare()
    b = FraudDataset(DataConfig(csv_path=CSV, random_state=7)).prepare()
    assert not np.array_equal(a.y_test, b.y_test)


# --- Leakage guards --------------------------------------------------------
@requires_csv
def test_scaler_fitted_on_training_only():
    """LEAKAGE GUARD. If the scaler saw the full dataset, the test split would
    also be centred at ~0. Training must be centred; test must not be exactly."""
    s = FraudDataset(DataConfig(csv_path=CSV)).prepare()
    assert abs(s.X_train.mean()) < 1e-10, "training data should be centred"
    assert abs(s.X_test.mean()) > 1e-12, "test mean is suspiciously exact"


@requires_csv
def test_no_duplicate_rows_across_train_and_test():
    """LEAKAGE GUARD. Identical records in both splits inflate every metric."""
    s = FraudDataset(DataConfig(csv_path=CSV, dedup_scope="features")).prepare()
    tr = {r.tobytes() for r in np.round(s.X_train, 8)}
    te = {r.tobytes() for r in np.round(s.X_test, 8)}
    assert not (tr & te), "identical rows found in both train and test"


@requires_csv
def test_cv_folds_are_stratified_and_cover_training_set():
    ds = FraudDataset(DataConfig(csv_path=CSV))
    ds.prepare()
    seen = set()
    for tr, va in ds.cv_splits():
        assert ds.splits.y_train[va].sum() > 0, "fold has no fraud cases"
        assert not (set(tr) & set(va)), "fold train/val overlap"
        seen |= set(va)
    assert len(seen) == len(ds.splits.y_train), "folds do not cover training set"


# --- Round trip ------------------------------------------------------------
@requires_csv
def test_save_and_reload_round_trip(tmp_path):
    ds = FraudDataset(DataConfig(csv_path=CSV))
    ds.prepare()
    ds.save(tmp_path)
    reloaded, manifest = FraudDataset.load_splits(tmp_path)
    assert np.allclose(reloaded.X_test, ds.splits.X_test)
    assert manifest["config"]["random_state"] == 42
    assert "file_sha256" in manifest["validation_report"]


@requires_csv
def test_temporal_split_requires_time_column():
    cfg = DataConfig(csv_path=CSV, split_strategy="temporal", drop_time=True)
    with pytest.raises(ValueError, match="Time"):
        FraudDataset(cfg).prepare()
