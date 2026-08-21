"""
data_loader.py — Dataset loading, validation, splitting and scaling.

Dissertation: "How do different class imbalance handling techniques compare in
performance and practical deployability for financial fraud detection?"

Maps to: Chapter 3 §3.1 (Dataset), §3.2 (Split), §3.3 (Normalisation).

Design principles (from literature review §5.3):
  1. Every split is STRATIFIED — the imbalance ratio must be preserved in
     train/val/test, otherwise the test set no longer reflects deployment.
  2. The scaler is FIT ON TRAINING DATA ONLY, then applied to val/test.
     Fitting on the full dataset leaks test-set distribution into training.
  3. NO RESAMPLING HAPPENS HERE. Resampling belongs inside the CV fold,
     after the split (see src/imbalance_methods.py). This module deliberately
     has no SMOTE import — that separation is the guard against the most
     common leakage error in the fraud-detection literature.
  4. Every run is reproducible: fixed seed, and a manifest written to disk
     recording exactly what was done.

Usage:
    from data_loader import FraudDataset, DataConfig

    ds = FraudDataset(DataConfig(csv_path="data/raw/creditcard.csv"))
    report = ds.validate()
    splits = ds.prepare()          # load -> dedup -> split -> scale
    ds.save("data/processed")

CLI:
    python src/data_loader.py --csv data/raw/creditcard.csv --out data/processed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Published benchmarks for the ULB European credit-card dataset.
# validate() asserts the loaded file matches these, so a corrupted or
# re-sampled copy of the CSV cannot silently invalidate every downstream result.
# ---------------------------------------------------------------------------
EXPECTED_N_ROWS = 284_807
EXPECTED_N_FRAUD = 492
EXPECTED_FRAUD_RATE = 0.001727
EXPECTED_COLUMNS = (
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
)


@dataclass
class DataConfig:
    """All experimental choices in one auditable object."""

    csv_path: str = "data/raw/creditcard.csv"

    # --- Split -------------------------------------------------------------
    test_size: float = 0.15
    val_size: float = 0.15          # taken from the remaining 85%
    random_state: int = 42
    split_strategy: str = "stratified"   # "stratified" | "temporal"

    # --- Duplicate handling ------------------------------------------------
    # Duplicates must be defined over the feature space the MODEL actually
    # sees, not over the raw file:
    #   "exact"    1,081 rows are duplicates across all 31 raw columns.
    #   "features" 9,144 rows are duplicates once Time is excluded — and Time
    #              is dropped as a feature, so these rows are indistinguishable
    #              to the classifier. 32 fraud rows are affected. Labels never
    #              conflict within a duplicate group, so this is redundancy,
    #              not label noise.
    #   "none"     retain everything (the literature convention, N=284,807).
    # Default "features": it is the only scope that actually prevents identical
    # records appearing in both train and test. See Chapter 3 §3.1.
    dedup_scope: str = "features"

    # --- Feature treatment -------------------------------------------------
    drop_time: bool = True
    log_amount: bool = True
    scale_features: bool = True

    # --- Cross-validation --------------------------------------------------
    n_folds: int = 5

    # --- Validation strictness --------------------------------------------
    strict_validation: bool = True  # raise on benchmark mismatch vs. warn

    def __post_init__(self) -> None:
        if self.split_strategy not in {"stratified", "temporal"}:
            raise ValueError(
                f"split_strategy must be 'stratified' or 'temporal', "
                f"got {self.split_strategy!r}"
            )
        if not 0 < self.test_size < 1 or not 0 < self.val_size < 1:
            raise ValueError("test_size and val_size must be in (0, 1)")
        if self.dedup_scope not in {"none", "exact", "features"}:
            raise ValueError(
                f"dedup_scope must be 'none', 'exact' or 'features', "
                f"got {self.dedup_scope!r}"
            )


@dataclass
class Splits:
    """Container for the prepared arrays."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        rows = []
        for name in ("train", "val", "test"):
            y = getattr(self, f"y_{name}")
            n, n_fraud = len(y), int(y.sum())
            rows.append(
                {
                    "Split": name,
                    "N": n,
                    "Fraud": n_fraud,
                    "Fraud rate (%)": round(100 * n_fraud / n, 4),
                    "Imbalance ratio": f"1:{round((n - n_fraud) / max(n_fraud, 1), 1)}",
                }
            )
        return pd.DataFrame(rows)


class DataValidationError(RuntimeError):
    """Raised when the loaded file does not match published benchmarks."""


class FraudDataset:
    """Loads, validates, splits and scales the ULB credit-card fraud dataset."""

    def __init__(self, config: DataConfig | None = None) -> None:
        self.config = config or DataConfig()
        self.df: pd.DataFrame | None = None
        self.splits: Splits | None = None
        self.scaler: StandardScaler | None = None
        self.report: dict = {}

    # -- 1. Load ------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        path = Path(self.config.csv_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {path}. Download the ULB European "
                f"credit-card dataset from Kaggle and unzip it there."
            )
        logger.info("Loading %s", path)
        self.df = pd.read_csv(path)
        self.report["file_sha256"] = self._hash_file(path)
        self.report["file_size_bytes"] = path.stat().st_size
        return self.df

    @staticmethod
    def _hash_file(path: Path, chunk: int = 1 << 20) -> str:
        """Hash the raw file so results are traceable to an exact copy."""
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(chunk), b""):
                h.update(block)
        return h.hexdigest()

    # -- 2. Validate --------------------------------------------------------
    def validate(self) -> dict:
        """Assert the file matches published benchmarks. Returns a report dict
        whose contents populate Chapter 3, Table 1."""
        if self.df is None:
            self.load()
        df = self.df
        problems: list[str] = []

        if list(df.columns) != EXPECTED_COLUMNS:
            problems.append(
                f"Column mismatch: expected {len(EXPECTED_COLUMNS)} columns "
                f"in canonical order, got {list(df.columns)}"
            )
        if len(df) != EXPECTED_N_ROWS:
            problems.append(f"Expected {EXPECTED_N_ROWS:,} rows, got {len(df):,}")

        n_fraud = int(df["Class"].sum())
        if n_fraud != EXPECTED_N_FRAUD:
            problems.append(f"Expected {EXPECTED_N_FRAUD} frauds, got {n_fraud}")

        fraud_rate = n_fraud / len(df)
        if abs(fraud_rate - EXPECTED_FRAUD_RATE) > 1e-5:
            problems.append(
                f"Fraud rate {fraud_rate:.5f} differs from published "
                f"{EXPECTED_FRAUD_RATE:.5f}"
            )

        n_null = int(df.isna().sum().sum())
        if n_null:
            problems.append(f"{n_null} missing values (expected 0)")

        n_dup = int(df.duplicated().sum())
        n_dup_fraud = int(df[df.duplicated()]["Class"].sum())

        self.report.update(
            {
                "n_rows": len(df),
                "n_features": df.shape[1] - 1,
                "n_fraud": n_fraud,
                "n_legitimate": len(df) - n_fraud,
                "fraud_rate_pct": round(100 * fraud_rate, 4),
                "imbalance_ratio": round((len(df) - n_fraud) / n_fraud, 1),
                "n_missing": n_null,
                "n_duplicate_rows": n_dup,
                "n_duplicate_fraud_rows": n_dup_fraud,
                "time_span_hours": round(df["Time"].max() / 3600, 2),
                "amount_mean": round(df["Amount"].mean(), 2),
                "amount_median": round(df["Amount"].median(), 2),
                "amount_max": round(df["Amount"].max(), 2),
                "amount_zero_count": int((df["Amount"] == 0).sum()),
                "amount_mean_fraud": round(
                    df.loc[df["Class"] == 1, "Amount"].mean(), 2
                ),
                "amount_mean_legitimate": round(
                    df.loc[df["Class"] == 0, "Amount"].mean(), 2
                ),
                "validation_problems": problems,
                "validation_passed": not problems,
            }
        )

        if problems:
            msg = "Dataset validation failed:\n  - " + "\n  - ".join(problems)
            if self.config.strict_validation:
                raise DataValidationError(msg)
            logger.warning(msg)
        else:
            logger.info(
                "Validation passed: %s rows, %s frauds (%.4f%%), IR 1:%s",
                f"{len(df):,}", n_fraud, 100 * fraud_rate,
                self.report["imbalance_ratio"],
            )
        if n_dup:
            logger.warning(
                "%d exact duplicate rows (%d fraud). dedup_scope=%s",
                n_dup, n_dup_fraud, self.config.dedup_scope,
            )
        return self.report

    # -- 3. Prepare ---------------------------------------------------------
    def prepare(self) -> Splits:
        """Full pipeline: load -> validate -> dedup -> features -> split -> scale."""
        if self.df is None:
            self.load()
        if not self.report.get("validation_passed"):
            self.validate()

        df = self.df.copy()

        # Order matters: "exact" dedups the raw frame, "features" dedups AFTER
        # Time is dropped, because that is the space the model sees.
        if self.config.dedup_scope == "exact":
            df = self._dedup(df, "exact")
        df = self._engineer(df)
        if self.config.dedup_scope == "features":
            df = self._dedup(df, "features")
        if self.config.dedup_scope == "none":
            self.report["n_rows_after_dedup"] = len(df)
            self.report["n_fraud_after_dedup"] = int(df["Class"].sum())

        y = df["Class"].to_numpy(dtype=np.int8)
        feature_names = [c for c in df.columns if c != "Class"]
        X = df[feature_names].to_numpy(dtype=np.float64)

        if self.config.split_strategy == "stratified":
            idx = self._stratified_indices(y)
        else:
            idx = self._temporal_indices(df)
        tr, va, te = idx

        self.splits = Splits(
            X_train=X[tr], y_train=y[tr],
            X_val=X[va], y_val=y[va],
            X_test=X[te], y_test=y[te],
            feature_names=feature_names,
        )

        if self.config.scale_features:
            self._scale()

        self.report["split_summary"] = self.splits.summary().to_dict("records")
        return self.splits

    def _dedup(self, df: pd.DataFrame, scope: str) -> pd.DataFrame:
        """Remove duplicate records and record the effect for Chapter 3."""
        before, before_fraud = len(df), int(df["Class"].sum())
        dup_mask = df.duplicated(keep="first")
        n_dup_fraud = int(df.loc[dup_mask, "Class"].sum())

        # A duplicate group carrying both labels would be genuine label noise
        # and must not be silently collapsed.
        feat_cols = [c for c in df.columns if c != "Class"]
        conflicting = int(
            (df[df.duplicated(subset=feat_cols, keep=False)]
             .groupby(feat_cols)["Class"].nunique() > 1).sum()
        ) if dup_mask.any() else 0
        if conflicting:
            logger.warning(
                "%d duplicate feature groups carry CONFLICTING labels — "
                "investigate before dropping", conflicting
            )

        df = df[~dup_mask].reset_index(drop=True)
        logger.info(
            "Dedup (scope=%s): dropped %d rows (%d fraud); %d -> %d",
            scope, before - len(df), n_dup_fraud, before, len(df),
        )
        self.report.update({
            "dedup_scope": scope,
            "n_rows_dropped_dedup": before - len(df),
            "n_fraud_dropped_dedup": before_fraud - int(df["Class"].sum()),
            "n_conflicting_label_groups": conflicting,
            "n_rows_after_dedup": len(df),
            "n_fraud_after_dedup": int(df["Class"].sum()),
        })
        return df

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Minimal, defensible transforms only. Real feature engineering lives
        in src/preprocessing.py so that it can be ablated separately."""
        if self.config.log_amount:
            # log1p handles the 1,825 zero-amount transactions without a shift.
            df["Amount"] = np.log1p(df["Amount"])
        if self.config.drop_time:
            df = df.drop(columns=["Time"])
        return df

    def _stratified_indices(
        self, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Two-stage stratified split -> train / val / test."""
        cfg = self.config
        all_idx = np.arange(len(y))
        trainval_idx, test_idx = train_test_split(
            all_idx,
            test_size=cfg.test_size,
            stratify=y,
            random_state=cfg.random_state,
            shuffle=True,
        )
        # val_size is expressed as a fraction of the FULL dataset, so rescale
        # it relative to the remaining train+val portion.
        val_fraction = cfg.val_size / (1.0 - cfg.test_size)
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=val_fraction,
            stratify=y[trainval_idx],
            random_state=cfg.random_state,
            shuffle=True,
        )
        return train_idx, val_idx, test_idx

    def _temporal_indices(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Chronological split — sensitivity check for concept drift (Gap 5).

        Requires drop_time=False so ordering information survives. Note the
        dataset spans only two days, so this is a weak drift test and must be
        reported as such in Chapter 6 §6.4.
        """
        if "Time" not in df.columns:
            raise ValueError(
                "temporal split requires Time; set DataConfig(drop_time=False)"
            )
        order = np.argsort(df["Time"].to_numpy(), kind="stable")
        n = len(order)
        n_test = int(round(n * self.config.test_size))
        n_val = int(round(n * self.config.val_size))
        n_train = n - n_val - n_test
        return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]

    def _scale(self) -> None:
        """Fit on train only — this is the leakage guard."""
        s = self.splits
        self.scaler = StandardScaler().fit(s.X_train)
        s.X_train = self.scaler.transform(s.X_train)
        s.X_val = self.scaler.transform(s.X_val)
        s.X_test = self.scaler.transform(s.X_test)
        logger.info("Scaled features (StandardScaler fit on training split only)")

    # -- 4. Cross-validation ------------------------------------------------
    def cv_splits(
        self, X: np.ndarray | None = None, y: np.ndarray | None = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Stratified k-fold indices over the TRAINING split.

        Resampling must be applied inside the loop, to the training fold only:

            for tr, va in ds.cv_splits():
                X_res, y_res = sampler.fit_resample(X_train[tr], y_train[tr])
                model.fit(X_res, y_res)
                score(model, X_train[va], y_train[va])   # never resampled
        """
        if X is None or y is None:
            if self.splits is None:
                raise RuntimeError("Call prepare() first.")
            X, y = self.splits.X_train, self.splits.y_train
        skf = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        yield from skf.split(X, y)

    # -- 5. Persist ---------------------------------------------------------
    def save(self, out_dir: str | Path) -> Path:
        """Persist arrays, scaler and a manifest so every result is traceable."""
        if self.splits is None:
            raise RuntimeError("Call prepare() first.")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        s = self.splits
        np.savez_compressed(
            out / "splits.npz",
            X_train=s.X_train, y_train=s.y_train,
            X_val=s.X_val, y_val=s.y_val,
            X_test=s.X_test, y_test=s.y_test,
        )
        if self.scaler is not None:
            import joblib
            joblib.dump(self.scaler, out / "scaler.joblib")

        manifest = {
            "config": asdict(self.config),
            "validation_report": self.report,
            "feature_names": s.feature_names,
            "n_features": len(s.feature_names),
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        logger.info("Saved splits and manifest to %s", out)
        return out

    @staticmethod
    def load_splits(out_dir: str | Path) -> tuple[Splits, dict]:
        """Reload a previously prepared dataset (avoids re-running the pipeline)."""
        out = Path(out_dir)
        z = np.load(out / "splits.npz")
        manifest = json.loads((out / "manifest.json").read_text())
        splits = Splits(
            X_train=z["X_train"], y_train=z["y_train"],
            X_val=z["X_val"], y_val=z["y_val"],
            X_test=z["X_test"], y_test=z["y_test"],
            feature_names=manifest["feature_names"],
        )
        return splits, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/raw/creditcard.csv")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split", choices=["stratified", "temporal"], default="stratified"
    )
    parser.add_argument(
        "--dedup", choices=["none", "exact", "features"], default="features",
        help="Duplicate-removal scope; 'features' ignores Time (default)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
    )

    cfg = DataConfig(
        csv_path=args.csv,
        random_state=args.seed,
        split_strategy=args.split,
        dedup_scope=args.dedup,
        drop_time=(args.split != "temporal"),
    )
    ds = FraudDataset(cfg)
    ds.validate()
    ds.prepare()
    ds.save(args.out)

    print("\n" + ds.splits.summary().to_string(index=False))


if __name__ == "__main__":
    main()
