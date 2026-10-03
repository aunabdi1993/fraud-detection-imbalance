"""
export_model.py — Package a trained model for serving.

Maps to: Chapter 4 (System Architecture), design points 1 and 2.

The API must score with exactly the objects the model was trained with. This
script fits the chosen model on the TRAINING split, chooses the decision
threshold on the VALIDATION split (rule 3), and writes one joblib bundle
holding:

  model, scaler   the fitted objects — never re-fitted at serve time, because
                  re-fitting silently shifts the feature distribution
                  (train/serve skew)
  threshold       a deployment parameter, adjustable without retraining
  val_y, val_p    validation labels and scores, so /metrics and /threshold can
                  recompute metrics on the true 1:~580 distribution. The test
                  split is deliberately NOT stored: tuning against it would
                  break rule 3.

Usage:
    python -m src.export_model --model xgboost
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib

from . import config
from .baseline_models import get_baseline_models
from .data_loader import FraudDataset
from .evaluation import choose_threshold

logger = logging.getLogger(__name__)


def build_bundle(splits, manifest: dict, scaler, model_name: str) -> dict:
    """Fit ``model_name`` on train, pick the threshold on validation."""
    models = get_baseline_models()
    if model_name not in models or model_name.startswith("dummy"):
        raise ValueError(
            f"model must be one of {[m for m in models if not m.startswith('dummy')]}"
        )
    model = models[model_name]
    model.fit(splits.X_train, splits.y_train)

    p_val = model.predict_proba(splits.X_val)[:, 1]
    threshold = choose_threshold(splits.y_val, p_val, objective="f1")
    cfg = manifest["config"]

    return {
        "model": model,
        "scaler": scaler,
        "threshold": float(threshold),
        "feature_names": list(manifest["feature_names"]),
        "log_amount": bool(cfg["log_amount"]),
        "val_y": splits.y_val.astype(int),
        "val_p": p_val,
        "model_id": f"{model_name}-baseline",
        "version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "data_config": cfg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(config.DATA_PROCESSED))
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--out", default=str(config.MODEL_BUNDLE_PATH))
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
    )

    splits, manifest = FraudDataset.load_splits(args.data)
    scaler = joblib.load(Path(args.data) / "scaler.joblib")
    bundle = build_bundle(splits, manifest, scaler, args.model)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    logger.info(
        "Wrote %s (model=%s, threshold=%.4f)", out, bundle["model_id"],
        bundle["threshold"],
    )


if __name__ == "__main__":
    main()
