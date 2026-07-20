"""Reproduce A/B research results with purged date-aware evaluation.

This script intentionally leaves legacy artefacts untouched. It writes a
separate results JSON and prediction parquet so the rerun can be audited.
Analyst features and the legacy stacking model are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.config import (
    FEATURES_CV_PATH,
    FEATURES_MARKET_PATH,
    FEATURES_NLP_PATH,
    PROCESSED_DIR,
    TARGET_CLASSES,
    TARGET_HORIZON_DAYS,
    TEST_END,
    TEST_START,
)
from src.models.splits import PurgedDateTimeSeriesSplit
from src.models.train_ml import (
    _get_feature_cols,
    _temporal_split,
    evaluate_model,
    load_combined_features,
    train_lightgbm,
    train_random_forest,
)

RESULTS_PATH = PROCESSED_DIR / "rerun_purged_ab_results.json"
PREDICTIONS_PATH = PROCESSED_DIR / "rerun_purged_ab_predictions.parquet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=TARGET_CLASSES, zero_division=0
    )
    return {
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 6),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "per_class": {
            label: {
                "precision": round(float(precision[i]), 6),
                "recall": round(float(recall[i]), 6),
                "f1": round(float(f1[i]), 6),
                "support": int(support[i]),
            }
            for i, label in enumerate(TARGET_CLASSES)
        },
        "confusion_matrix_labels": TARGET_CLASSES,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=TARGET_CLASSES).tolist(),
    }


def _test_frame(df: pd.DataFrame) -> pd.DataFrame:
    cutoff = pd.Timestamp(TEST_END) - pd.offsets.BDay(TARGET_HORIZON_DAYS)
    return df[(df.index >= TEST_START) & (df.index <= cutoff)].copy()


def _run_config(config: str, trials: int) -> tuple[dict, pd.DataFrame]:
    df = load_combined_features(config)
    if "sector" in df.columns:
        df = pd.get_dummies(df, columns=["sector"], prefix="sector", drop_first=False)
    feature_cols = _get_feature_cols(df)
    assert not any(col.startswith("analyst_") or col == "price_target_upside" for col in feature_cols)

    x_train, y_train, x_val, y_val, x_test, y_test = _temporal_split(df, feature_cols)
    splitter = PurgedDateTimeSeriesSplit(n_splits=5, embargo_days=TARGET_HORIZON_DAYS)
    candidates = {}

    rf_model, rf_cv = train_random_forest(x_train, y_train, splitter)
    candidates["RandomForest"] = {
        "model": rf_model,
        "cv": rf_cv,
        "validation": evaluate_model(rf_model, x_val, y_val, prefix="val"),
    }
    lgb_model, lgb_cv = train_lightgbm(x_train, y_train, splitter, n_trials=trials)
    lgb_cv.pop("best_params", None)
    candidates["LightGBM"] = {
        "model": lgb_model,
        "cv": lgb_cv,
        "validation": evaluate_model(lgb_model, x_val, y_val, prefix="val"),
    }

    selected_name = max(candidates, key=lambda name: candidates[name]["validation"]["val_f1_macro"])
    selected = candidates[selected_name]
    y_pred = selected["model"].predict(x_test)
    proba = selected["model"].predict_proba(x_test)

    majority = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    stratified = DummyClassifier(strategy="stratified", random_state=42).fit(x_train, y_train)
    momentum_pred = np.where(x_test["return_5d"].to_numpy() > 0, "UP", "DOWN")
    test_rows = _test_frame(df)
    if not x_test.index.equals(test_rows.index):
        raise RuntimeError("Reporting rows no longer align with the feature matrix")
    predictions = pd.DataFrame(
        {
            "date": x_test.index,
            "ticker": test_rows["ticker"].to_numpy(),
            "config": config,
            "actual": y_test.to_numpy(),
            "predicted": y_pred,
            "confidence": np.max(proba, axis=1),
        }
    )
    per_model = {
        name: {**item["cv"], **item["validation"]}
        for name, item in candidates.items()
    }
    return {
        "config": config,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "sample_sizes": {"train": len(x_train), "validation": len(x_val), "reporting": len(x_test)},
        "class_balance_reporting": y_test.value_counts(normalize=True).round(6).to_dict(),
        "selected_model": selected_name,
        "selection_metric": "validation_macro_f1",
        "selected_model_reporting_metrics": _metrics(y_test, y_pred),
        "benchmarks_reporting": {
            "majority_class": _metrics(y_test, majority.predict(x_test)),
            "stratified_random_seed_42": _metrics(y_test, stratified.predict(x_test)),
            "five_day_momentum": _metrics(y_test, momentum_pred),
        },
        "per_model_train_cv_and_validation": per_model,
    }, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run audited market-only and market+NLP reruns")
    parser.add_argument("--trials", type=int, default=40, help="Optuna trials per LightGBM configuration")
    args = parser.parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be positive")

    all_results, predictions = {}, []
    for config in ("A", "B"):
        result, config_predictions = _run_config(config, args.trials)
        all_results[config] = result
        predictions.append(config_predictions)

    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        revision = "unknown"
    all_results["run_metadata"] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_revision": revision,
        "target": "5-trading-day forward direction",
        "reporting_window": f"{TEST_START} through {TEST_END}, purged by {TARGET_HORIZON_DAYS} business days",
        "cv": "5-fold expanding date-grouped split with 5-business-day embargo",
        "excluded": ["analyst features", "stacking model", "legacy artefacts"],
        "input_sha256": {
            path.name: _sha256(path)
            for path in (FEATURES_MARKET_PATH, FEATURES_NLP_PATH, FEATURES_CV_PATH)
        },
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    pd.concat(predictions, ignore_index=True).to_parquet(PREDICTIONS_PATH, index=False)
    print(f"Saved results to {RESULTS_PATH}")
    print(f"Saved prediction rows to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
