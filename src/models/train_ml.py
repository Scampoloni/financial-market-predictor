"""
train_ml.py — Train all ML models and run the full ablation study.

Ablation configurations:
  Config A: market features only           (baseline)
  Config B: market + NLP features
  Config C: market + NLP + CV features     (full model)

Each config trains RandomForest (best Config A model) using TimeSeriesSplit CV,
then evaluates on the held-out test set (2025). Results are saved to
data/processed/ablation_results.json and the best Config C model to
models/stacking_final.pkl.

Usage:
    python -m src.models.train_ml          # full ablation
    python -m src.models.train_ml --config A   # single config
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb

from src.config import (
    CV_FOLDS,
    FEATURES_COMBINED_PATH,
    FEATURES_CV_PATH,
    FEATURES_MARKET_PATH,
    FEATURES_NLP_PATH,
    MODELS_DIR,
    PROCESSED_DIR,
    STACKING_MODEL_PATH,
    TARGET_CLASSES,
    TEST_START,
    TRAIN_END,
    VAL_END,
    VAL_START,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

ABLATION_RESULTS_PATH = PROCESSED_DIR / "ablation_results.json"

# Columns to always exclude from features
_EXCLUDE = {"ticker", "target", "close", "vix_regime", "rsi_zone",
            "vader_label", "finbert_label", "chart_available"}


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in _EXCLUDE and not c.startswith("Unnamed")]


def load_combined_features(config: str = "C") -> pd.DataFrame:
    """Load and join feature blocks for the requested config.

    Joins are performed on (date, ticker) to avoid cross-product explosion.

    Args:
        config: 'A' (market only), 'B' (market + NLP), 'C' (market + NLP + CV).

    Returns:
        Combined DataFrame with DatetimeIndex, sorted by date.
    """
    logger.info("Loading market features ...")
    market = pd.read_parquet(FEATURES_MARKET_PATH)
    market.index = pd.to_datetime(market.index)
    market.index.name = "date"

    if config == "A":
        return market.sort_index()

    # Use (date, ticker) multi-index for correct 1-to-1 joins
    market_mi = market.set_index("ticker", append=True)

    logger.info("Loading NLP features ...")
    nlp = pd.read_parquet(FEATURES_NLP_PATH)
    nlp.index = pd.to_datetime(nlp.index)
    nlp.index.name = "date"
    nlp_cols = [c for c in nlp.columns if c != "ticker"]
    nlp_mi = nlp.set_index("ticker", append=True)[nlp_cols]

    combined_mi = market_mi.join(nlp_mi, how="left")
    combined_mi[nlp_cols] = combined_mi[nlp_cols].fillna(0)

    if config == "B":
        return combined_mi.reset_index("ticker").sort_index()

    logger.info("Loading CV features ...")
    cv = pd.read_parquet(FEATURES_CV_PATH)
    cv.index = pd.to_datetime(cv.index)
    cv.index.name = "date"
    cv_cols = [c for c in cv.columns if c not in {"ticker", "chart_available"}]
    cv_mi = cv.set_index("ticker", append=True)[cv_cols]

    combined_mi = combined_mi.join(cv_mi, how="left")
    combined_mi[cv_cols] = combined_mi[cv_cols].fillna(0)

    return combined_mi.reset_index("ticker").sort_index()


def _temporal_split(df: pd.DataFrame, feature_cols: list[str]):
    """Split into train/val/test preserving temporal order.

    Returns (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    train = df[df.index <= TRAIN_END]
    val   = df[(df.index >= VAL_START) & (df.index <= VAL_END)]
    test  = df[df.index >= TEST_START]

    def xy(d):
        return d[feature_cols].fillna(0), d["target"]

    return (*xy(train), *xy(val), *xy(test))


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tscv: TimeSeriesSplit,
) -> tuple[RandomForestClassifier, dict]:
    """Train RandomForest with TimeSeriesSplit CV.

    Returns (fitted model, cv metrics dict).
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    scores = cross_validate(
        model, X_train, y_train,
        cv=tscv,
        scoring=["f1_macro", "accuracy"],
        return_train_score=False,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    return model, {
        "cv_f1_mean": float(scores["test_f1_macro"].mean()),
        "cv_f1_std":  float(scores["test_f1_macro"].std()),
        "cv_acc_mean": float(scores["test_accuracy"].mean()),
        "fold_f1": scores["test_f1_macro"].tolist(),
    }


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Return test-set metrics dict."""
    y_pred = model.predict(X_test)
    f1  = float(f1_score(y_test, y_pred, average="macro"))
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, labels=TARGET_CLASSES, output_dict=True)
    return {
        "test_f1_macro": f1,
        "test_accuracy": acc,
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"], 4),
                "f1":        round(report[cls]["f1-score"], 4),
            }
            for cls in TARGET_CLASSES
        },
    }


def run_ablation(configs: list[str] = ("A", "B", "C")) -> dict:
    """Run the full ablation study across specified configs.

    Args:
        configs: List of config names to evaluate.

    Returns:
        Dict with results per config.
    """
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    results = {}

    for config in configs:
        logger.info("=" * 55)
        logger.info("Config %s: loading features ...", config)
        df = load_combined_features(config)

        # One-hot encode sector if present
        if "sector" in df.columns:
            df = pd.get_dummies(df, columns=["sector"], prefix="sector", drop_first=False)

        feature_cols = _get_feature_cols(df)
        logger.info("Config %s: %d features, %d rows", config, len(feature_cols), len(df))

        X_train, y_train, X_val, y_val, X_test, y_test = _temporal_split(df, feature_cols)

        logger.info("Config %s: training RandomForest (%d train rows) ...", config, len(X_train))
        model, cv_metrics = train_random_forest(X_train, y_train, tscv)

        logger.info("Config %s: evaluating on test set (%d rows) ...", config, len(X_test))
        test_metrics = evaluate_model(model, X_test, y_test)

        results[config] = {
            "n_features": len(feature_cols),
            "feature_cols": feature_cols,
            **cv_metrics,
            **test_metrics,
        }

        logger.info(
            "Config %s done — CV F1: %.4f ± %.4f | Test F1: %.4f | Test Acc: %.4f",
            config,
            cv_metrics["cv_f1_mean"], cv_metrics["cv_f1_std"],
            test_metrics["test_f1_macro"], test_metrics["test_accuracy"],
        )

        # Save best model (Config C)
        if config == "C":
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            with open(STACKING_MODEL_PATH, "wb") as f:
                pickle.dump({"model": model, "feature_cols": feature_cols}, f)
            logger.info("Config C model saved to %s", STACKING_MODEL_PATH)

    # Save results
    ABLATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ABLATION_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Ablation results saved to %s", ABLATION_RESULTS_PATH)

    return results


def print_ablation_table(results: dict) -> None:
    """Print a formatted ablation table."""
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)
    print(f"{'Config':<10} {'Features':<10} {'CV F1':<18} {'Test F1':<10} {'Test Acc':<10}")
    print("-" * 70)

    config_a_f1 = results.get("A", {}).get("test_f1_macro", None)
    for config, r in results.items():
        delta = ""
        if config_a_f1 and config != "A":
            d = r["test_f1_macro"] - config_a_f1
            delta = f"  ({d:+.4f})"
        cv_str = f"{r['cv_f1_mean']:.4f} ± {r['cv_f1_std']:.4f}"
        print(
            f"Config {config:<4} {r['n_features']:<10} {cv_str:<18} "
            f"{r['test_f1_macro']:.4f}{delta:<12} {r['test_accuracy']:.4f}"
        )
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--config", choices=["A", "B", "C"], help="Run single config")
    args = parser.parse_args()

    configs = [args.config] if args.config else ["A", "B", "C"]
    results = run_ablation(configs)
    print_ablation_table(results)
