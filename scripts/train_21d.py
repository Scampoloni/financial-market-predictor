"""Train a 21-day horizon model using the same pipeline as the 5-day ablation.

Features: Market + NLP + Analyst + CV (same 66-feature Config D set).
Target:   21-day forward return direction (UP / DOWN).
Models:   RandomForest, LightGBM (Optuna 40 trials).
Selection: best model by validation F1 (2024 H2), same protocol as train_ml.py.
Output:   models/model_21d.pkl  (dict with keys 'model', 'feature_cols')

Usage:
    python scripts/train_21d.py
"""

import logging
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import cross_validate, cross_val_score

from src.config import (
    CV_FOLDS,
    FEATURES_ANALYST_PATH,
    FEATURES_CV_PATH,
    FEATURES_MARKET_PATH,
    FEATURES_NLP_PATH,
    MODEL_21D_PATH,
    MODELS_DIR,
    RAW_MARKET_DIR,
    TARGET_HORIZON_DAYS_LONG,
    TEST_START,
    TEST_END,
    TRAIN_END,
    VAL_END,
    VAL_START,
)
from src.models.splits import PurgedDateTimeSeriesSplit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

HORIZON = TARGET_HORIZON_DAYS_LONG   # 21 trading days (~1 month), from config
_EXCLUDE = {"ticker", "target", "close", "vix_regime", "rsi_zone",
            "vader_label", "finbert_label", "chart_available"}


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def _recompute_target(market_df: pd.DataFrame) -> pd.DataFrame:
    """Replace 5-day target with 21-day forward return direction."""
    result = []
    for ticker in market_df["ticker"].unique():
        mask = market_df["ticker"] == ticker
        chunk = market_df.loc[mask].copy()

        csv_path = RAW_MARKET_DIR / f"{ticker}.csv"
        if not csv_path.exists():
            continue
        raw = pd.read_csv(csv_path, index_col="Date", parse_dates=True).sort_index()
        close = raw["Close"]

        fwd_return = close.shift(-HORIZON) / close - 1
        target = pd.Series("DOWN", index=close.index, name="target")
        target[fwd_return > 0] = "UP"
        target[fwd_return.isna()] = np.nan

        chunk["target"] = target.reindex(chunk.index)
        chunk = chunk.dropna(subset=["target"])
        result.append(chunk)

    return pd.concat(result)


def load_features() -> pd.DataFrame:
    """Load and join all feature blocks (Market + NLP + Analyst + CV)."""
    logger.info("Loading market features ...")
    market = pd.read_parquet(FEATURES_MARKET_PATH)
    market.index = pd.to_datetime(market.index)
    market.index.name = "date"

    logger.info("Recomputing target for %d-day horizon ...", HORIZON)
    market = _recompute_target(market)
    logger.info("  %d rows after target recompute", len(market))

    market_mi = market.set_index("ticker", append=True)

    # NLP
    logger.info("Loading NLP features ...")
    nlp = pd.read_parquet(FEATURES_NLP_PATH)
    nlp.index = pd.to_datetime(nlp.index)
    nlp.index.name = "date"
    nlp_cols = [c for c in nlp.columns if c != "ticker"]
    nlp_mi = nlp.set_index("ticker", append=True)[nlp_cols]
    combined_mi = market_mi.join(nlp_mi, how="left")
    combined_mi[nlp_cols] = combined_mi[nlp_cols].fillna(0)

    # Analyst
    if FEATURES_ANALYST_PATH.exists():
        logger.info("Loading analyst features ...")
        analyst = pd.read_parquet(FEATURES_ANALYST_PATH)
        analyst.index = pd.to_datetime(analyst.index)
        analyst.index.name = "date"
        analyst_cols = [c for c in analyst.columns if c != "ticker"]
        analyst_mi = analyst.set_index("ticker", append=True)[analyst_cols]
        combined_mi = combined_mi.join(analyst_mi, how="left")
        combined_mi[analyst_cols] = combined_mi[analyst_cols].fillna(0)
        logger.info("  Joined %d analyst feature columns", len(analyst_cols))
    else:
        logger.warning("features_analyst.parquet not found — skipping analyst features")

    # CV
    logger.info("Loading CV features ...")
    cv = pd.read_parquet(FEATURES_CV_PATH)
    cv.index = pd.to_datetime(cv.index)
    cv.index.name = "date"
    cv_cols = [c for c in cv.columns if c not in {"ticker", "chart_available"}]
    cv_mi = cv.set_index("ticker", append=True)[cv_cols]
    combined_mi = combined_mi.join(cv_mi, how="left")
    combined_mi[cv_cols] = combined_mi[cv_cols].fillna(0)

    df = combined_mi.reset_index("ticker").sort_index()

    # Sector dummies
    if "sector" in df.columns:
        df = pd.get_dummies(df, columns=["sector"], prefix="sector", drop_first=False)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Model trainers (same as train_ml.py)
# ──────────────────────────────────────────────────────────────────────────────

def _train_rf(X_train, y_train, tscv):
    model = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    scores = cross_validate(
        model, X_train, y_train, cv=tscv,
        scoring=["f1_macro", "accuracy"], return_train_score=False, n_jobs=1,
    )
    model.fit(X_train, y_train)
    return model, {
        "cv_f1_mean": float(scores["test_f1_macro"].mean()),
        "cv_f1_std":  float(scores["test_f1_macro"].std()),
        "fold_f1":    scores["test_f1_macro"].tolist(),
    }


def _optuna_lgb(X_train, y_train, tscv, n_trials=40):
    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 800),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        m = lgb.LGBMClassifier(**params, is_unbalance=True, random_state=42, verbose=-1)
        return cross_val_score(m, X_train, y_train, cv=tscv, scoring="f1_macro").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info("Optuna LGB best F1: %.4f  params: %s", study.best_value, study.best_params)
    return study.best_params


def _train_lgb(X_train, y_train, tscv, n_trials=40):
    logger.info("Optuna tuning LightGBM (%d trials) ...", n_trials)
    best_params = _optuna_lgb(X_train, y_train, tscv, n_trials=n_trials)
    model = lgb.LGBMClassifier(**best_params, is_unbalance=True, random_state=42, verbose=-1)
    scores = cross_validate(
        model, X_train, y_train, cv=tscv,
        scoring=["f1_macro", "accuracy"], return_train_score=False, n_jobs=1,
    )
    model.fit(X_train, y_train)
    return model, {
        "cv_f1_mean": float(scores["test_f1_macro"].mean()),
        "cv_f1_std":  float(scores["test_f1_macro"].std()),
        "fold_f1":    scores["test_f1_macro"].tolist(),
        "best_params": best_params,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("21-day horizon model training")
    logger.info("=" * 60)

    df = load_features()

    feature_cols = [c for c in df.columns
                    if c not in _EXCLUDE and not c.startswith("Unnamed")]

    horizon = pd.offsets.BDay(HORIZON)
    train = df[df.index <= pd.Timestamp(TRAIN_END) - horizon]
    val = df[(df.index >= VAL_START) & (df.index <= pd.Timestamp(VAL_END) - horizon)]
    test = df[(df.index >= TEST_START) & (df.index <= pd.Timestamp(TEST_END) - horizon)]

    X_train, y_train = train[feature_cols].fillna(0), train["target"]
    X_val,   y_val   = val[feature_cols].fillna(0),   val["target"]
    X_test,  y_test  = test[feature_cols].fillna(0),  test["target"]

    logger.info(
        "21d: %d features, %d train / %d val / %d test rows",
        len(feature_cols), len(X_train), len(X_val), len(X_test),
    )

    tscv = PurgedDateTimeSeriesSplit(n_splits=CV_FOLDS, embargo_days=HORIZON)
    results = {}

    # ── RandomForest ──────────────────────────────────────────────────────────
    logger.info("Training RandomForest ...")
    rf_model, rf_meta = _train_rf(X_train, y_train, tscv)
    rf_val_f1  = f1_score(y_val,  rf_model.predict(X_val),  average="macro")
    rf_test_f1 = f1_score(y_test, rf_model.predict(X_test), average="macro")
    rf_test_acc = accuracy_score(y_test, rf_model.predict(X_test))
    logger.info(
        "  RF  ▸ CV F1: %.4f | Val F1: %.4f | Test F1: %.4f",
        rf_meta["cv_f1_mean"], rf_val_f1, rf_test_f1,
    )
    results["RandomForest"] = {**rf_meta, "val_f1": rf_val_f1,
                                "test_f1": rf_test_f1, "test_acc": rf_test_acc,
                                "_model": rf_model}

    # ── LightGBM ──────────────────────────────────────────────────────────────
    logger.info("Training LightGBM (Optuna) ...")
    lgb_model, lgb_meta = _train_lgb(X_train, y_train, tscv)
    lgb_val_f1  = f1_score(y_val,  lgb_model.predict(X_val),  average="macro")
    lgb_test_f1 = f1_score(y_test, lgb_model.predict(X_test), average="macro")
    lgb_test_acc = accuracy_score(y_test, lgb_model.predict(X_test))
    logger.info(
        "  LGB ▸ CV F1: %.4f | Val F1: %.4f | Test F1: %.4f",
        lgb_meta["cv_f1_mean"], lgb_val_f1, lgb_test_f1,
    )
    results["LightGBM"] = {**lgb_meta, "val_f1": lgb_val_f1,
                            "test_f1": lgb_test_f1, "test_acc": lgb_test_acc,
                            "_model": lgb_model}

    # ── Select best by validation F1 ──────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["val_f1"])
    best = results[best_name]
    logger.info(
        "21d BEST: %s ▸ Val F1: %.4f | Test F1: %.4f | Test Acc: %.4f",
        best_name, best["val_f1"], best["test_f1"], best["test_acc"],
    )

    # Per-class report
    best_model = best["_model"]
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["DOWN", "UP"], output_dict=True)

    # ── Save model ────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": best_model,
        "feature_cols": feature_cols,
        "best_model": best_name,
        "horizon": HORIZON,
        "cv_f1_mean": best["cv_f1_mean"],
        "cv_f1_std":  best["cv_f1_std"],
        "val_f1_macro": best["val_f1"],
        "test_f1_macro": best["test_f1"],
        "test_accuracy": best["test_acc"],
        "per_class": {
            cls: {k: round(v, 4) for k, v in report[cls].items() if k != "support"}
            for cls in ("DOWN", "UP")
        },
        "per_model": {
            name: {k: v for k, v in r.items() if k != "_model"}
            for name, r in results.items()
        },
    }
    with open(MODEL_21D_PATH, "wb") as f:
        pickle.dump(bundle, f)
    logger.info("21-day model saved to %s", MODEL_21D_PATH)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("21-DAY MODEL RESULTS")
    print("=" * 72)
    print(f"{'Model':<16} {'CV F1':>12}  {'Val F1':>10}  {'Test F1':>10}  {'Test Acc':>10}")
    print("-" * 72)
    for name, r in results.items():
        star = " *BEST*" if name == best_name else ""
        print(
            f"{name:<16} {r['cv_f1_mean']:.4f}±{r['cv_f1_std']:.4f}  "
            f"{r['val_f1']:>10.4f}  {r['test_f1']:>10.4f}  {r['test_acc']:>10.4f}{star}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
