"""Train a 21-day horizon model using existing features with recomputed target."""

import logging
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_validate

from src.config import (
    CV_FOLDS,
    FEATURES_CV_PATH,
    FEATURES_MARKET_PATH,
    FEATURES_NLP_PATH,
    MODEL_21D_PATH,
    MODELS_DIR,
    RAW_MARKET_DIR,
    TARGET_CLASSES,
    TEST_START,
    TRAIN_END,
    VAL_END,
    VAL_START,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

HORIZON = 21
EXCLUDE = {"ticker", "target", "close", "vix_regime", "rsi_zone",
           "vader_label", "finbert_label", "chart_available"}


def recompute_target(market_df: pd.DataFrame) -> pd.DataFrame:
    """Replace target column with 21-day forward return direction."""
    result = []
    for ticker in market_df["ticker"].unique():
        mask = market_df["ticker"] == ticker
        chunk = market_df.loc[mask].copy()

        # Load raw OHLCV to get Close prices for forward return
        csv_path = RAW_MARKET_DIR / f"{ticker}.csv"
        if not csv_path.exists():
            continue
        raw = pd.read_csv(csv_path, index_col="Date", parse_dates=True).sort_index()
        close = raw["Close"]

        # Compute 21-day forward return
        fwd_return = close.shift(-HORIZON) / close - 1
        target = pd.Series("DOWN", index=close.index, name="target")
        target[fwd_return > 0] = "UP"
        target[fwd_return.isna()] = np.nan

        # Map target to feature index
        chunk["target"] = target.reindex(chunk.index)
        chunk = chunk.dropna(subset=["target"])
        result.append(chunk)

    return pd.concat(result)


def main():
    logger.info("Loading market features ...")
    market = pd.read_parquet(FEATURES_MARKET_PATH)
    market.index = pd.to_datetime(market.index)
    market.index.name = "date"

    logger.info("Recomputing target with horizon=%d ...", HORIZON)
    market = recompute_target(market)
    logger.info("After target recompute: %d rows", len(market))

    # Join NLP features
    market_mi = market.set_index("ticker", append=True)

    logger.info("Loading NLP features ...")
    nlp = pd.read_parquet(FEATURES_NLP_PATH)
    nlp.index = pd.to_datetime(nlp.index)
    nlp.index.name = "date"
    nlp_cols = [c for c in nlp.columns if c != "ticker"]
    nlp_mi = nlp.set_index("ticker", append=True)[nlp_cols]
    combined_mi = market_mi.join(nlp_mi, how="left")
    combined_mi[nlp_cols] = combined_mi[nlp_cols].fillna(0)

    # Join CV features
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

    feature_cols = [c for c in df.columns
                    if c not in EXCLUDE and not c.startswith("Unnamed")]

    # Temporal split
    train = df[df.index <= TRAIN_END]
    test = df[df.index >= TEST_START]

    X_train = train[feature_cols].fillna(0)
    y_train = train["target"]
    X_test = test[feature_cols].fillna(0)
    y_test = test["target"]

    logger.info("Training RF for %d-day horizon: %d train / %d test rows, %d features",
                HORIZON, len(X_train), len(X_test), len(feature_cols))

    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    model = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    scores = cross_validate(
        model, X_train, y_train, cv=tscv,
        scoring=["f1_macro", "accuracy"], n_jobs=1,
    )
    logger.info("CV F1: %.4f +/- %.4f",
                scores["test_f1_macro"].mean(), scores["test_f1_macro"].std())

    model.fit(X_train, y_train)

    from sklearn.metrics import f1_score, accuracy_score
    y_pred = model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    test_acc = accuracy_score(y_test, y_pred)
    logger.info("Test F1: %.4f | Test Acc: %.4f", test_f1, test_acc)

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_21D_PATH, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)
    logger.info("21-day model saved to %s", MODEL_21D_PATH)


if __name__ == "__main__":
    main()
