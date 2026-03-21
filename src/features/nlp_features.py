"""
nlp_features.py — Aggregate NLP sentiment outputs into per-ticker-day ML features.

Takes the raw headline-level FinBERT + VADER scores and produces a flat
DataFrame (one row per ticker-day) that can be joined with market features.

Coverage strategy (multi-layer fallback):
  1. Ticker-specific news → direct FinBERT/VADER scores
  2. Sector-level average sentiment → for tickers with no news on a given day
  3. Market-wide average sentiment → for days with no sector news
  4. Forward-fill remaining gaps → carry last known sentiment

Features produced per ticker-day:
  finbert_sentiment      — mean FinBERT compound score
  finbert_confidence     — mean FinBERT confidence
  vader_sentiment        — mean VADER compound score
  news_volume_1d         — headline count on this day
  news_volume_5d         — rolling 5-day headline count
  headline_avg_length    — mean word count of headlines
  sentiment_momentum     — finbert_sentiment - finbert_sentiment 5 days ago
  sentiment_dispersion   — std of FinBERT scores across headlines
  sentiment_shift_3d     — finbert change over 3 days
  sentiment_surprise     — z-score vs 20-day rolling mean
  sentiment_x_volume     — finbert * volume_ratio interaction
  news_volume_zscore     — z-score of news_volume_1d vs 20-day rolling
  is_sentiment_imputed   — 1 if sentiment was imputed (sector/market/ffill)
  finbert_embed_pca_1..10 — 10 PCA dims of mean CLS embeddings

Usage:
    python -m src.features.nlp_features --test   # AAPL, MSFT, NVDA
    python -m src.features.nlp_features          # all tickers
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURES_MARKET_PATH,
    FEATURES_NLP_PATH,
    NLP_PCA_COMPONENTS,
    PROCESSED_DIR,
    RAW_NEWS_DIR,
    TICKER_SECTOR_MAP,
    TICKERS_ALL,
)
from src.nlp.finbert_sentiment import FinBertPipeline, FINBERT_CACHE_PATH
from src.nlp.vader_sentiment import VaderPipeline

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

TEST_TICKERS = ["AAPL", "MSFT", "NVDA"]

# PCA model cache
PCA_CACHE_PATH = PROCESSED_DIR / "pca_nlp_embeddings.pkl"


def _load_scored_news(ticker: str, news_dir: Path) -> pd.DataFrame:
    """Load news for a ticker and run FinBERT + VADER if not cached.

    Args:
        ticker: Ticker symbol.
        news_dir: Directory with raw news Parquet files.

    Returns:
        DataFrame with headline-level FinBERT and VADER scores.
        Empty if no news available.
    """
    news_path = news_dir / f"{ticker}.parquet"
    if not news_path.exists() or pd.read_parquet(news_path).empty:
        return pd.DataFrame()

    news_df = pd.read_parquet(news_path)
    if news_df.empty:
        return pd.DataFrame()

    # Ensure published is datetime with UTC tz removed
    news_df["published"] = pd.to_datetime(news_df["published"], utc=True).dt.tz_localize(None)
    news_df["date"] = news_df["published"].dt.normalize()

    # Run FinBERT
    finbert = FinBertPipeline()
    texts = news_df["title"].fillna("").tolist()
    logger.info("%s: running FinBERT on %d headlines ...", ticker, len(texts))
    fb_scores = finbert.score(texts, return_embeddings=True)
    news_df = pd.concat([news_df.reset_index(drop=True), fb_scores.drop(columns=["text"])], axis=1)

    # Run VADER
    vader = VaderPipeline()
    vd_scores = vader.score(texts).drop(columns=["text"])
    news_df = pd.concat([news_df, vd_scores], axis=1)

    # Headline word count
    news_df["headline_length"] = news_df["title"].fillna("").str.split().str.len()

    return news_df


def build_ticker_nlp_features(
    ticker: str,
    news_dir: Path = RAW_NEWS_DIR,
    market_path: Path = FEATURES_MARKET_PATH,
) -> pd.DataFrame:
    """Aggregate headline-level NLP scores into per-ticker-day features.

    Uses raw OHLCV trading days as the date index (not the features parquet,
    which drops the last 5 days for the target label). This ensures recent
    news headlines align with the most recent market dates.

    Args:
        ticker: Ticker symbol.
        news_dir: Directory with raw news Parquet files.
        market_path: Path to market features Parquet (for date index alignment).

    Returns:
        DataFrame with DatetimeIndex (date) and NLP feature columns.
        Index dates match trading days in the raw OHLCV CSV.
    """
    scored = _load_scored_news(ticker, news_dir)

    # Use raw OHLCV dates so recent news (after target-label cutoff) is included
    from src.config import RAW_MARKET_DIR
    raw_csv = RAW_MARKET_DIR / f"{ticker}.csv"
    if raw_csv.exists():
        raw_df = pd.read_csv(raw_csv, index_col="Date", parse_dates=True)
        date_index = pd.DatetimeIndex(sorted(raw_df.index.normalize().unique()))
    else:
        # Fallback: use market features dates
        market_df = pd.read_parquet(market_path)
        market_df.index = pd.to_datetime(market_df.index)
        ticker_dates = market_df[market_df["ticker"] == ticker].index.normalize().unique()
        date_index = pd.DatetimeIndex(sorted(ticker_dates))

    # Template: one row per market date, NaN-filled by default
    feat = pd.DataFrame(index=date_index)
    feat.index.name = "date"
    feat["ticker"] = ticker

    if scored.empty:
        # No news → fill all NLP features with 0 (neutral)
        _fill_zero_nlp_features(feat)
        return feat

    # Daily aggregation
    embed_cols = [c for c in scored.columns if c.startswith("embed_")]

    daily = scored.groupby("date").agg(
        finbert_sentiment=("finbert_score", "mean"),
        finbert_confidence=("finbert_confidence", "mean"),
        vader_sentiment=("vader_compound", "mean"),
        news_volume_1d=("title", "count"),
        headline_avg_length=("headline_length", "mean"),
        sentiment_dispersion=("finbert_score", "std"),
    )

    # Mean embedding per day
    if embed_cols:
        daily_embeds = scored.groupby("date")[embed_cols].mean()
        daily = daily.join(daily_embeds)

    # Reindex to market dates (forward-fill for gaps, then fill remaining NaN)
    daily = daily.reindex(date_index)
    daily["news_volume_1d"] = daily["news_volume_1d"].fillna(0)

    # Rolling 5-day news volume
    daily["news_volume_5d"] = daily["news_volume_1d"].rolling(5, min_periods=1).sum()

    # Sentiment momentum: today - 5 days ago (forward-fill sentiment first)
    daily["finbert_sentiment"] = daily["finbert_sentiment"].ffill().fillna(0)
    daily["sentiment_momentum"] = (
        daily["finbert_sentiment"] - daily["finbert_sentiment"].shift(5)
    ).fillna(0)

    # Fill remaining NaN columns
    daily = daily.fillna(0)

    feat = feat.join(daily.drop(columns=["ticker"], errors="ignore"))
    return feat


def _fill_zero_nlp_features(feat: pd.DataFrame) -> None:
    """Fill all NLP feature columns with neutral/zero values in-place."""
    scalar_cols = [
        "finbert_sentiment", "finbert_confidence", "vader_sentiment",
        "news_volume_1d", "news_volume_5d", "headline_avg_length",
        "sentiment_momentum", "sentiment_dispersion",
        "sentiment_shift_3d", "sentiment_surprise",
        "sentiment_x_volume", "news_volume_zscore",
        "is_sentiment_imputed",
    ]
    for col in scalar_cols:
        feat[col] = 0.0
    for i in range(NLP_PCA_COMPONENTS):
        feat[f"finbert_embed_pca_{i+1}"] = 0.0


def _apply_sector_and_market_fallback(combined: pd.DataFrame) -> pd.DataFrame:
    """Apply multi-layer sentiment fallback to fill coverage gaps.

    Layer 1: Ticker-specific sentiment (already computed)
    Layer 2: Sector average sentiment for that day
    Layer 3: Market-wide average sentiment for that day
    Layer 4: Forward-fill remaining gaps

    Args:
        combined: Full NLP feature DataFrame with all tickers.

    Returns:
        DataFrame with improved coverage and is_sentiment_imputed flag.
    """
    sentiment_cols = ["finbert_sentiment", "finbert_confidence", "vader_sentiment"]

    # Mark which rows originally had ticker-specific news
    has_news = combined["news_volume_1d"] > 0
    combined["is_sentiment_imputed"] = 0.0

    # Add sector column for grouping
    combined["_sector"] = combined["ticker"].map(TICKER_SECTOR_MAP).fillna("Unknown")

    # Compute sector-level daily average (only from rows WITH news)
    news_rows = combined[has_news]
    sector_daily = news_rows.groupby([news_rows.index, "_sector"])[sentiment_cols].mean()
    sector_daily.index.names = ["date", "_sector"]

    # Compute market-wide daily average (only from rows WITH news)
    market_daily = news_rows.groupby(news_rows.index)[sentiment_cols].mean()

    # Apply fallbacks
    for col in sentiment_cols:
        missing = combined[col] == 0  # rows without direct news
        if not missing.any():
            continue

        # Layer 2: Sector fallback
        for idx in combined[missing].index.unique():
            for sector in combined.loc[[idx], "_sector"].unique():
                mask = missing & (combined.index == idx) & (combined["_sector"] == sector)
                if mask.any() and (idx, sector) in sector_daily.index:
                    combined.loc[mask, col] = sector_daily.loc[(idx, sector), col]
                    combined.loc[mask, "is_sentiment_imputed"] = 1.0

        # Layer 3: Market-wide fallback (still missing)
        still_missing = (combined[col] == 0) & missing
        for idx in combined[still_missing].index.unique():
            mask = still_missing & (combined.index == idx)
            if mask.any() and idx in market_daily.index:
                combined.loc[mask, col] = market_daily.loc[idx, col]
                combined.loc[mask, "is_sentiment_imputed"] = 1.0

    # Layer 4: Forward-fill remaining gaps per ticker
    for col in sentiment_cols:
        combined[col] = combined.groupby("ticker")[col].ffill()
        still_zero = combined[col] == 0
        combined.loc[still_zero, col] = 0  # keep 0 for truly unavailable
        combined.loc[still_zero & ~has_news, "is_sentiment_imputed"] = 1.0

    combined = combined.drop(columns=["_sector"])

    return combined


def _add_dynamic_nlp_features(combined: pd.DataFrame) -> pd.DataFrame:
    """Add dynamic NLP features that capture sentiment CHANGES, not just levels.

    Args:
        combined: Full NLP feature DataFrame (after fallback).

    Returns:
        DataFrame with additional dynamic features.
    """
    # Sentiment shift over 3 days
    combined["sentiment_shift_3d"] = combined.groupby("ticker")["finbert_sentiment"].transform(
        lambda x: x - x.shift(3)
    ).fillna(0)

    # Sentiment surprise: z-score vs 20-day rolling mean
    rolling_mean = combined.groupby("ticker")["finbert_sentiment"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    rolling_std = combined.groupby("ticker")["finbert_sentiment"].transform(
        lambda x: x.rolling(20, min_periods=5).std()
    )
    combined["sentiment_surprise"] = (
        (combined["finbert_sentiment"] - rolling_mean) / (rolling_std + 1e-8)
    ).fillna(0)

    # News volume z-score (unusual number of articles)
    vol_mean = combined.groupby("ticker")["news_volume_1d"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    vol_std = combined.groupby("ticker")["news_volume_1d"].transform(
        lambda x: x.rolling(20, min_periods=5).std()
    )
    combined["news_volume_zscore"] = (
        (combined["news_volume_1d"] - vol_mean) / (vol_std + 1e-8)
    ).fillna(0)

    # Sentiment × volume interaction — needs volume_ratio from market features
    # We'll compute this using news_volume_1d as a proxy since we don't have
    # market volume here. The actual interaction will be captured via the model.
    combined["sentiment_x_volume"] = (
        combined["finbert_sentiment"] * combined["news_volume_1d"]
    ).fillna(0)

    return combined


def build_all_nlp_features(
    tickers: list[str] = TICKERS_ALL,
    news_dir: Path = RAW_NEWS_DIR,
    market_path: Path = FEATURES_MARKET_PATH,
    output_path: Path = FEATURES_NLP_PATH,
    n_pca: int = NLP_PCA_COMPONENTS,
) -> pd.DataFrame:
    """Build the full NLP feature matrix for all tickers and save to Parquet.

    Runs FinBERT + VADER per ticker, aggregates to daily features,
    applies sector/market fallback for coverage, adds dynamic features,
    fits PCA on CLS embeddings, and saves the result.

    Args:
        tickers: List of ticker symbols.
        news_dir: Directory with raw news Parquet files.
        market_path: Market features path for date alignment.
        output_path: Output Parquet path.
        n_pca: Number of PCA dimensions for embedding features.

    Returns:
        Combined NLP feature DataFrame.
    """
    all_frames: list[pd.DataFrame] = []

    for i, ticker in enumerate(tickers, 1):
        logger.info("[%d/%d] NLP features for %s ...", i, len(tickers), ticker)
        feat = build_ticker_nlp_features(ticker, news_dir=news_dir, market_path=market_path)
        if feat is not None and not feat.empty:
            all_frames.append(feat)

    if not all_frames:
        raise RuntimeError("No NLP features computed — check news data exists.")

    combined = pd.concat(all_frames, axis=0)
    combined.index.name = "date"

    # --- Multi-layer sentiment fallback ---
    before_coverage = (combined["news_volume_1d"] > 0).sum()
    logger.info("Before fallback: %d/%d rows have direct news (%.1f%%)",
                before_coverage, len(combined), 100 * before_coverage / len(combined))

    combined = _apply_sector_and_market_fallback(combined)

    has_sentiment = (combined["finbert_sentiment"] != 0).sum()
    logger.info("After fallback: %d/%d rows have sentiment (%.1f%%)",
                has_sentiment, len(combined), 100 * has_sentiment / len(combined))

    # --- Dynamic NLP features ---
    combined = _add_dynamic_nlp_features(combined)

    # --- Fit PCA on CLS embeddings ---
    embed_cols = [c for c in combined.columns if c.startswith("embed_")]
    if embed_cols:
        rows_with_news = combined["news_volume_1d"] > 0
        n_news_rows = rows_with_news.sum()
        logger.info(
            "Fitting PCA (%d dims) on %d rows with news coverage (of %d total) ...",
            n_pca, n_news_rows, len(combined),
        )
        # Initialise PCA columns to zero for all rows
        for i in range(n_pca):
            combined[f"finbert_embed_pca_{i+1}"] = 0.0

        if n_news_rows >= n_pca:
            scaler = StandardScaler()
            embed_matrix = combined.loc[rows_with_news, embed_cols].fillna(0).values
            embed_scaled = scaler.fit_transform(embed_matrix)
            pca = PCA(n_components=n_pca, random_state=42)
            pca_result = pca.fit_transform(embed_scaled)
            for i in range(n_pca):
                combined.loc[rows_with_news, f"finbert_embed_pca_{i+1}"] = pca_result[:, i]
            explained = pca.explained_variance_ratio_.sum() * 100
            logger.info("PCA explained variance: %.1f%%", explained)

            import pickle
            PCA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(PCA_CACHE_PATH, "wb") as f:
                pickle.dump({"scaler": scaler, "pca": pca}, f)
        else:
            logger.warning(
                "Too few news rows (%d) for PCA — embed PCA features set to 0. "
                "Re-run after collecting more historical news.",
                n_news_rows,
            )

        combined = combined.drop(columns=embed_cols)

    # Ensure new dynamic features are present even if no news at all
    for col in ["sentiment_shift_3d", "sentiment_surprise",
                "sentiment_x_volume", "news_volume_zscore", "is_sentiment_imputed"]:
        if col not in combined.columns:
            combined[col] = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path)

    covered = combined[combined["news_volume_1d"] > 0]["ticker"].nunique()
    imputed = (combined["is_sentiment_imputed"] == 1).sum()
    logger.info(
        "NLP features saved: %d rows x %d cols | %d/%d tickers have news | %d rows imputed",
        len(combined), len(combined.columns), covered, len(tickers), imputed,
    )
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build NLP feature matrix")
    parser.add_argument("--test", action="store_true", help="3-ticker smoke test")
    args = parser.parse_args()

    tickers = TEST_TICKERS if args.test else TICKERS_ALL
    build_all_nlp_features(tickers=tickers)
