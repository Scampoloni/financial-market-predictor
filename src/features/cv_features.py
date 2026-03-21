"""
cv_features.py — Aggregate CNN chart embeddings into per-ticker-day ML features.

Takes the pre-generated candlestick chart PNGs, runs EfficientNet-B0 to get
1280-dim embeddings, applies PCA to 10 components, and produces a flat
DataFrame (one row per ticker-day) aligned to the market feature date index.

Features produced per ticker-day:
  chart_embed_pca_1..10 — 10 PCA dims of EfficientNet-B0 embeddings
  chart_available        — 1 if a chart image exists for this date, 0 otherwise

Usage:
    python -m src.features.cv_features --test   # AAPL, MSFT, NVDA
    python -m src.features.cv_features          # all tickers
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import (
    CV_PCA_COMPONENTS,
    FEATURES_CV_PATH,
    FEATURES_MARKET_PATH,
    MODELS_DIR,
    PCA_CV_PATH,
    RAW_CHARTS_DIR,
    RAW_MARKET_DIR,
    TICKERS_ALL,
)
from src.cv.chart_classifier import ChartCNN

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

TEST_TICKERS = ["AAPL", "MSFT", "NVDA"]


def _get_date_index(ticker: str, market_path: Path) -> pd.DatetimeIndex:
    """Return trading-day date index for a ticker from raw OHLCV CSV.

    Falls back to market features parquet if CSV is missing.

    Args:
        ticker: Ticker symbol.
        market_path: Path to market features parquet.

    Returns:
        Sorted DatetimeIndex of trading days.
    """
    raw_csv = RAW_MARKET_DIR / f"{ticker}.csv"
    if raw_csv.exists():
        raw_df = pd.read_csv(raw_csv, index_col="Date", parse_dates=True)
        return pd.DatetimeIndex(sorted(raw_df.index.normalize().unique()))

    market_df = pd.read_parquet(market_path)
    market_df.index = pd.to_datetime(market_df.index)
    ticker_dates = market_df[market_df["ticker"] == ticker].index.normalize().unique()
    return pd.DatetimeIndex(sorted(ticker_dates))


def build_ticker_cv_features(
    ticker: str,
    charts_dir: Path = RAW_CHARTS_DIR,
    market_path: Path = FEATURES_MARKET_PATH,
    cnn: ChartCNN | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Embed all available chart images for a ticker.

    Args:
        ticker: Ticker symbol.
        charts_dir: Root directory for chart PNGs.
        market_path: Market features path for date alignment.
        cnn: Shared ChartCNN instance (created if None).

    Returns:
        Tuple of:
          - DataFrame with DatetimeIndex (date), 'ticker', 'chart_available'
            columns, plus raw embed columns (embed_0..embed_1279).
          - np.ndarray of raw embeddings (N_charts, 1280) for PCA fitting.
    """
    if cnn is None:
        cnn = ChartCNN()

    date_index = _get_date_index(ticker, market_path)
    feat = pd.DataFrame(index=date_index)
    feat.index.name = "date"
    feat["ticker"] = ticker
    feat["chart_available"] = 0

    ticker_chart_dir = charts_dir / ticker
    if not ticker_chart_dir.exists():
        logger.warning("%s: no chart directory at %s", ticker, ticker_chart_dir)
        return feat, np.empty((0, 1280), dtype=np.float32)

    chart_files = sorted(ticker_chart_dir.glob("*.png"))
    if not chart_files:
        logger.warning("%s: no chart PNGs found", ticker)
        return feat, np.empty((0, 1280), dtype=np.float32)

    # Build date → file mapping
    date_to_file: dict[pd.Timestamp, Path] = {}
    for f in chart_files:
        try:
            date_to_file[pd.Timestamp(f.stem)] = f
        except Exception:
            pass

    # Find dates that have chart images and are in the date index
    chart_dates = [d for d in date_index if d in date_to_file]
    if not chart_dates:
        logger.warning("%s: chart dates do not overlap with market dates", ticker)
        return feat, np.empty((0, 1280), dtype=np.float32)

    logger.info("%s: embedding %d chart images ...", ticker, len(chart_dates))
    paths = [date_to_file[d] for d in chart_dates]
    embeddings = cnn.embed_batch(paths)  # (N, 1280)

    # Store raw embeddings in feat for PCA fitting later
    embed_cols = [f"embed_{i}" for i in range(embeddings.shape[1])]
    embed_df = pd.DataFrame(embeddings, index=chart_dates, columns=embed_cols)
    embed_df.index.name = "date"

    feat = feat.join(embed_df)
    feat.loc[chart_dates, "chart_available"] = 1

    return feat, embeddings


def build_all_cv_features(
    tickers: list[str] = TICKERS_ALL,
    charts_dir: Path = RAW_CHARTS_DIR,
    market_path: Path = FEATURES_MARKET_PATH,
    output_path: Path = FEATURES_CV_PATH,
    n_pca: int = CV_PCA_COMPONENTS,
) -> pd.DataFrame:
    """Build the full CV feature matrix for all tickers and save to Parquet.

    Runs EfficientNet-B0 on all chart images, fits PCA on covered rows,
    and saves the result.

    Args:
        tickers: List of ticker symbols.
        charts_dir: Root chart image directory.
        market_path: Market features path for date alignment.
        output_path: Output Parquet path.
        n_pca: Number of PCA components.

    Returns:
        Combined CV feature DataFrame.
    """
    cnn = ChartCNN()
    all_frames: list[pd.DataFrame] = []
    all_embeddings: list[np.ndarray] = []

    for i, ticker in enumerate(tickers, 1):
        logger.info("[%d/%d] CV features for %s ...", i, len(tickers), ticker)
        feat, embeds = build_ticker_cv_features(
            ticker, charts_dir=charts_dir, market_path=market_path, cnn=cnn
        )
        if feat is not None and not feat.empty:
            all_frames.append(feat)
        if embeds.shape[0] > 0:
            all_embeddings.append(embeds)

    if not all_frames:
        raise RuntimeError("No CV features computed — run chart_generator first.")

    combined = pd.concat(all_frames, axis=0)
    combined.index.name = "date"

    # Initialise PCA columns
    for i in range(n_pca):
        combined[f"chart_embed_pca_{i+1}"] = 0.0

    embed_cols = [c for c in combined.columns if c.startswith("embed_")]
    rows_with_chart = combined["chart_available"] == 1
    n_chart_rows = rows_with_chart.sum()

    logger.info(
        "Fitting PCA (%d dims) on %d chart rows (of %d total) ...",
        n_pca, n_chart_rows, len(combined),
    )

    if n_chart_rows >= n_pca and all_embeddings:
        embed_matrix = combined.loc[rows_with_chart, embed_cols].fillna(0).values
        scaler = StandardScaler()
        embed_scaled = scaler.fit_transform(embed_matrix)
        pca = PCA(n_components=n_pca, random_state=42)
        pca_result = pca.fit_transform(embed_scaled)

        for i in range(n_pca):
            combined.loc[rows_with_chart, f"chart_embed_pca_{i+1}"] = pca_result[:, i]

        explained = pca.explained_variance_ratio_.sum() * 100
        logger.info("PCA explained variance: %.1f%%", explained)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(PCA_CV_PATH, "wb") as f:
            pickle.dump({"scaler": scaler, "pca": pca}, f)
        logger.info("PCA model saved to %s", PCA_CV_PATH)
    else:
        logger.warning(
            "Too few chart rows (%d) for PCA — CV PCA features set to 0. "
            "Run chart_generator first.",
            n_chart_rows,
        )

    # Drop raw embedding columns — keep only PCA + metadata
    combined = combined.drop(columns=embed_cols, errors="ignore")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path)

    covered = combined[combined["chart_available"] == 1]["ticker"].nunique()
    logger.info(
        "CV features saved: %d rows x %d cols | %d/%d tickers have charts",
        len(combined), len(combined.columns), covered, len(tickers),
    )
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build CV feature matrix")
    parser.add_argument("--test", action="store_true", help="3-ticker smoke test")
    args = parser.parse_args()

    tickers = TEST_TICKERS if args.test else TICKERS_ALL
    build_all_cv_features(tickers=tickers)
