"""
market_features.py — Compute technical indicators and target labels from OHLCV data.

Produces a flat feature DataFrame (one row per ticker-day) ready for ML training.
All features use ONLY past/current data — no future leakage.

Usage:
    python -m src.features.market_features              # build full feature set
    python -m src.features.market_features --test       # AAPL, MSFT, NVDA only
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ATR_PERIOD,
    BB_PERIOD,
    BB_STD,
    DATA_END_DATE,
    DATA_START_DATE,
    EMA_PERIOD,
    FEATURES_MARKET_PATH,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MARKET_INDICES,
    RAW_MARKET_DIR,
    RETURN_PERIODS,
    RSI_PERIOD,
    SMA_LONG,
    SMA_SHORT,
    TARGET_DOWN_THRESHOLD,
    TARGET_HORIZON_DAYS,
    TARGET_UP_THRESHOLD,
    TICKER_SECTOR_MAP,
    TICKERS_ALL,
    VOLATILITY_PERIOD,
    VOLUME_AVG_PERIOD,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TEST_TICKERS = ["AAPL", "MSFT", "NVDA"]


# ---------------------------------------------------------------------------
# Individual indicator functions
# ---------------------------------------------------------------------------

def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Compute Relative Strength Index (RSI).

    RSI measures momentum: values >70 indicate overbought, <30 oversold.
    Uses Wilder's smoothing (exponential weighted mean with alpha=1/period).

    Args:
        close: Series of closing prices.
        period: Lookback window (default 14 days).

    Returns:
        Series of RSI values in range [0, 100].
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename("rsi_14")


def compute_macd(
    close: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    """Compute MACD line, signal line, and histogram.

    MACD = EMA(fast) - EMA(slow). Signal = EMA(MACD, signal period).
    Positive MACD histogram → bullish momentum; negative → bearish.

    Args:
        close: Series of closing prices.
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal: Signal EMA period (default 9).

    Returns:
        DataFrame with columns [macd, macd_signal, macd_hist].
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram}
    )


def compute_bollinger_bands(
    close: pd.Series,
    period: int = BB_PERIOD,
    num_std: float = BB_STD,
) -> pd.DataFrame:
    """Compute Bollinger Band distances.

    Returns distance from close to upper/lower bands (not the bands themselves).
    Positive bb_upper_dist → price is below upper band; negative → above upper band.

    Args:
        close: Series of closing prices.
        period: Rolling window (default 20).
        num_std: Number of standard deviations (default 2).

    Returns:
        DataFrame with columns [bb_upper_dist, bb_lower_dist, bb_width].
    """
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return pd.DataFrame(
        {
            "bb_upper_dist": upper - close,
            "bb_lower_dist": close - lower,
            "bb_width": (upper - lower) / sma,  # normalized band width
        }
    )


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = ATR_PERIOD,
) -> pd.Series:
    """Compute Average True Range (ATR) — a measure of volatility.

    True Range = max(H-L, |H-prevC|, |L-prevC|). ATR = EMA of TR.
    Higher ATR → higher intraday volatility.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        period: Lookback window (default 14).

    Returns:
        Series of ATR values.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean().rename("atr_14")


def compute_returns(close: pd.Series, periods: list[int] = RETURN_PERIODS) -> pd.DataFrame:
    """Compute rolling returns for multiple lookback periods.

    Return = (close_t - close_{t-n}) / close_{t-n}.
    These capture short-, medium-, and long-term price momentum.

    Args:
        close: Series of closing prices.
        periods: List of lookback days (default [1, 5, 20]).

    Returns:
        DataFrame with columns [return_1d, return_5d, return_20d].
    """
    cols = {}
    for p in periods:
        cols[f"return_{p}d"] = close.pct_change(p)
    return pd.DataFrame(cols)


def compute_volume_ratio(volume: pd.Series, period: int = VOLUME_AVG_PERIOD) -> pd.Series:
    """Compute volume relative to its rolling average.

    Ratio > 1 → above-average trading activity (potential breakout or news event).

    Args:
        volume: Daily volume series.
        period: Rolling average window (default 20).

    Returns:
        Series of volume ratios.
    """
    avg_vol = volume.rolling(period).mean()
    return (volume / avg_vol.replace(0, np.nan)).rename("volume_ratio")


def compute_sma_ratios(close: pd.Series) -> pd.DataFrame:
    """Compute close-to-SMA and close-to-EMA ratios.

    Ratio > 1 → price above moving average (bullish signal).
    Ratio < 1 → price below moving average (bearish signal).

    Args:
        close: Series of closing prices.

    Returns:
        DataFrame with columns [sma_20_ratio, sma_50_ratio, ema_12_ratio].
    """
    sma_20 = close.rolling(SMA_SHORT).mean()
    sma_50 = close.rolling(SMA_LONG).mean()
    ema_12 = close.ewm(span=EMA_PERIOD, adjust=False).mean()
    return pd.DataFrame(
        {
            "sma_20_ratio": close / sma_20.replace(0, np.nan),
            "sma_50_ratio": close / sma_50.replace(0, np.nan),
            "ema_12_ratio": close / ema_12.replace(0, np.nan),
        }
    )


def compute_volatility(close: pd.Series, period: int = VOLATILITY_PERIOD) -> pd.Series:
    """Compute rolling standard deviation of daily returns (historical volatility).

    Higher volatility → less predictable price movements.

    Args:
        close: Series of closing prices.
        period: Rolling window (default 20).

    Returns:
        Series of annualized-equivalent rolling volatility.
    """
    return close.pct_change().rolling(period).std().rename("volatility_20d")


def compute_cyclical_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Encode day-of-week and month as sine/cosine pairs to capture cyclicality.

    Sin/cos encoding avoids the artificial discontinuity between e.g. Monday (0)
    and Friday (4) that integer encoding would imply.

    Args:
        index: DatetimeIndex of the OHLCV data.

    Returns:
        DataFrame with columns [dow_sin, dow_cos, month_sin, month_cos].
    """
    dow = index.dayofweek  # 0=Monday, 4=Friday
    month = index.month    # 1–12
    return pd.DataFrame(
        {
            "dow_sin": np.sin(2 * np.pi * dow / 5),
            "dow_cos": np.cos(2 * np.pi * dow / 5),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
        },
        index=index,
    )


def compute_target(close: pd.Series, horizon: int = TARGET_HORIZON_DAYS) -> pd.Series:
    """Compute the classification target: forward return direction.

    Target classes:
      UP      →  forward return > TARGET_UP_THRESHOLD (+1%)
      DOWN    →  forward return < TARGET_DOWN_THRESHOLD (-1%)
      SIDEWAYS → otherwise

    IMPORTANT: The target uses FUTURE close prices. Rows near the end of the
    series will be NaN — these must be dropped before training to prevent leakage.

    Args:
        close: Series of closing prices.
        horizon: Number of trading days ahead to predict (default 5).

    Returns:
        Series of string labels ['DOWN', 'SIDEWAYS', 'UP'].
    """
    fwd_return = close.shift(-horizon) / close - 1
    target = pd.Series("SIDEWAYS", index=close.index, name="target")
    target[fwd_return > TARGET_UP_THRESHOLD] = "UP"
    target[fwd_return < TARGET_DOWN_THRESHOLD] = "DOWN"
    target[fwd_return.isna()] = np.nan
    return target


# ---------------------------------------------------------------------------
# Per-ticker pipeline
# ---------------------------------------------------------------------------

def build_ticker_features(
    ticker: str,
    data_dir: Path = RAW_MARKET_DIR,
    vix_df: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Build the full feature matrix for a single ticker.

    Reads raw OHLCV CSV, computes all technical indicators, adds VIX,
    sector encoding, time features, and the classification target.

    Args:
        ticker: Ticker symbol.
        data_dir: Directory containing raw CSV files.
        vix_df: Pre-loaded VIX DataFrame to avoid repeated disk reads.

    Returns:
        Feature DataFrame with DatetimeIndex, or None on failure.
    """
    csv_path = data_dir / f"{ticker}.csv"
    if not csv_path.exists():
        logger.warning("%s: CSV not found at %s — skipping", ticker, csv_path)
        return None

    try:
        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        df = df.sort_index()

        # Require minimum 60 rows to compute SMA-50 + some margin
        if len(df) < 60:
            logger.warning("%s: only %d rows — skipping", ticker, len(df))
            return None

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        features = pd.DataFrame(index=df.index)
        features["ticker"] = ticker
        features["sector"] = TICKER_SECTOR_MAP.get(ticker, "Unknown")
        features["close"] = close

        # Returns
        features = features.join(compute_returns(close))

        # Momentum
        features["rsi_14"] = compute_rsi(close)
        features = features.join(compute_macd(close))

        # Trend
        features = features.join(compute_sma_ratios(close))

        # Volatility
        features["volatility_20d"] = compute_volatility(close)
        features["atr_14"] = compute_atr(high, low, close)

        # Bollinger Bands
        features = features.join(compute_bollinger_bands(close))

        # Volume
        features["volume_ratio"] = compute_volume_ratio(volume)

        # Time features (cyclical)
        features = features.join(compute_cyclical_time_features(df.index))

        # VIX (market fear index)
        if vix_df is not None and not vix_df.empty:
            features = features.join(vix_df[["Close"]].rename(columns={"Close": "vix_level"}))
        else:
            features["vix_level"] = np.nan

        # Target label (uses future prices — rows at tail will be NaN)
        features["target"] = compute_target(close)

        # Drop rows with NaN target (end of series) or insufficient history (start)
        features = features.dropna(subset=["target", "return_20d", "sma_50_ratio"])

        return features

    except Exception as exc:
        logger.error("%s: feature computation failed — %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_all_features(
    tickers: list[str] = TICKERS_ALL,
    data_dir: Path = RAW_MARKET_DIR,
    output_path: Path = FEATURES_MARKET_PATH,
) -> pd.DataFrame:
    """Build and save the combined market feature matrix for all tickers.

    Args:
        tickers: List of ticker symbols to process.
        data_dir: Directory containing raw OHLCV CSVs.
        output_path: Path to save the combined Parquet file.

    Returns:
        Combined DataFrame with MultiIndex (Date, ticker) or flat with both cols.
    """
    # Load VIX once
    vix_path = data_dir / "^VIX.csv"
    vix_df = None
    if vix_path.exists():
        vix_df = pd.read_csv(vix_path, index_col="Date", parse_dates=True).sort_index()
        logger.info("VIX loaded: %d rows", len(vix_df))
    else:
        logger.warning("VIX file not found — vix_level feature will be NaN")

    all_frames: list[pd.DataFrame] = []
    failures: list[str] = []
    total = len(tickers)

    for i, ticker in enumerate(tickers, start=1):
        logger.info("[%d/%d] Computing features for %s …", i, total, ticker)
        feat = build_ticker_features(ticker, data_dir=data_dir, vix_df=vix_df)
        if feat is not None:
            all_frames.append(feat)
        else:
            failures.append(ticker)

    if not all_frames:
        raise RuntimeError("No features computed — check that raw data exists in data/raw/market_data/")

    combined = pd.concat(all_frames, axis=0)
    combined = combined.sort_values(["ticker", combined.index.name or "Date"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path)

    logger.info(
        "Feature matrix saved: %d rows × %d cols → %s",
        len(combined),
        len(combined.columns),
        output_path,
    )
    if failures:
        logger.warning("Failed tickers: %s", failures)

    # Print class distribution
    dist = combined["target"].value_counts(normalize=True).mul(100).round(1)
    logger.info("Target distribution:\n%s", dist.to_string())

    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build market feature matrix")
    parser.add_argument("--test", action="store_true", help="3-ticker smoke test")
    args = parser.parse_args()

    tickers = TEST_TICKERS if args.test else TICKERS_ALL
    build_all_features(tickers=tickers)
