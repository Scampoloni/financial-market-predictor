"""
market_collector.py — Download 5-year OHLCV data from Yahoo Finance.

Usage:
    python -m src.data_collection.market_collector           # full run (all tickers)
    python -m src.data_collection.market_collector --test    # 3-ticker smoke test
"""

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import (
    DATA_START_DATE,
    DATA_END_DATE,
    MARKET_INDICES,
    MIN_STOCK_PRICE,
    RAW_MARKET_DIR,
    TICKERS_ALL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TEST_TICKERS = ["AAPL", "MSFT", "NVDA"]


def _clean_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Validate and clean raw yfinance OHLCV data.

    Args:
        df: Raw DataFrame from yfinance.
        ticker: Ticker symbol (used for logging).

    Returns:
        Cleaned DataFrame indexed by Date with columns
        [Open, High, Low, Close, Volume].
    """
    # Flatten multi-level columns if present (yfinance >=0.2 style)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{ticker}: missing columns {missing}")

    df = df[list(required)].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)  # strip tz, keep date
    df.index.name = "Date"
    df.sort_index(inplace=True)

    # Drop rows where OHLC are NaN; skip volume filter for indices (Volume=0 is normal)
    is_index = ticker.startswith("^")
    if not is_index:
        df = df[df["Volume"] > 0]
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    # Filter penny stocks (close < MIN_STOCK_PRICE on most days) — skip for indices
    if not ticker.startswith("^"):
        median_close = df["Close"].median()
        if median_close < MIN_STOCK_PRICE:
            raise ValueError(
                f"{ticker}: median close {median_close:.2f} < {MIN_STOCK_PRICE} (penny stock)"
            )

    return df


def download_ticker(
    ticker: str,
    start: str = DATA_START_DATE,
    end: str = DATA_END_DATE,
    output_dir: Path = RAW_MARKET_DIR,
) -> pd.DataFrame | None:
    """Download OHLCV data for a single ticker and save to CSV.

    Args:
        ticker: Yahoo Finance ticker symbol.
        start: Start date string 'YYYY-MM-DD'.
        end: End date string 'YYYY-MM-DD'.
        output_dir: Directory where the CSV is saved.

    Returns:
        Cleaned DataFrame if successful, None on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{ticker}.csv"

    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            logger.warning("%s: no data returned — skipping", ticker)
            return None

        df = _clean_ohlcv(raw, ticker)
        df.to_csv(out_path)
        logger.info(
            "%s: %d rows saved → %s",
            ticker,
            len(df),
            out_path.relative_to(out_path.parents[3]),
        )
        return df

    except Exception as exc:
        logger.error("%s: download failed — %s", ticker, exc)
        return None


def collect_all(
    tickers: list[str],
    start: str = DATA_START_DATE,
    end: str = DATA_END_DATE,
    output_dir: Path = RAW_MARKET_DIR,
    delay_seconds: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV data for all tickers.

    Args:
        tickers: List of ticker symbols.
        start: Start date string.
        end: End date string.
        output_dir: Directory to save CSV files.
        delay_seconds: Pause between requests to be polite to the API.

    Returns:
        Dict mapping ticker → cleaned DataFrame (only successful downloads).
    """
    total = len(tickers)
    results: dict[str, pd.DataFrame] = {}
    failures: list[str] = []

    logger.info("Starting download for %d tickers  [%s → %s]", total, start, end)

    for i, ticker in enumerate(tickers, start=1):
        logger.info("[%d/%d] Downloading %s …", i, total, ticker)
        df = download_ticker(ticker, start=start, end=end, output_dir=output_dir)
        if df is not None:
            results[ticker] = df
        else:
            failures.append(ticker)

        if i < total:
            time.sleep(delay_seconds)

    # Also download market indices (VIX, S&P 500) needed as features
    for idx_ticker in MARKET_INDICES:
        logger.info("Downloading index %s …", idx_ticker)
        df = download_ticker(idx_ticker, start=start, end=end, output_dir=output_dir)
        if df is not None:
            results[idx_ticker] = df
        else:
            failures.append(idx_ticker)

    logger.info(
        "Done. %d/%d tickers downloaded successfully.",
        len(results),
        total + len(MARKET_INDICES),
    )
    if failures:
        logger.warning("Failed tickers: %s", failures)

    return results


def load_ticker(ticker: str, data_dir: Path = RAW_MARKET_DIR) -> pd.DataFrame:
    """Load a previously saved ticker CSV from disk.

    Args:
        ticker: Ticker symbol.
        data_dir: Directory containing the CSVs.

    Returns:
        DataFrame with DatetimeIndex.

    Raises:
        FileNotFoundError: If the CSV does not exist.
    """
    path = data_dir / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No data file found for {ticker} at {path}")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect market OHLCV data")
    parser.add_argument(
        "--test",
        action="store_true",
        help=f"Smoke-test with 3 tickers: {TEST_TICKERS}",
    )
    parser.add_argument("--start", default=DATA_START_DATE, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=DATA_END_DATE, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    tickers = TEST_TICKERS if args.test else TICKERS_ALL
    collect_all(tickers, start=args.start, end=args.end)
