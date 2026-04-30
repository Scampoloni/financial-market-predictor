"""
build_smoke_dataset.py — Create a small reproducible dataset for quick end-to-end runs.

Builds a smoke-test subset from existing raw data and saves it to data/smoke/:
- market_data: AAPL, MSFT, NVDA, ^VIX, ^GSPC (fixed 3-month window)
- news: AAPL, MSFT, NVDA headlines in the same window (if available)

Usage:
    python scripts/build_smoke_dataset.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import RAW_MARKET_DIR, RAW_NEWS_DIR

SMOKE_DIR = ROOT / "data" / "smoke"
SMOKE_MARKET_DIR = SMOKE_DIR / "market_data"
SMOKE_NEWS_DIR = SMOKE_DIR / "news"

TICKERS = ["AAPL", "MSFT", "NVDA"]
INDICES = ["^VIX", "^GSPC"]
DATE_START = "2025-01-01"
DATE_END = "2025-03-31"


def _slice_market_csv(ticker: str) -> None:
    src = RAW_MARKET_DIR / f"{ticker}.csv"
    if not src.exists():
        print(f"Missing market CSV: {src}")
        return
    df = pd.read_csv(src, index_col="Date", parse_dates=True).sort_index()
    sliced = df.loc[DATE_START:DATE_END]
    if sliced.empty:
        print(f"No rows in range for {ticker}")
        return
    SMOKE_MARKET_DIR.mkdir(parents=True, exist_ok=True)
    out = SMOKE_MARKET_DIR / f"{ticker}.csv"
    sliced.to_csv(out)
    print(f"Saved market slice: {out} ({len(sliced)} rows)")


def _slice_news_parquet(ticker: str) -> None:
    src = RAW_NEWS_DIR / f"{ticker}.parquet"
    if not src.exists():
        print(f"Missing news parquet: {src}")
        # Create empty placeholder
        SMOKE_NEWS_DIR.mkdir(parents=True, exist_ok=True)
        out = SMOKE_NEWS_DIR / f"{ticker}.parquet"
        pd.DataFrame(columns=["title", "summary", "published", "link", "source", "ticker"]).to_parquet(out, index=False)
        return
    df = pd.read_parquet(src)
    if df.empty or "published" not in df.columns:
        print(f"No usable news rows for {ticker}")
        return
    df["published"] = pd.to_datetime(df["published"], utc=True)
    mask = (df["published"] >= DATE_START) & (df["published"] <= DATE_END)
    sliced = df.loc[mask].copy()
    if sliced.empty:
        print(f"No news in range for {ticker}")
        SMOKE_NEWS_DIR.mkdir(parents=True, exist_ok=True)
        out = SMOKE_NEWS_DIR / f"{ticker}.parquet"
        pd.DataFrame(columns=["title", "summary", "published", "link", "source", "ticker"]).to_parquet(out, index=False)
        return
    SMOKE_NEWS_DIR.mkdir(parents=True, exist_ok=True)
    out = SMOKE_NEWS_DIR / f"{ticker}.parquet"
    sliced.to_parquet(out, index=False)
    print(f"Saved news slice: {out} ({len(sliced)} rows)")


def main() -> None:
    print("Building smoke dataset...")
    for ticker in TICKERS + INDICES:
        _slice_market_csv(ticker)
    for ticker in TICKERS:
        _slice_news_parquet(ticker)
    print("Done. Smoke dataset is in data/smoke/.")


if __name__ == "__main__":
    main()
