"""
chart_generator.py — Generate 30-day candlestick chart images for each ticker-day.

Produces one 224x224 PNG per (ticker, date) window, saved under:
    data/raw/charts/{ticker}/{YYYY-MM-DD}.png

The image covers the 30 trading days ending on the given date.
Volume bars are included. Axes and labels are stripped so the CNN
sees only price-action patterns (no date/price leakage).

Usage:
    python -m src.data_collection.chart_generator --test   # AAPL, 5 charts
    python -m src.data_collection.chart_generator          # all tickers, all dates
    python -m src.data_collection.chart_generator --ticker MSFT
"""

import logging
from pathlib import Path

import pandas as pd
import mplfinance as mpf
import matplotlib
matplotlib.use("Agg")   # headless rendering

from src.config import (
    CHART_IMAGE_SIZE,
    CHART_WINDOW_DAYS,
    RAW_CHARTS_DIR,
    RAW_MARKET_DIR,
    TICKERS_ALL,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

TEST_TICKERS = ["AAPL", "MSFT", "NVDA"]
_CHART_STYLE = mpf.make_mpf_style(
    base_mpf_style="nightclouds",
    facecolor="black",
    edgecolor="black",
    figcolor="black",
    gridcolor="black",
)
_CHART_KWARGS = dict(
    type="candle",
    style=_CHART_STYLE,
    volume=True,
    axisoff=True,        # no axes — CNN sees pure pattern
    tight_layout=True,
    returnfig=True,
)


def _load_ohlcv(ticker: str) -> pd.DataFrame:
    """Load OHLCV CSV for a ticker and return a clean DataFrame.

    Args:
        ticker: Ticker symbol.

    Returns:
        DataFrame with DatetimeIndex and columns [Open, High, Low, Close, Volume].
        Empty if file not found.
    """
    csv_path = RAW_MARKET_DIR / f"{ticker}.csv"
    if not csv_path.exists():
        logger.warning("%s: no CSV at %s", ticker, csv_path)
        return pd.DataFrame()

    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    # Keep only the OHLCV columns mplfinance expects (capitalised)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        logger.warning("%s: missing columns %s", ticker, missing)
        return pd.DataFrame()

    df = df[needed].copy()
    df = df[df["Volume"] > 0]   # drop non-trading days / index rows
    df.sort_index(inplace=True)
    return df


def generate_charts_for_ticker(
    ticker: str,
    output_dir: Path = RAW_CHARTS_DIR,
    window: int = CHART_WINDOW_DAYS,
    image_size: tuple[int, int] = CHART_IMAGE_SIZE,
    step: int = 5,
    force: bool = False,
) -> int:
    """Generate candlestick chart PNGs for every 5th trading day for a ticker.

    Generates one chart per `step` trading days (default: every 5 days) to
    keep disk usage manageable. The chart covers the `window` trading days
    ending on that date.

    Args:
        ticker: Ticker symbol.
        output_dir: Root directory for chart images.
        window: Number of trading days per chart (default 30).
        image_size: Output image size in pixels (width, height).
        step: Generate a chart every N trading days.
        force: If True, regenerate existing charts.

    Returns:
        Number of charts generated.
    """
    df = _load_ohlcv(ticker)
    if df.empty or len(df) < window:
        logger.warning("%s: insufficient data (%d rows)", ticker, len(df))
        return 0

    ticker_dir = output_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    dpi = image_size[0] / 2.24   # ~100 dpi for 224px → figure size 2.24 inches
    fig_size = (image_size[0] / dpi, image_size[1] / dpi)

    dates = df.index[window - 1 :: step]   # one chart per `step` days, after warmup
    generated = 0

    for date in dates:
        out_path = ticker_dir / f"{date.date()}.png"
        if out_path.exists() and not force:
            continue

        window_df = df.loc[:date].tail(window)
        if len(window_df) < window:
            continue

        try:
            fig, axes = mpf.plot(
                window_df,
                figsize=fig_size,
                **_CHART_KWARGS,
            )
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
            matplotlib.pyplot.close(fig)
            generated += 1
        except Exception as exc:
            logger.debug("%s %s: chart error — %s", ticker, date.date(), exc)

    logger.info("%s: %d charts generated in %s", ticker, generated, ticker_dir)
    return generated


def generate_all_charts(
    tickers: list[str] = TICKERS_ALL,
    output_dir: Path = RAW_CHARTS_DIR,
    step: int = 2,
    force: bool = False,
) -> dict[str, int]:
    """Generate charts for all tickers.

    Args:
        tickers: List of ticker symbols.
        output_dir: Root directory for chart images.
        step: Generate one chart every N trading days (default 2 for ~50% coverage).
        force: If True, regenerate existing charts.

    Returns:
        Dict mapping ticker → number of charts generated.
    """
    results: dict[str, int] = {}
    for i, ticker in enumerate(tickers, 1):
        logger.info("[%d/%d] Generating charts for %s ...", i, len(tickers), ticker)
        results[ticker] = generate_charts_for_ticker(
            ticker, output_dir=output_dir, step=step, force=force
        )
    total = sum(results.values())
    logger.info("Done. %d charts generated across %d tickers.", total, len(tickers))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate candlestick chart images")
    parser.add_argument("--test", action="store_true", help="3-ticker smoke test (5 charts each)")
    parser.add_argument("--ticker", type=str, help="Single ticker")
    parser.add_argument("--force", action="store_true", help="Regenerate existing charts")
    parser.add_argument("--step", type=int, default=2, help="Chart every N trading days")
    args = parser.parse_args()

    if args.ticker:
        generate_charts_for_ticker(args.ticker, step=args.step, force=args.force)
    elif args.test:
        for t in TEST_TICKERS:
            generate_charts_for_ticker(t, step=60, force=args.force)  # ~25 charts per ticker
    else:
        generate_all_charts(step=args.step, force=args.force)
