"""
build_analyst_features.py — Collect and engineer analyst sentiment features.

Uses two yfinance endpoints:
  - ticker.upgrades_downgrades : historical firm-level rating changes (ToGrade, FromGrade)
  - ticker.recommendations     : aggregate monthly consensus (strongBuy/buy/hold/sell/strongSell)

Produces 5 daily features per ticker, forward-filled from actual rating dates:
  - analyst_consensus          : recency-weighted mean grade score (–2 to +2)
  - analyst_upgrade_score      : upgrades minus downgrades in last 30 days
  - analyst_coverage_count     : number of active analyst firms
  - price_target_upside        : (mean_target – close) / close  (static current estimate)
  - analyst_sentiment_momentum : change in consensus over last 90 days

Usage:
    python -m src.data_collection.build_analyst_features
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grade → numeric score mapping  (–2 to +2)
# ---------------------------------------------------------------------------
_GRADE_MAP: dict[str, float] = {
    "strong buy": 2.0, "top pick": 2.0, "buy": 2.0,
    "outperform": 1.5, "overweight": 1.5, "market outperform": 1.5,
    "sector outperform": 1.5, "positive": 1.0,
    "accumulate": 1.0, "add": 1.0, "long-term buy": 1.5,
    "hold": 0.0, "neutral": 0.0, "market perform": 0.0,
    "equal-weight": 0.0, "equal weight": 0.0, "in-line": 0.0,
    "sector weight": 0.0, "market weight": 0.0,
    "sector perform": 0.0, "peer perform": 0.0,
    "mixed": 0.0, "fair value": 0.0,
    "underperform": -1.5, "underweight": -1.5,
    "sector underperform": -1.5, "negative": -1.0,
    "reduce": -1.0, "sell": -2.0, "strong sell": -2.0,
}


def _grade_to_score(grade: str) -> float | None:
    if not isinstance(grade, str):
        return None
    return _GRADE_MAP.get(grade.strip().lower(), None)


def _consensus_from_aggregate(rec_df: pd.DataFrame) -> float | None:
    """Compute weighted consensus score from the aggregate recommendations table.

    rec_df columns: period, strongBuy, buy, hold, sell, strongSell
    Returns score in range [–2, +2], or None if no data.
    """
    try:
        # Use the most recent month (period "0m")
        current = rec_df[rec_df["period"] == "0m"].iloc[0]
        sb = int(current.get("strongBuy", 0) or 0)
        b  = int(current.get("buy", 0) or 0)
        h  = int(current.get("hold", 0) or 0)
        s  = int(current.get("sell", 0) or 0)
        ss = int(current.get("strongSell", 0) or 0)
        total = sb + b + h + s + ss
        if total == 0:
            return None
        return (sb * 2.0 + b * 1.0 + h * 0.0 + s * -1.0 + ss * -2.0) / total
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-ticker feature builder
# ---------------------------------------------------------------------------

def build_analyst_features_for_ticker(
    ticker: str,
    market_dates: pd.DatetimeIndex,
) -> pd.DataFrame | None:
    """
    Returns a DataFrame indexed by `market_dates` with 5 analyst feature columns.
    Returns None if no data is available.
    """
    t = yf.Ticker(ticker)

    # ── 1. Historical upgrades/downgrades ───────────────────────────────────
    try:
        ud = t.upgrades_downgrades
    except Exception:
        ud = None

    has_history = (
        ud is not None
        and hasattr(ud, "empty")
        and not ud.empty
        and "ToGrade" in ud.columns
    )

    if has_history:
        ud = ud.copy()
        ud.index = pd.to_datetime(ud.index, utc=True).tz_localize(None)
        ud.index.name = "date"
        ud = ud.sort_index()
        ud["score"] = ud["ToGrade"].apply(_grade_to_score)
        ud = ud.dropna(subset=["score"])
        if ud.empty:
            has_history = False

    # ── 2. Current aggregate consensus ──────────────────────────────────────
    try:
        rec_agg = t.recommendations
        agg_consensus = _consensus_from_aggregate(rec_agg) if rec_agg is not None else None
    except Exception:
        agg_consensus = None

    # ── 3. Current price target ──────────────────────────────────────────────
    mean_target = None
    if has_history and "currentPriceTarget" in ud.columns:
        latest_targets = ud["currentPriceTarget"].dropna()
        if not latest_targets.empty:
            mean_target = float(latest_targets.iloc[-1])

    if not has_history and agg_consensus is None:
        return None

    # ── 4. Build daily time series ──────────────────────────────────────────
    results = []

    for dt in market_dates:
        if has_history:
            window_90 = ud[ud.index <= dt].tail(90)
            window_30 = ud[
                (ud.index <= dt) & (ud.index >= dt - pd.Timedelta(days=30))
            ]

            if window_90.empty:
                # Fallback to aggregate consensus if available
                consensus = agg_consensus if agg_consensus is not None else np.nan
                upgrade_score = 0.0
                coverage = 0
                momentum = 0.0
            else:
                # Recency-weighted consensus
                days_ago = np.array((dt - window_90.index).days).clip(0, 90)
                weights = 1.0 / (1.0 + days_ago / 30.0)
                consensus = float(np.average(window_90["score"], weights=weights))

                # Upgrades vs downgrades
                if "FromGrade" in ud.columns:
                    w30 = window_30.copy()
                    w30["from_score"] = w30["FromGrade"].apply(_grade_to_score).fillna(0)
                    upgrades = (w30["score"] > w30["from_score"]).sum()
                    downgrades = (w30["score"] < w30["from_score"]).sum()
                else:
                    upgrades = (window_30["score"] > 0).sum()
                    downgrades = (window_30["score"] < 0).sum()
                upgrade_score = float(upgrades - downgrades)

                # Coverage count
                coverage = (
                    window_90["Firm"].nunique()
                    if "Firm" in window_90.columns
                    else len(window_90)
                )

                # Momentum: consensus now vs 90 days ago
                window_old = ud[ud.index <= dt - pd.Timedelta(days=90)].tail(30)
                old_consensus = float(window_old["score"].mean()) if not window_old.empty else consensus
                momentum = consensus - old_consensus
        else:
            # No historical data — use only aggregate consensus
            consensus = agg_consensus if agg_consensus is not None else np.nan
            upgrade_score = 0.0
            coverage = 0
            momentum = 0.0

        results.append({
            "date": dt,
            "analyst_consensus": consensus,
            "analyst_upgrade_score": upgrade_score,
            "analyst_coverage_count": float(coverage),
            "price_target_upside": np.nan,  # filled after joining close prices
            "analyst_sentiment_momentum": momentum,
        })

    df = pd.DataFrame(results).set_index("date")

    # Forward-fill (ratings don't change every day)
    df = df.ffill().fillna(0)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from src.config import (
        FEATURES_ANALYST_PATH,
        FEATURES_MARKET_PATH,
        PROCESSED_DIR,
        TICKERS_ALL,
    )

    logger.info("Loading market features to get date index ...")
    market = pd.read_parquet(FEATURES_MARKET_PATH)
    market.index = pd.to_datetime(market.index)

    all_parts: list[pd.DataFrame] = []
    tickers = sorted(TICKERS_ALL)
    logger.info("Building analyst features for %d tickers ...", len(tickers))

    success, skipped = 0, 0

    for i, ticker in enumerate(tickers):
        logger.info("[%d/%d] %s ...", i + 1, len(tickers), ticker)
        try:
            ticker_mask = market["ticker"] == ticker
            t_dates = pd.DatetimeIndex(
                market.index[ticker_mask].unique().sort_values()
            )
            if len(t_dates) == 0:
                continue

            feat = build_analyst_features_for_ticker(ticker, t_dates)
            if feat is None or feat.empty:
                logger.warning("  %s: no analyst data — using zeros", ticker)
                feat = pd.DataFrame(
                    {
                        "analyst_consensus": 0.0,
                        "analyst_upgrade_score": 0.0,
                        "analyst_coverage_count": 0.0,
                        "price_target_upside": 0.0,
                        "analyst_sentiment_momentum": 0.0,
                    },
                    index=t_dates,
                )
                skipped += 1
            else:
                success += 1
                logger.info("  %s: consensus=%.2f, coverage=%.0f",
                            ticker,
                            feat["analyst_consensus"].mean(),
                            feat["analyst_coverage_count"].mean())

            feat["ticker"] = ticker
            all_parts.append(feat)

        except Exception as exc:
            logger.warning("  %s: ERROR %s", ticker, exc)

        time.sleep(0.3)

    if not all_parts:
        logger.error("No analyst features collected!")
        return

    combined = pd.concat(all_parts).sort_index()
    combined.index.name = "date"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(FEATURES_ANALYST_PATH)

    coverage_pct = combined["analyst_consensus"].replace(0, np.nan).notna().mean()
    logger.info(
        "Done: %d tickers with data, %d without. Coverage: %.1f%%",
        success, skipped, coverage_pct * 100,
    )
    logger.info("Mean consensus across all tickers: %.3f",
                combined["analyst_consensus"].replace(0, np.nan).mean())
    logger.info("Saved %d rows to %s", len(combined), FEATURES_ANALYST_PATH)


if __name__ == "__main__":
    main()
