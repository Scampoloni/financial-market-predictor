"""
news_scraper.py — Collect financial news headlines from RSS feeds and NewsAPI.

Headlines are matched to tickers by company name / symbol mentions.
Output: one Parquet file per ticker in data/raw/news/{ticker}.parquet

Usage:
    python -m src.data_collection.news_scraper --test      # AAPL, MSFT, NVDA only
    python -m src.data_collection.news_scraper             # all tickers
    python -m src.data_collection.news_scraper --ticker TSLA  # single ticker
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import pandas as pd
from rapidfuzz import fuzz

from src.config import (
    NEWS_API_KEY,
    NEWS_DEDUP_THRESHOLD,
    RAW_NEWS_DIR,
    RSS_FEEDS,
    TICKERS_ALL,
    TICKER_SECTOR_MAP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TEST_TICKERS = ["AAPL", "MSFT", "NVDA"]

# Company name → ticker mapping for entity matching
# Covers common name variants (e.g. "Apple" → AAPL)
COMPANY_NAMES: dict[str, str] = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "amd": "AMD",
    "advanced micro": "AMD",
    "intel": "INTC",
    "qualcomm": "QCOM",
    "salesforce": "CRM",
    "oracle": "ORCL",
    "adobe": "ADBE",
    "servicenow": "NOW",
    "intuit": "INTU",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "goldman sachs": "GS",
    "bank of america": "BAC",
    "morgan stanley": "MS",
    "visa": "V",
    "mastercard": "MA",
    "berkshire": "BRK-B",
    "citigroup": "C",
    "citi": "C",
    "wells fargo": "WFC",
    "american express": "AXP",
    "amex": "AXP",
    "blackrock": "BLK",
    "charles schwab": "SCHW",
    "schwab": "SCHW",
    "aig": "AIG",
    "american international": "AIG",
    "metlife": "MET",
    "prudential": "PRU",
    "allstate": "ALL",
    "travelers": "TRV",
    "johnson & johnson": "JNJ",
    "johnson and johnson": "JNJ",
    "pfizer": "PFE",
    "unitedhealth": "UNH",
    "united health": "UNH",
    "abbvie": "ABBV",
    "merck": "MRK",
    "eli lilly": "LLY",
    "lilly": "LLY",
    "thermo fisher": "TMO",
    "abbott": "ABT",
    "amgen": "AMGN",
    "gilead": "GILD",
    "coca-cola": "KO",
    "coca cola": "KO",
    "coke": "KO",
    "pepsico": "PEP",
    "pepsi": "PEP",
    "mcdonald": "MCD",
    "nike": "NKE",
    "starbucks": "SBUX",
    "procter & gamble": "PG",
    "procter and gamble": "PG",
    "walmart": "WMT",
    "costco": "COST",
    "target": "TGT",
    "home depot": "HD",
    "exxon": "XOM",
    "exxonmobil": "XOM",
    "chevron": "CVX",
    "conocophillips": "COP",
    "conoco": "COP",
    "schlumberger": "SLB",
    "eog resources": "EOG",
    "phillips 66": "PSX",
    "boeing": "BA",
    "caterpillar": "CAT",
    "general electric": "GE",
    "3m": "MMM",
    "honeywell": "HON",
    "ups": "UPS",
    "united parcel": "UPS",
    "fedex": "FDX",
    "raytheon": "RTX",
    "lockheed": "LMT",
    "lockheed martin": "LMT",
}


# ---------------------------------------------------------------------------
# RSS helpers
# ---------------------------------------------------------------------------

def _fetch_rss(url: str, timeout: int = 10) -> list[dict]:
    """Fetch and parse a single RSS feed.

    Args:
        url: RSS feed URL.
        timeout: Request timeout in seconds.

    Returns:
        List of entry dicts with keys: title, summary, published, link, source.
    """
    try:
        feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})
        entries = []
        for entry in feed.entries:
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            if published:
                pub_dt = datetime(*published[:6], tzinfo=timezone.utc)
            else:
                pub_dt = datetime.now(timezone.utc)

            entries.append(
                {
                    "title": entry.get("title", "").strip(),
                    "summary": entry.get("summary", "").strip(),
                    "published": pub_dt,
                    "link": entry.get("link", ""),
                    "source": feed.feed.get("title", url),
                }
            )
        return entries
    except Exception as exc:
        logger.warning("RSS fetch failed for %s: %s", url, exc)
        return []


def fetch_all_rss() -> pd.DataFrame:
    """Fetch headlines from all configured RSS feeds.

    Returns:
        DataFrame with columns [title, summary, published, link, source].
    """
    all_entries: list[dict] = []
    for name, url in RSS_FEEDS.items():
        logger.info("Fetching RSS: %s", name)
        entries = _fetch_rss(url)
        all_entries.extend(entries)
        time.sleep(0.3)

    df = pd.DataFrame(all_entries)
    if df.empty:
        return df

    df["published"] = pd.to_datetime(df["published"], utc=True)
    df = df.drop_duplicates(subset=["title"])
    df = df.sort_values("published", ascending=False).reset_index(drop=True)
    logger.info("RSS: %d unique headlines fetched", len(df))
    return df


# ---------------------------------------------------------------------------
# NewsAPI helpers (optional — requires API key)
# ---------------------------------------------------------------------------

def fetch_newsapi(ticker: str, company_name: str, page_size: int = 100) -> pd.DataFrame:
    """Fetch headlines from NewsAPI for a specific ticker.

    Requires NEWS_API_KEY in .env. Returns empty DataFrame if key is missing.
    The free tier returns the 100 most recent articles (last ~24-48h).
    Date filtering (from/to) requires a paid NewsAPI plan.

    Args:
        ticker: Ticker symbol.
        company_name: Company name to search for.
        page_size: Max results per request (NewsAPI free tier: 100).

    Returns:
        DataFrame with columns [title, summary, published, link, source].
    """
    if not NEWS_API_KEY:
        return pd.DataFrame()

    try:
        import requests

        params = {
            "q": f'"{company_name}" OR "{ticker}"',
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "apiKey": NEWS_API_KEY,
        }
        resp = requests.get(
            "https://newsapi.org/v2/everything", params=params, timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("articles", [])
        logger.info(
            "%s: NewsAPI returned %d articles (total available: %s)",
            ticker, len(articles), data.get("totalResults", "?"),
        )

        rows = [
            {
                "title": (a.get("title") or "").strip(),
                "summary": (a.get("description") or "").strip(),
                "published": pd.to_datetime(a.get("publishedAt"), utc=True),
                "link": a.get("url", ""),
                "source": a.get("source", {}).get("name", "NewsAPI"),
            }
            for a in articles
            if (a.get("title") or "").strip()
        ]
        return pd.DataFrame(rows)
    except Exception as exc:
        logger.warning("NewsAPI failed for %s: %s", ticker, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Ticker matching
# ---------------------------------------------------------------------------

def _mentions_ticker(text: str, ticker: str) -> bool:
    """Check if text explicitly mentions the ticker symbol.

    Args:
        text: Headline or summary text.
        ticker: Ticker symbol (e.g. 'AAPL').

    Returns:
        True if the ticker symbol appears as a word in the text.
    """
    text_upper = text.upper()
    # Match ticker as whole word (surrounded by non-alpha or string boundary)
    import re
    pattern = rf"\b{re.escape(ticker)}\b"
    return bool(re.search(pattern, text_upper))


def _mentions_company(text: str, ticker: str) -> bool:
    """Check if text mentions any known company name for a given ticker.

    Args:
        text: Headline or summary text.
        ticker: Ticker symbol.

    Returns:
        True if a known company name variant appears in the text.
    """
    text_lower = text.lower()
    for name, t in COMPANY_NAMES.items():
        if t == ticker and name in text_lower:
            return True
    return False


def match_headlines_to_ticker(
    df: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """Filter headlines relevant to a specific ticker.

    Matching strategy (either condition is sufficient):
    1. Ticker symbol appears verbatim in title/summary
    2. Known company name variant appears in title/summary

    Args:
        df: DataFrame with columns [title, summary, ...].
        ticker: Ticker symbol.

    Returns:
        Filtered DataFrame with a 'ticker' column added.
    """
    if df.empty:
        return pd.DataFrame()

    mask = df.apply(
        lambda row: (
            _mentions_ticker(row["title"], ticker)
            or _mentions_ticker(row.get("summary", ""), ticker)
            or _mentions_company(row["title"], ticker)
            or _mentions_company(row.get("summary", ""), ticker)
        ),
        axis=1,
    )
    matched = df[mask].copy()
    matched["ticker"] = ticker
    return matched


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_headlines(df: pd.DataFrame, threshold: int = NEWS_DEDUP_THRESHOLD) -> pd.DataFrame:
    """Remove near-duplicate headlines using fuzzy string matching.

    Iterates through headlines and drops any that are >= threshold% similar
    to an already-kept headline.

    Args:
        df: DataFrame with a 'title' column, sorted by date descending.
        threshold: RapidFuzz ratio threshold (0–100). Default from config.

    Returns:
        Deduplicated DataFrame.
    """
    if df.empty or len(df) <= 1:
        return df

    titles = df["title"].tolist()
    keep_indices = [0]

    for i in range(1, len(titles)):
        is_dup = any(
            fuzz.ratio(titles[i], titles[keep_indices[j]]) >= threshold
            for j in range(len(keep_indices))
        )
        if not is_dup:
            keep_indices.append(i)

    return df.iloc[keep_indices].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-ticker collection
# ---------------------------------------------------------------------------

def collect_ticker_news(
    ticker: str,
    rss_df: pd.DataFrame,
    output_dir: Path = RAW_NEWS_DIR,
) -> pd.DataFrame:
    """Collect, match, and save news for a single ticker.

    Combines RSS headlines (already fetched) with optional NewsAPI results,
    deduplicates, and saves to a Parquet file.

    Args:
        ticker: Ticker symbol.
        rss_df: Pre-fetched RSS headlines DataFrame.
        output_dir: Directory to save {ticker}.parquet.

    Returns:
        Cleaned DataFrame for the ticker (may be empty if no matches).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{ticker}.parquet"

    # 1. Match from RSS
    matched = match_headlines_to_ticker(rss_df, ticker)

    # 2. Supplement with NewsAPI if key is available
    company_name = next(
        (name for name, t in COMPANY_NAMES.items() if t == ticker), ticker
    )
    api_df = fetch_newsapi(ticker, company_name)
    if not api_df.empty:
        api_df["ticker"] = ticker
        matched = pd.concat([matched, api_df], ignore_index=True)

    if matched.empty:
        logger.warning("%s: no headlines found", ticker)
        # Save empty file so downstream code knows we tried
        pd.DataFrame(
            columns=["title", "summary", "published", "link", "source", "ticker"]
        ).to_parquet(out_path, index=False)
        return pd.DataFrame()

    # 3. Sort + deduplicate
    matched = matched.sort_values("published", ascending=False).reset_index(drop=True)
    before = len(matched)
    matched = deduplicate_headlines(matched)
    after = len(matched)

    matched.to_parquet(out_path, index=False)
    logger.info(
        "%s: %d headlines saved (%d duplicates removed) → %s",
        ticker,
        after,
        before - after,
        out_path.name,
    )
    return matched


def collect_all(
    tickers: list[str],
    output_dir: Path = RAW_NEWS_DIR,
) -> dict[str, pd.DataFrame]:
    """Collect news for all tickers.

    Fetches RSS feeds once, then matches headlines to each ticker individually.
    Supplements with NewsAPI if key is configured.

    Args:
        tickers: List of ticker symbols.
        output_dir: Directory to save Parquet files.

    Returns:
        Dict mapping ticker → DataFrame (only tickers with results).
    """
    logger.info("Fetching RSS feeds …")
    rss_df = fetch_all_rss()

    results: dict[str, pd.DataFrame] = {}
    total = len(tickers)

    for i, ticker in enumerate(tickers, start=1):
        logger.info("[%d/%d] Matching news for %s …", i, total, ticker)
        df = collect_ticker_news(ticker, rss_df, output_dir=output_dir)
        if not df.empty:
            results[ticker] = df

    covered = len(results)
    logger.info(
        "News collection done. %d/%d tickers have at least one headline.",
        covered,
        total,
    )
    if covered < total:
        missing = [t for t in tickers if t not in results]
        logger.warning(
            "No headlines found for: %s — sector news will be used as fallback "
            "during feature engineering.",
            missing,
        )
    return results


def load_ticker_news(ticker: str, data_dir: Path = RAW_NEWS_DIR) -> pd.DataFrame:
    """Load saved news Parquet file for a ticker.

    Args:
        ticker: Ticker symbol.
        data_dir: Directory containing Parquet files.

    Returns:
        DataFrame with columns [title, summary, published, link, source, ticker].

    Raises:
        FileNotFoundError: If no news file exists for the ticker.
    """
    path = data_dir / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No news file for {ticker} at {path}")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect financial news headlines")
    parser.add_argument("--test", action="store_true", help=f"3-ticker smoke test: {TEST_TICKERS}")
    parser.add_argument("--ticker", type=str, help="Single ticker to collect")
    args = parser.parse_args()

    if args.ticker:
        tickers = [args.ticker.upper()]
    elif args.test:
        tickers = TEST_TICKERS
    else:
        tickers = TICKERS_ALL

    collect_all(tickers)
