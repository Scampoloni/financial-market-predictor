"""predictor.py — Live prediction page."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta

from src.app.utils import get_predictor, TICKERS_SORTED
from src.config import TICKER_SECTOR_MAP


# ── Colour palette ───────────────────────────────────────────────────────────
_PRED_COLOR  = {"UP": "#3fb950", "DOWN": "#f85149"}
_PRED_ICON   = {"UP": "▲", "DOWN": "▼"}
_PRED_LABEL  = {"UP": "Bullish", "DOWN": "Bearish"}


def _price_chart(ticker: str) -> None:
    """Draw a clean 3-month candlestick-style line chart."""
    end   = datetime.today()
    start = end - timedelta(days=90)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].squeeze()
    color = "#3fb950" if float(close.iloc[-1]) >= float(close.iloc[0]) else "#f85149"

    fig, ax = plt.subplots(figsize=(9, 3))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    ax.plot(close.index, close.values, color=color, linewidth=1.8)
    ax.fill_between(close.index, close.values, close.values.min(), alpha=0.12, color=color)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.get_xticklabels(), color="#8b949e", fontsize=8, rotation=0)
    plt.setp(ax.get_yticklabels(), color="#8b949e", fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.tick_params(colors="#30363d", length=0)
    ax.grid(axis="y", color="#21262d", linewidth=0.7)
    fig.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _prob_gauge(probs: dict, prediction: str) -> None:
    """Horizontal stacked bar showing DOWN / UP probabilities."""
    order  = ["DOWN", "UP"]
    colors = [_PRED_COLOR[c] for c in order]
    vals   = [probs.get(c, 0) for c in order]

    fig, ax = plt.subplots(figsize=(7, 0.7))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    left = 0
    for val, color, label in zip(vals, colors, order):
        ax.barh(0, val, left=left, color=color,
                height=0.55,
                alpha=1.0 if label == prediction else 0.35)
        if val >= 0.08:
            ax.text(left + val / 2, 0, f"{val:.0%}",
                    ha="center", va="center", color="white",
                    fontsize=9, fontweight="bold")
        left += val

    ax.set_xlim(0, 1)
    ax.axis("off")
    fig.tight_layout(pad=0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render() -> None:
    # ── Ticker selector ──────────────────────────────────────────────────────
    col_sel, col_btn, col_info = st.columns([2, 1, 3])

    with col_sel:
        ticker = st.selectbox(
            "Ticker",
            TICKERS_SORTED,
            index=TICKERS_SORTED.index("AAPL") if "AAPL" in TICKERS_SORTED else 0,
            label_visibility="collapsed",
        )

    with col_btn:
        run = st.button("Predict", type="primary", use_container_width=True)

    with col_info:
        sector = TICKER_SECTOR_MAP.get(ticker, "")
        st.markdown(
            f"<p style='margin:0.4rem 0 0;color:#8b949e;font-size:0.9rem'>"
            f"<b style='color:#e6edf3'>{ticker}</b> &nbsp;·&nbsp; {sector}</p>",
            unsafe_allow_html=True,
        )

    # ── Price chart (always shown) ───────────────────────────────────────────
    _price_chart(ticker)

    if not run:
        st.markdown(
            "<p style='color:#8b949e;text-align:center;margin-top:1rem'>"
            "Click <b>Predict</b> to generate a 5-day forecast</p>",
            unsafe_allow_html=True,
        )
        return

    # ── Run prediction ───────────────────────────────────────────────────────
    with st.spinner("Running model pipeline…"):
        try:
            result = get_predictor().predict(ticker)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

    pred       = result["prediction"]
    conf       = result["confidence"]
    probs      = result["probabilities"]
    mkt_date   = result["market_date"]
    n_headlines = result["n_headlines"]

    st.markdown("<hr style='margin:0.5rem 0'>", unsafe_allow_html=True)

    # ── Result row ───────────────────────────────────────────────────────────
    col_pred, col_conf, col_head, col_date = st.columns(4)

    with col_pred:
        color = _PRED_COLOR[pred]
        icon  = _PRED_ICON[pred]
        label = _PRED_LABEL[pred]
        st.markdown(
            f"<div style='background:#161b22;border:1px solid {color}44;"
            f"border-radius:12px;padding:1rem;text-align:center'>"
            f"<div style='font-size:2.2rem;color:{color}'>{icon}</div>"
            f"<div style='font-size:1.4rem;font-weight:700;color:{color}'>{pred}</div>"
            f"<div style='color:#8b949e;font-size:0.85rem'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_conf:
        st.metric("Confidence", f"{conf:.1%}")

    with col_head:
        st.metric("Headlines used", n_headlines)

    with col_date:
        st.metric("Market date", mkt_date)

    # ── Probability bar ──────────────────────────────────────────────────────
    st.markdown(
        "<p style='color:#8b949e;font-size:0.8rem;margin:1rem 0 0.2rem'>Class probabilities</p>",
        unsafe_allow_html=True,
    )
    _prob_gauge(probs, pred)

    # ── Headlines ────────────────────────────────────────────────────────────
    st.markdown("<hr style='margin:1rem 0 0.5rem'>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:#8b949e;font-size:0.85rem;margin:0'>Recent headlines for <b>{ticker}</b></p>",
        unsafe_allow_html=True,
    )

    try:
        from src.data_collection.news_scraper import load_ticker_news
        news = load_ticker_news(ticker)
        if not news.empty:
            news["published"] = pd.to_datetime(news["published"]).dt.strftime("%b %d")
            for _, row in news.head(8).iterrows():
                st.markdown(
                    f"<div style='padding:0.45rem 0;border-bottom:1px solid #21262d'>"
                    f"<span style='color:#8b949e;font-size:0.78rem'>{row['published']} &nbsp;·&nbsp; "
                    f"{row.get('source','')}</span><br>"
                    f"<span style='font-size:0.92rem'>{row['title']}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No saved headlines for this ticker.")
    except FileNotFoundError:
        st.caption("No saved news file found.")

    # ── Disclaimer ───────────────────────────────────────────────────────────
    st.markdown(
        "<p style='color:#484f58;font-size:0.75rem;margin-top:1.5rem'>"
        "Research prototype — not financial advice. Past patterns do not guarantee future returns.</p>",
        unsafe_allow_html=True,
    )
