"""predictor.py — Live prediction interface page."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from src.app.utils import (
    get_predictor,
    load_nlp_features,
    prediction_badge,
    confidence_bar,
    TICKERS_SORTED,
)
from src.config import TICKER_SECTOR_MAP


def render() -> None:
    st.header("Live Prediction")
    st.markdown(
        "Select a ticker to generate a next-day **UP / DOWN / SIDEWAYS** forecast "
        "using the full Config C model (market + NLP + chart embeddings)."
    )

    col1, col2 = st.columns([2, 3])

    with col1:
        ticker = st.selectbox(
            "Ticker",
            TICKERS_SORTED,
            index=TICKERS_SORTED.index("AAPL") if "AAPL" in TICKERS_SORTED else 0,
        )
        sector = TICKER_SECTOR_MAP.get(ticker, "—")
        st.caption(f"Sector: **{sector}**")
        run = st.button("Run Prediction", type="primary", use_container_width=True)

    if not run:
        st.info("Select a ticker and click **Run Prediction**.")
        return

    with st.spinner(f"Running pipeline for {ticker}…"):
        try:
            predictor = get_predictor()
            result = predictor.predict(ticker)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

    # --- Result card ---
    st.divider()
    col_res, col_prob = st.columns([1, 1])

    with col_res:
        st.subheader("Prediction")
        st.markdown(prediction_badge(result["prediction"]), unsafe_allow_html=True)
        st.caption(f"As of market date: {result['market_date']}")
        st.markdown("**Confidence**")
        confidence_bar(result["confidence"])
        st.caption(f"Headlines used: {result['n_headlines']}")

    with col_prob:
        st.subheader("Class Probabilities")
        probs = result["probabilities"]
        labels = ["DOWN", "SIDEWAYS", "UP"]
        values = [probs.get(l, 0) for l in labels]
        colors = ["#ef4444", "#f59e0b", "#22c55e"]

        fig, ax = plt.subplots(figsize=(4, 2.5))
        bars = ax.barh(labels, values, color=colors, height=0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        for bar, val in zip(bars, values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1%}", va="center", fontsize=9)
        ax.set_title(f"{ticker} — Next-Day Probabilities")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # --- Recent news headlines ---
    st.divider()
    st.subheader("Recent Headlines Used")
    nlp_df = load_nlp_features()
    if not nlp_df.empty and "ticker" in nlp_df.columns:
        try:
            from src.data_collection.news_scraper import load_ticker_news
            news = load_ticker_news(ticker)
            if not news.empty:
                show = news[["published", "title", "source"]].head(10).copy()
                show["published"] = pd.to_datetime(show["published"]).dt.strftime("%Y-%m-%d")
                st.dataframe(show, use_container_width=True, hide_index=True)
            else:
                st.caption("No saved headlines for this ticker.")
        except FileNotFoundError:
            st.caption("No saved news file for this ticker.")
    else:
        st.caption("NLP feature data not available.")

    # --- Disclaimer ---
    st.divider()
    st.warning(
        "**Disclaimer:** This is a research prototype. Predictions are for educational "
        "purposes only and do not constitute financial advice."
    )
