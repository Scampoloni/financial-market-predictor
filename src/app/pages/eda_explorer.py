"""eda_explorer.py — Dataset insights and EDA visualization page."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.app.utils import (
    load_market_features,
    load_nlp_features,
    load_cv_features,
    TICKERS_SORTED,
)
from src.config import TARGET_CLASSES, TICKER_SECTOR_MAP


def render() -> None:
    st.header("EDA Explorer")
    st.markdown("Explore the training dataset across all three feature blocks.")

    tab_market, tab_nlp, tab_cv = st.tabs(["Market Features", "NLP Features", "CV Features"])

    # ------------------------------------------------------------------
    # Tab 1: Market features
    # ------------------------------------------------------------------
    with tab_market:
        st.subheader("Market Feature Dataset")
        df = load_market_features()
        df.index = pd.to_datetime(df.index)

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Features", f"{len(df.columns):,}")
        col3.metric("Tickers", df["ticker"].nunique() if "ticker" in df.columns else "—")

        # Target distribution
        if "target" in df.columns:
            st.subheader("Target Class Distribution")
            counts = df["target"].value_counts().reindex(TARGET_CLASSES, fill_value=0)
            fig, ax = plt.subplots(figsize=(6, 3))
            colors = ["#ef4444", "#f59e0b", "#22c55e"]
            bars = ax.bar(counts.index, counts.values, color=colors, width=0.5)
            ax.set_ylabel("Count")
            ax.set_title("Training Label Distribution")
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                        f"{val:,}\n({val/len(df):.1%})", ha="center", fontsize=9)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # Ticker selector — price history
        st.subheader("Price History by Ticker")
        ticker = st.selectbox("Select ticker", TICKERS_SORTED, key="eda_ticker")
        if "ticker" in df.columns and "close" in df.columns:
            tdf = df[df["ticker"] == ticker]["close"].dropna()
            if not tdf.empty:
                fig2, ax2 = plt.subplots(figsize=(10, 3))
                ax2.plot(tdf.index, tdf.values, linewidth=1)
                ax2.set_title(f"{ticker} — Closing Price")
                ax2.set_ylabel("Price (USD)")
                fig2.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

        # Sector breakdown
        if "ticker" in df.columns and "target" in df.columns:
            st.subheader("Target Distribution by Sector")
            df["sector"] = df["ticker"].map(TICKER_SECTOR_MAP)
            sector_target = df.groupby(["sector", "target"]).size().unstack(fill_value=0)
            sector_target = sector_target.reindex(columns=TARGET_CLASSES, fill_value=0)
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            sector_target.plot(kind="bar", ax=ax3, color=["#ef4444", "#f59e0b", "#22c55e"],
                               width=0.7, alpha=0.85)
            ax3.set_title("Label Distribution by Sector")
            ax3.set_xlabel("")
            ax3.set_ylabel("Count")
            ax3.tick_params(axis="x", rotation=30)
            ax3.legend(title="Label")
            fig3.tight_layout()
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

    # ------------------------------------------------------------------
    # Tab 2: NLP features
    # ------------------------------------------------------------------
    with tab_nlp:
        st.subheader("NLP Feature Dataset")
        nlp = load_nlp_features()
        if nlp.empty:
            st.warning("NLP features not found. Run `python -m src.features.nlp_features`.")
            return

        nlp.index = pd.to_datetime(nlp.index)
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(nlp):,}")
        col2.metric("NLP Features", f"{len(nlp.columns):,}")
        covered = int((nlp["news_volume_1d"] > 0).sum()) if "news_volume_1d" in nlp.columns else 0
        col3.metric("News-covered rows", f"{covered:,} ({covered/len(nlp):.1%})")

        if "finbert_sentiment" in nlp.columns:
            st.subheader("FinBERT Sentiment Distribution")
            fig4, ax4 = plt.subplots(figsize=(8, 3))
            sent = nlp["finbert_sentiment"].dropna()
            ax4.hist(sent, bins=60, color="#4e79a7", alpha=0.8, edgecolor="white")
            ax4.axvline(0, color="red", linestyle="--", linewidth=1)
            ax4.set_xlabel("FinBERT Score")
            ax4.set_ylabel("Count")
            ax4.set_title("Distribution of Daily FinBERT Sentiment")
            fig4.tight_layout()
            st.pyplot(fig4, use_container_width=True)
            plt.close(fig4)

        if "news_volume_1d" in nlp.columns and "ticker" in nlp.columns:
            st.subheader("News Volume by Ticker (top 20)")
            vol = (nlp.groupby("ticker")["news_volume_1d"]
                   .sum().sort_values(ascending=False).head(20))
            fig5, ax5 = plt.subplots(figsize=(10, 3.5))
            vol.plot(kind="bar", ax=ax5, color="#ff7f0e", alpha=0.85)
            ax5.set_title("Total News Volume by Ticker")
            ax5.set_ylabel("Sum of daily headline counts")
            ax5.tick_params(axis="x", rotation=45)
            fig5.tight_layout()
            st.pyplot(fig5, use_container_width=True)
            plt.close(fig5)

    # ------------------------------------------------------------------
    # Tab 3: CV features
    # ------------------------------------------------------------------
    with tab_cv:
        st.subheader("CV Feature Dataset")
        cv = load_cv_features()
        if cv.empty:
            st.warning("CV features not found. Run `python -m src.features.cv_features`.")
            return

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(cv):,}")
        col2.metric("CV Features", f"{len(cv.columns):,}")
        if "chart_available" in cv.columns:
            covered_cv = int(cv["chart_available"].sum())
            col3.metric("Chart-covered rows", f"{covered_cv:,} ({covered_cv/len(cv):.1%})")

        pca_cols = [c for c in cv.columns if c.startswith("chart_embed_pca_")]
        if len(pca_cols) >= 2:
            st.subheader("PCA Embedding Scatter (PC1 vs PC2)")
            sample = cv[cv["chart_available"] == 1][pca_cols[:2]].dropna().sample(
                min(2000, len(cv)), random_state=42
            ) if "chart_available" in cv.columns else cv[pca_cols[:2]].dropna().sample(
                min(2000, len(cv)), random_state=42
            )
            fig6, ax6 = plt.subplots(figsize=(7, 5))
            ax6.scatter(sample.iloc[:, 0], sample.iloc[:, 1], s=5, alpha=0.4, color="#2ca02c")
            ax6.set_xlabel("PC1")
            ax6.set_ylabel("PC2")
            ax6.set_title("EfficientNet-B0 Chart Embeddings — PCA PC1 vs PC2")
            fig6.tight_layout()
            st.pyplot(fig6, use_container_width=True)
            plt.close(fig6)
