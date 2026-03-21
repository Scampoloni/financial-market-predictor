"""utils.py — Shared Streamlit helper functions."""

from __future__ import annotations

import json

import pandas as pd

import streamlit as st

from pathlib import Path

from src.config import (
    FEATURES_MARKET_PATH,
    FEATURES_NLP_PATH,
    FEATURES_CV_PATH,
    TICKERS_ALL,
    TICKER_SECTOR_MAP,
    PROCESSED_DIR,
)

ABLATION_RESULTS_PATH = PROCESSED_DIR / "ablation_results.json"


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading prediction model…")
def get_predictor():
    """Lazy-load and cache the LivePredictor."""
    from src.models.predict import LivePredictor
    p = LivePredictor()
    p._load_main_model()
    p._load_nlp_pca()
    p._load_cv_pca()
    return p


@st.cache_data(show_spinner="Loading market features…")
def load_market_features() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_MARKET_PATH)


@st.cache_data(show_spinner="Loading NLP features…")
def load_nlp_features() -> pd.DataFrame:
    if FEATURES_NLP_PATH.exists():
        return pd.read_parquet(FEATURES_NLP_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner="Loading CV features…")
def load_cv_features() -> pd.DataFrame:
    if FEATURES_CV_PATH.exists():
        return pd.read_parquet(FEATURES_CV_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner="Loading ablation results…")
def load_ablation_results() -> dict:
    if ABLATION_RESULTS_PATH.exists():
        with open(ABLATION_RESULTS_PATH) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def prediction_badge(prediction: str) -> str:
    """Return coloured HTML badge for a prediction label."""
    colors = {"UP": "#22c55e", "DOWN": "#ef4444", "SIDEWAYS": "#f59e0b"}
    icons  = {"UP": "▲", "DOWN": "▼", "SIDEWAYS": "◆"}
    c = colors.get(prediction, "#6b7280")
    i = icons.get(prediction, "?")
    return (
        f'<span style="background:{c};color:white;padding:4px 12px;'
        f'border-radius:6px;font-weight:bold;font-size:1.1em;">{i} {prediction}</span>'
    )


def confidence_bar(confidence: float) -> None:
    """Render a colour-coded confidence progress bar."""
    pct = int(confidence * 100)
    color = "#22c55e" if confidence >= 0.5 else "#f59e0b" if confidence >= 0.38 else "#ef4444"
    st.markdown(
        f"""
        <div style="background:#e5e7eb;border-radius:6px;height:18px;width:100%;">
          <div style="background:{color};width:{pct}%;height:18px;border-radius:6px;"></div>
        </div>
        <p style="margin:2px 0 8px;font-size:0.85em;color:#6b7280;">
          Confidence: {pct}%
        </p>
        """,
        unsafe_allow_html=True,
    )


TICKERS_SORTED = sorted(TICKERS_ALL)
