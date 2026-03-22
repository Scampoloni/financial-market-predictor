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

@st.cache_resource(show_spinner="Loading prediction models...")
def get_predictor():
    """Lazy-load and cache the LivePredictor with all available models."""
    from src.models.predict import LivePredictor
    p = LivePredictor()
    # Pre-load all available horizon models
    for h in p.available_horizons:
        p.load_model(h)
    p.load_nlp_pca()
    p.load_cv_pca()
    return p


@st.cache_data(show_spinner="Loading market features...")
def load_market_features() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_MARKET_PATH)


@st.cache_data(show_spinner="Loading NLP features...")
def load_nlp_features() -> pd.DataFrame:
    if FEATURES_NLP_PATH.exists():
        return pd.read_parquet(FEATURES_NLP_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner="Loading CV features...")
def load_cv_features() -> pd.DataFrame:
    if FEATURES_CV_PATH.exists():
        return pd.read_parquet(FEATURES_CV_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner="Loading ablation results...")
def load_ablation_results() -> dict:
    if ABLATION_RESULTS_PATH.exists():
        with open(ABLATION_RESULTS_PATH) as f:
            return json.load(f)
    return {}


TICKERS_SORTED = sorted(TICKERS_ALL)
