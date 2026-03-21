"""app.py — Streamlit entry point for the Financial Market Predictor."""

import sys
from pathlib import Path

# Ensure project root is on the path when launched via `streamlit run app.py`
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="Financial Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.app.pages import predictor, model_analysis, eda_explorer, about, rag_chat

PAGES = {
    "Live Prediction": predictor,
    "Model Analysis": model_analysis,
    "EDA Explorer": eda_explorer,
    "News Chatbot (RAG)": rag_chat,
    "About": about,
}

with st.sidebar:
    st.title("Financial Market Predictor")
    st.caption("ZHAW AI Applications — 2026")
    st.divider()
    page_name = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    st.caption(
        "Disclaimer: Research prototype only. "
        "Not financial advice."
    )

PAGES[page_name].render()
