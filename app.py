"""app.py — Financial Market Predictor · Streamlit entry point."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.6rem 2rem;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: .85; }

/* Selectbox */
.stSelectbox > div > div {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    color: #e6edf3;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid #30363d; gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8b949e;
    border-radius: 6px 6px 0 0;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: #1f6feb22;
    color: #58a6ff;
    border-bottom: 2px solid #1f6feb;
}

/* Divider */
hr { border-color: #30363d; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

from src.app.pages import predictor, about

# ── Navigation ──────────────────────────────────────────────────────────────
col_logo, col_nav = st.columns([1, 4])
with col_logo:
    st.markdown("### 📈 Market Predictor")
with col_nav:
    nav = st.radio(
        "", ["Prediction", "About"],
        horizontal=True,
        label_visibility="collapsed",
    )

st.markdown("<hr style='margin:0 0 1.5rem 0'>", unsafe_allow_html=True)

if nav == "Prediction":
    predictor.render()
else:
    about.render()
