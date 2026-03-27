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

# ── Premium Dark Theme CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Inter font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0a0e17;
    color: #f0f6fc;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Block container ── */
.block-container {
    padding: 1.5rem 2rem 2rem;
    max-width: 1280px;
}

/* ── Typography ── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: #f0f6fc;
}
h1 { font-size: 1.75rem !important; }
h2 { font-size: 1.4rem !important; }
h3 { font-size: 1.15rem !important; }

/* ── Navigation bar ── */
.nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.8rem 0;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 1.5rem;
}
.nav-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.2rem;
    font-weight: 700;
    color: #f0f6fc;
    letter-spacing: -0.02em;
}
.nav-brand span { color: #3b82f6; }
/* nav-links removed — using st.tabs() */

/* ── Cards ── */
.glass-card {
    background: linear-gradient(145deg, #111827, #0f172a);
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 1rem;
    transition: border-color 0.2s ease;
}
.glass-card:hover { border-color: #334155; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1e40af, #3b82f6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(59,130,246,0.25) !important;
}
.stButton > button:hover {
    box-shadow: 0 4px 16px rgba(59,130,246,0.4) !important;
    transform: translateY(-1px);
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #111827 !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    color: #f0f6fc !important;
}
.stSelectbox label { color: #94a3b8 !important; }

/* radio nav removed — using st.tabs() */

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, #111827, #0f172a);
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid #1e293b;
    gap: 0;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    border-radius: 8px 8px 0 0;
    font-weight: 500;
    padding: 0.6rem 1.4rem;
    font-size: 0.9rem;
}
.stTabs [aria-selected="true"] {
    background: #1e293b;
    color: #60a5fa;
    border-bottom: 2px solid #3b82f6;
}

/* ── Tables / DataFrames ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e293b;
    border-radius: 10px;
    overflow: hidden;
}

/* ── Divider ── */
hr { border-color: #1e293b !important; margin: 1rem 0 !important; }

/* ── Status container ── */
[data-testid="stStatusWidget"] {
    background: #111827 !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
}

/* ── Plotly chart background ── */
.stPlotlyChart { border-radius: 12px; overflow: hidden; }

/* ── Spinner ── */
.stSpinner > div { color: #3b82f6 !important; }

/* ── Pill buttons for chart timeframe ── */
.pill-group {
    display: flex;
    gap: 4px;
    margin: 0.5rem 0 0.8rem;
}
.pill-btn {
    padding: 6px 16px;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #94a3b8;
    background: #111827;
    border: 1px solid #1e293b;
    cursor: pointer;
    transition: all 0.15s ease;
    text-decoration: none;
}
.pill-btn:hover { background: #1e293b; color: #e2e8f0; }
.pill-btn.active {
    background: #1e3a5f;
    color: #60a5fa;
    border-color: #2563eb;
}

/* ── Prediction cards ── */
.pred-card {
    border-radius: 16px;
    padding: 28px 24px;
    text-align: center;
    transition: transform 0.2s ease;
}
.pred-card:hover { transform: translateY(-2px); }
.pred-card-up {
    background: linear-gradient(145deg, #052e16, #14532d);
    border: 1px solid #166534;
}
.pred-card-down {
    background: linear-gradient(145deg, #450a0a, #7f1d1d);
    border: 1px solid #991b1b;
}

/* ── Headline card ── */
.headline-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 6px;
    transition: border-color 0.15s ease;
}
.headline-card:hover { border-color: #334155; }

/* ── Info tag ── */
.info-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
    background: #1e293b;
    color: #94a3b8;
}
.info-tag.green { background: #052e16; color: #4ade80; }
.info-tag.red { background: #450a0a; color: #f87171; }
.info-tag.blue { background: #172554; color: #60a5fa; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e17; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }

/* ── Disclaimer ── */
.disclaimer {
    color: #475569;
    font-size: 0.75rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #1e293b;
}
</style>
""", unsafe_allow_html=True)

from src.app.pages import predictor, about, model_analysis, rag_chat

# ── Branding ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="nav-bar"><div class="nav-brand"><span>📈</span> Market Predictor</div></div>',
    unsafe_allow_html=True,
)

# ── Navigation via native Streamlit tabs ─────────────────────────────────────
tab_pred, tab_compare, tab_analysis, tab_chat, tab_about = st.tabs(
    ["Prediction", "Compare", "Analysis", "News Chat", "About"]
)

with tab_pred:
    predictor.render()
with tab_compare:
    predictor.render_compare()
with tab_analysis:
    model_analysis.render()
with tab_chat:
    rag_chat.render()
with tab_about:
    about.render()
