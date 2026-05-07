"""about.py — Project info page."""

from __future__ import annotations

import streamlit as st

_MUTED = "#64748b"


def render() -> None:
    # ── 1. Project Overview ──────────────────────────────────────────────────
    st.markdown(
        "<h2 style='margin-bottom:2px'>Financial Market Predictor</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='color:{_MUTED};font-size:0.92rem;margin-bottom:1.2rem'>"
        "ZHAW AI Applications FS2026 — Academic Research Prototype</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="glass-card" style="line-height:1.7;color:#94a3b8;font-size:0.92rem">'
        "<b style='color:#f0f6fc'>Scientific question:</b> Can publicly available market data, "
        "financial news sentiment, and candlestick chart patterns predict short-term stock price "
        "direction better than a random baseline?<br><br>"
        "We approach this as a binary classification problem — predicting whether a stock's "
        "closing price will be <b style='color:#10b981'>UP</b> or "
        "<b style='color:#ef4444'>DOWN</b> over 5- and 21-trading-day horizons — using a "
        "temporal train/validation/test split to prevent data leakage."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── 2. Three blocks side by side ─────────────────────────────────────────
    st.markdown("<h3 style='margin-top:1.2rem;margin-bottom:0.6rem'>The Three AI Blocks</h3>",
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            '<div class="glass-card" style="text-align:center;height:100%">'
            '<div style="font-size:1.6rem;margin-bottom:6px">📊</div>'
            '<div style="font-weight:700;color:#10b981;font-size:1rem;margin-bottom:8px">Block 1: ML</div>'
            '<div style="color:#94a3b8;font-size:0.85rem;line-height:1.6">'
            'LightGBM + RandomForest on <b style="color:#e2e8f0">28 market features</b><br>'
            '<span style="color:#475569">RSI, MACD, Bollinger Bands, ATR, VIX, volume ratio, '
            'SMA/EMA ratios, sector dummies</span><br><br>'
            '<span style="color:#475569;font-size:0.8rem">Source: Yahoo Finance (yfinance)</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<div class="glass-card" style="text-align:center;height:100%">'
            '<div style="font-size:1.6rem;margin-bottom:6px">📰</div>'
            '<div style="font-weight:700;color:#f59e0b;font-size:1rem;margin-bottom:8px">Block 2: NLP</div>'
            '<div style="color:#94a3b8;font-size:0.85rem;line-height:1.6">'
            'FinBERT + VADER on <b style="color:#e2e8f0">8,552 news headlines</b><br>'
            '<span style="color:#475569">Sentiment score, momentum, surprise, dispersion, '
            'news volume z-score, sector fallback. '
            'RAG chatbot for interactive Q&amp;A.</span><br><br>'
            '<span style="color:#475569;font-size:0.8rem">Source: RSS feeds + yfinance news</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            '<div class="glass-card" style="text-align:center;height:100%">'
            '<div style="font-size:1.6rem;margin-bottom:6px">📈</div>'
            '<div style="font-weight:700;color:#60a5fa;font-size:1rem;margin-bottom:8px">Block 3: CV</div>'
            '<div style="color:#94a3b8;font-size:0.85rem;line-height:1.6">'
            'EfficientNet-B0 on <b style="color:#e2e8f0">61,640+ chart images</b><br>'
            '<span style="color:#475569">Transfer learning (1280-dim) → PCA (10 dims). '
            'Fine-tuning on chart labels via '
            '<code style="font-size:0.78rem">scripts/finetune_cnn.py</code>.</span><br><br>'
            '<span style="color:#475569;font-size:0.8rem">Generated via mplfinance</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    # ── 3. Data Sources ───────────────────────────────────────────────────────
    st.markdown("<h3 style='margin-top:1.4rem;margin-bottom:0.6rem'>Data Sources</h3>",
                unsafe_allow_html=True)

    rows = [
        ("#10b981", "Yahoo Finance",    "5yr daily OHLCV for 67 S&P 500 tickers (2020–2026), ~550k rows"),
        ("#f59e0b", "Financial News",   "RSS feeds (Reuters, MarketWatch) + yfinance news API, 8,552 headlines"),
        ("#60a5fa", "Candlestick Charts","61,640+ images generated via mplfinance (30-day windows, step=2)"),
        ("#8b5cf6", "Pre-trained Models","ProsusAI/finbert (HuggingFace) · EfficientNet-B0 (torchvision ImageNet)"),
    ]
    table_rows = "".join(
        f'<tr style="border-bottom:1px solid #1e293b">'
        f'<td style="padding:10px 14px;color:{c};font-weight:700;font-size:0.9rem;width:160px">{name}</td>'
        f'<td style="padding:10px 14px;color:#94a3b8;font-size:0.88rem">{desc}</td></tr>'
        for c, name, desc in rows
    )
    st.markdown(
        f'<div class="glass-card" style="padding:0;overflow:hidden">'
        f'<table style="width:100%;border-collapse:collapse">'
        f'<tbody>{table_rows}</tbody></table></div>',
        unsafe_allow_html=True,
    )

    # ── 4. Methodology ────────────────────────────────────────────────────────
    st.markdown("<h3 style='margin-top:1.4rem;margin-bottom:0.6rem'>Methodology</h3>",
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            '<div class="glass-card">'
            '<div style="color:#60a5fa;font-weight:700;font-size:0.9rem;margin-bottom:8px">'
            'Temporal Train/Test Split</div>'
            '<div style="color:#94a3b8;font-size:0.85rem;line-height:1.7">'
            '<b style="color:#e2e8f0">Train:</b> 2020–2024 H1<br>'
            '<b style="color:#e2e8f0">Validation:</b> 2024 H2<br>'
            '<b style="color:#e2e8f0">Test:</b> 2025 (held-out)<br><br>'
            'TimeSeriesSplit (5-fold) for cross-validation. '
            'No future data ever seen during training.'
            '</div></div>',
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            '<div class="glass-card">'
            '<div style="color:#60a5fa;font-weight:700;font-size:0.9rem;margin-bottom:8px">'
            'Ablation Study Design</div>'
            '<div style="color:#94a3b8;font-size:0.85rem;line-height:1.7">'
            '<b style="color:#94a3b8">Config A:</b> Market features only (baseline)<br>'
            '<b style="color:#8b5cf6">Config B:</b> + NLP sentiment features<br>'
            '<b style="color:#10b981">Config C:</b> + CV chart embeddings<br><br>'
            'Binary classification: UP/DOWN direction more actionable '
            'and class-balanced than a continuous return target.'
            '</div></div>',
            unsafe_allow_html=True,
        )

    # ── 5. Feature Glossary ───────────────────────────────────────────────────
    st.markdown("<h3 style='margin-top:1.4rem;margin-bottom:0.4rem'>Feature Glossary</h3>",
                unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{_MUTED};font-size:0.85rem;margin-bottom:0.6rem'>"
        "Expand each block to see what each feature captures and how to interpret it.</p>",
        unsafe_allow_html=True,
    )

    _MARKET_FEATURES = [
        ("RSI (14)", "Relative Strength Index. >70 = overbought (potential reversal down), <30 = oversold."),
        ("MACD / Signal / Histogram", "Momentum oscillator (12-26 EMA). Positive histogram = bullish momentum."),
        ("Bollinger Band Width", "Price channel width. Wide = high volatility; narrow = potential breakout."),
        ("ATR (14)", "Average True Range — absolute daily price swing. Higher = more volatile."),
        ("20D Volatility", "Rolling std dev of log-returns. High = uncertain environment."),
        ("Volume Ratio", "Today's volume ÷ 20-day avg. >1.5 = unusual activity spike."),
        ("VIX Level", "CBOE fear gauge. >25 = fear; <18 = complacency."),
        ("Return 1D / 5D / 20D", "Price return over 1-, 5-, and 20-day horizons."),
        ("Price/SMA20 & Price/SMA50", "Distance of price from 20- and 50-day moving averages."),
        ("Sector Dummies", "One-hot encoding of the GICS sector (11 total)."),
    ]
    _NLP_FEATURES = [
        ("FinBERT Sentiment", "Finance-domain BERT (ProsusAI/finbert). Score: -1 (negative) to +1 (positive)."),
        ("VADER Compound", "Rule-based sentiment. Fast, no GPU required. Used as second NLP signal."),
        ("Sentiment Momentum", "3-day rolling mean. Trend in news tone is more predictive than absolute level."),
        ("Sentiment Surprise (Z-score)", "Deviation from 20-day mean. A sudden shift signals new information."),
        ("Sentiment × Volume", "Interaction term: high-volume negative news amplifies the bearish signal."),
        ("Sentiment Dispersion", "Disagreement across headlines. High = uncertain market narrative."),
        ("News Volume Z-score", "Spike in headline count — may precede market-moving events."),
        ("Sector / Market Fallback", "Sector- or market-wide sentiment used when ticker has no direct coverage (>99% of days)."),
    ]
    _CV_FEATURES = [
        ("EfficientNet-B0 (1280-dim)", "Global average pool output from ImageNet-pretrained CNN. Encodes visual patterns."),
        ("PCA (10 components)", "Dimensionality reduction. Captures main variance across 41k chart embeddings."),
        ("chart_available flag", "Indicator for whether a chart was generated for this date."),
        ("Fine-tuned backbone", "Optional: run scripts/finetune_cnn.py to adapt EfficientNet to chart UP/DOWN labels."),
    ]

    def _glossary_rows(items, color):
        for name, desc in items:
            st.markdown(
                f'<div style="display:flex;gap:16px;padding:8px 0;border-bottom:1px solid #1e293b">'
                f'<div style="min-width:190px;color:{color};font-weight:600;font-size:0.84rem">{name}</div>'
                f'<div style="color:#94a3b8;font-size:0.84rem;line-height:1.5">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    with st.expander("📊 Market Block — 28 features", expanded=False):
        _glossary_rows(_MARKET_FEATURES, "#4a90d9")

    with st.expander("📰 NLP Block — 24 features", expanded=False):
        _glossary_rows(_NLP_FEATURES, "#8b5cf6")

    with st.expander("📈 CV Block — 10 features", expanded=False):
        _glossary_rows(_CV_FEATURES, "#60a5fa")

    # ── 6. Technology Stack ───────────────────────────────────────────────────
    st.markdown("<h3 style='margin-top:1.4rem;margin-bottom:0.6rem'>Technology Stack</h3>",
                unsafe_allow_html=True)

    tech_rows = [
        ("📊", "#10b981", "ML / Numeric",    "scikit-learn · LightGBM · XGBoost · Optuna"),
        ("📰", "#8b5cf6", "NLP",             "HuggingFace Transformers (FinBERT) · NLTK (VADER) · sentence-transformers"),
        ("📈", "#60a5fa", "Computer Vision", "PyTorch · torchvision (EfficientNet-B0) · mplfinance"),
        ("💬", "#f59e0b", "RAG Chatbot",     "sentence-transformers · cosine similarity · Gemini 1.5 Flash / OpenAI fallback"),
        ("📦", "#94a3b8", "Data",            "pandas · numpy · yfinance · feedparser · Plotly"),
        ("🌐", "#94a3b8", "App",             "Streamlit · Plotly"),
    ]
    tech_html = "".join(
        f'<div style="display:flex;align-items:center;gap:12px;padding:10px 16px;'
        f'border-bottom:1px solid #1e293b">'
        f'<div style="font-size:1.2rem">{icon}</div>'
        f'<div style="min-width:140px;color:{color};font-weight:700;font-size:0.88rem">{cat}</div>'
        f'<div style="color:#94a3b8;font-size:0.85rem">{libs}</div></div>'
        for icon, color, cat, libs in tech_rows
    )
    st.markdown(
        f'<div class="glass-card" style="padding:0;overflow:hidden">{tech_html}</div>',
        unsafe_allow_html=True,
    )

    # ── 7. Ethical Considerations ─────────────────────────────────────────────
    st.warning(
        "⚠️ Research prototype — not financial advice. "
        "Academic exercise for ZHAW AI Applications (FS2026). "
        "Do NOT use predictions for real trading. "
        "Survivorship bias: only currently listed S&P 500 stocks included. "
        "~0.50 F1 = wrong ~half the time."
    )

    st.markdown(
        "<p class='disclaimer'>"
        "Research prototype — not financial advice. Predictions are for educational purposes only "
        "and do not constitute investment recommendations.</p>",
        unsafe_allow_html=True,
    )
