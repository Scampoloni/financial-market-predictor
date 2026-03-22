"""predictor.py — Live prediction page with Plotly candlestick chart."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from html import escape as html_escape

from src.app.utils import get_predictor, TICKERS_SORTED
from src.config import COMPANY_KEYWORDS, SPAM_KEYWORDS, TICKER_SECTOR_MAP

# ── Palette ──────────────────────────────────────────────────────────────────
_UP_COLOR = "#10b981"
_DOWN_COLOR = "#ef4444"
_AMBER = "#f59e0b"
_MUTED = "#64748b"

_CHART_PERIODS = {
    "5D": 5, "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "MAX": 9999,
}

# Feature display names and direction heuristics
_FEAT_LABELS: dict[str, str] = {
    "rsi_14": "RSI (14)", "volatility_20d": "20D Volatility", "volume_ratio": "Volume Ratio",
    "return_1d": "1D Return", "return_5d": "5D Return", "return_20d": "20D Return",
    "vix_level": "VIX Level", "macd_hist": "MACD Histogram", "macd": "MACD",
    "atr_14": "ATR (14)", "bb_width": "Bollinger Width", "bb_upper_dist": "BB Upper Dist",
    "bb_lower_dist": "BB Lower Dist", "sma_20_ratio": "Price/SMA20", "sma_50_ratio": "Price/SMA50",
    "ema_12_ratio": "Price/EMA12", "finbert_sentiment": "FinBERT Sentiment",
    "vader_sentiment": "VADER Sentiment", "sentiment_momentum": "Sentiment Momentum",
    "sentiment_shift_3d": "Sentiment Shift 3D", "sentiment_surprise": "Sentiment Surprise",
    "sentiment_dispersion": "Sentiment Dispersion", "news_volume_zscore": "News Volume Z",
    "sentiment_x_volume": "Sentiment × Volume",
}


def _feat_direction(name: str, value: float) -> tuple[str, str]:
    """Return (label, color) for a feature value's directional meaning."""
    n = name.lower()
    if "vix" in n:
        return ("fear", _DOWN_COLOR) if value > 25 else ("calm", _UP_COLOR) if value < 18 else ("normal", _MUTED)
    if "rsi" in n:
        return ("overbought", _DOWN_COLOR) if value > 70 else ("oversold", _UP_COLOR) if value < 30 else ("neutral", _MUTED)
    if "sentiment" in n or "vader" in n:
        return ("bullish", _UP_COLOR) if value > 0.05 else ("bearish", _DOWN_COLOR) if value < -0.05 else ("neutral", _MUTED)
    if "return" in n or "macd_hist" in n:
        return ("bullish", _UP_COLOR) if value > 0 else ("bearish", _DOWN_COLOR)
    if "volume_ratio" in n:
        return ("high vol", _AMBER) if value > 1.5 else ("low vol", _MUTED) if value < 0.7 else ("normal", _MUTED)
    return ("", _MUTED)


def _signal_strength(conf: float) -> tuple[str, str]:
    """Return (label, color) for confidence threshold."""
    if conf >= 0.65:
        return "Strong Signal", _UP_COLOR
    if conf >= 0.55:
        return "Moderate Signal", "#3b82f6"
    return "Uncertain Signal", _AMBER


# ── News relevance filter ────────────────────────────────────────────────────

_AMBIGUOUS_TICKERS = {"C", "V", "MA", "ALL", "MS", "META", "NOW", "HD"}


def _is_relevant_headline(title: str, ticker: str) -> bool:
    title_lower = title.lower()
    if any(spam in title_lower for spam in SPAM_KEYWORDS):
        return False
    if ticker.upper() not in _AMBIGUOUS_TICKERS:
        ticker_lower = ticker.lower()
        if f" {ticker_lower} " in f" {title_lower} " or title_lower.startswith(f"{ticker_lower} "):
            return True
    company_terms = COMPANY_KEYWORDS.get(ticker.upper(), [])
    return any(term in title_lower for term in company_terms)


# ── Cached data ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_ohlcv(ticker: str) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=730)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info or {}
        return {
            "name": info.get("shortName", info.get("longName", ticker)),
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "change_pct": info.get("regularMarketChangePercent"),
            "high_52w": info.get("fiftyTwoWeekHigh"),
            "low_52w": info.get("fiftyTwoWeekLow"),
        }
    except Exception:
        return {"name": ticker, "price": None, "change_pct": None,
                "high_52w": None, "low_52w": None}


@st.cache_data(ttl=3600, show_spinner=False)
def _load_sentiment_timeline(ticker: str) -> pd.DataFrame | None:
    """Load pre-computed daily sentiment + close price for a ticker (last 60 days)."""
    from src.config import FEATURES_NLP_PATH, FEATURES_MARKET_PATH

    try:
        nlp = pd.read_parquet(FEATURES_NLP_PATH)
        nlp.index = pd.to_datetime(nlp.index)
        t_nlp = nlp[nlp["ticker"] == ticker][["finbert_sentiment", "is_sentiment_imputed"]].tail(60)

        market = pd.read_parquet(FEATURES_MARKET_PATH)
        market.index = pd.to_datetime(market.index)
        t_mkt = market[market["ticker"] == ticker][["close"]].tail(60)

        merged = t_mkt.join(t_nlp, how="inner")
        if len(merged) < 3:
            return None
        return merged
    except Exception:
        return None


def _nlp_importance_pct(predictor, horizon: int) -> float | None:
    """Compute what % of total feature importance comes from NLP features."""
    if horizon not in predictor._models:
        return None
    model, feature_cols = predictor._models[horizon]
    imp = None
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_cols)
    elif hasattr(model, "named_estimators_"):
        for est in model.named_estimators_.values():
            if hasattr(est, "feature_importances_"):
                imp = pd.Series(est.feature_importances_, index=feature_cols)
                break
    if imp is None:
        return None
    nlp_keys = ("finbert", "vader", "news", "headline", "sentiment")
    nlp_imp = imp[[c for c in imp.index if any(k in c for k in nlp_keys)]].sum()
    return float(nlp_imp / imp.sum() * 100) if imp.sum() > 0 else 0.0


def _get_feature_importances(predictor, horizon: int, n: int = 5) -> pd.Series | None:
    """Extract top N feature importances from the model."""
    if horizon not in predictor._models:
        return None
    model, feature_cols = predictor._models[horizon]
    imp = None
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_cols)
    elif hasattr(model, "named_estimators_"):
        for est in model.named_estimators_.values():
            if hasattr(est, "feature_importances_"):
                imp = pd.Series(est.feature_importances_, index=feature_cols)
                break
    if imp is not None:
        return imp.nlargest(n)
    return None


# ── Chart ────────────────────────────────────────────────────────────────────

def _candlestick_chart(df: pd.DataFrame, days: int = 90) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if days < 9999:
        df = df.tail(days).copy()
    if df.empty:
        return

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.02,
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color=_UP_COLOR, decreasing_line_color=_DOWN_COLOR,
        increasing_fillcolor=_UP_COLOR, decreasing_fillcolor=_DOWN_COLOR,
        name="OHLC",
    ), row=1, col=1)
    vol_colors = [_UP_COLOR if c >= o else _DOWN_COLOR
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], marker_color=vol_colors, opacity=0.4,
        name="Volume", showlegend=False,
    ), row=2, col=1)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,23,0.6)",
        xaxis_rangeslider_visible=False,
        height=460,
        margin=dict(l=0, r=0, t=8, b=0),
        showlegend=False,
        font=dict(family="Inter, sans-serif"),
    )
    for row in [1, 2]:
        fig.update_yaxes(gridcolor="#1e293b", gridwidth=0.5, row=row, col=1, zeroline=False)
    fig.update_xaxes(gridcolor="#1e293b", gridwidth=0.5, zeroline=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _sentiment_timeline(ticker: str, nlp_pct: float | None = None) -> None:
    """Render a dual-axis chart: closing price (line) + daily sentiment (bars)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    data = _load_sentiment_timeline(ticker)
    if data is None:
        st.markdown(
            f'<div class="glass-card" style="text-align:center;color:{_MUTED};font-size:0.88rem">'
            f'Limited news data available for {html_escape(ticker)}. '
            f'Using sector-level sentiment.</div>',
            unsafe_allow_html=True,
        )
        return

    # Last 30 trading days
    data = data.tail(30)

    bar_colors = [_UP_COLOR if s > 0 else _DOWN_COLOR for s in data["finbert_sentiment"]]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=data.index, y=data["close"],
        mode="lines", name="Close",
        line=dict(color="#94a3b8", width=1.5),
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        x=data.index, y=data["finbert_sentiment"],
        name="Sentiment", marker_color=bar_colors, opacity=0.7,
    ), secondary_y=True)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=200,
        margin=dict(l=0, r=0, t=5, b=0),
        showlegend=False,
        font=dict(family="Inter, sans-serif", size=11),
    )
    fig.update_yaxes(
        title_text="Close", gridcolor="#1e293b", zeroline=False,
        showticklabels=True, tickfont=dict(color="#64748b", size=10),
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Sentiment", gridcolor="rgba(0,0,0,0)", zeroline=True,
        zerolinecolor="#1e293b", showticklabels=True,
        tickfont=dict(color="#64748b", size=10),
        secondary_y=True,
    )
    fig.update_xaxes(gridcolor="#1e293b", gridwidth=0.5)

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if nlp_pct is not None:
        if nlp_pct >= 1.0:
            note = (f'NLP sentiment features contribute <b style="color:#f59e0b">{nlp_pct:.1f}%</b>'
                    f' of total model feature importance.')
        else:
            note = (f'NLP contribution is minimal (<b style="color:#f59e0b">{nlp_pct:.1f}%</b> importance) '
                    f'— most days rely on sector-level sentiment fallback due to sparse news coverage.')
        st.markdown(
            f'<p style="color:{_MUTED};font-size:0.82rem;margin-top:-0.5rem">{note}</p>',
            unsafe_allow_html=True,
        )


# ── Prediction card ──────────────────────────────────────────────────────────

def _render_prediction_card(result: dict) -> None:
    pred = result["prediction"]
    conf = result["confidence"]
    probs = result["probabilities"]
    horizon = result.get("horizon", 5)

    is_up = pred == "UP"
    color = _UP_COLOR if is_up else _DOWN_COLOR
    arrow = "↑" if is_up else "↓"
    card_class = "pred-card-up" if is_up else "pred-card-down"
    up_prob = probs.get("UP", 0)
    down_prob = probs.get("DOWN", 0)

    signal_label, signal_color = _signal_strength(conf)

    st.markdown(
        f'<div class="pred-card {card_class}">'
        f'<div style="font-size:0.8rem;color:{_MUTED};font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.05em;margin-bottom:8px">{horizon}-Day Forecast</div>'
        f'<div style="font-size:3rem;line-height:1;margin:4px 0">{arrow}</div>'
        f'<div style="font-size:1.8rem;font-weight:800;color:{color};margin:4px 0;'
        f'letter-spacing:-0.02em">{pred}</div>'
        f'<div style="color:{_MUTED};font-size:0.88rem">'
        f'Confidence: <b style="color:{color}">{conf:.0%}</b></div>'
        f'<div style="margin:8px 0 16px"><span style="background:{signal_color}18;color:{signal_color};'
        f'padding:3px 12px;border-radius:20px;font-size:0.78rem;font-weight:700">'
        f'{signal_label}</span></div>'
        f'<div>'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:6px">'
        f'<span style="color:{_DOWN_COLOR};font-size:0.82rem;font-weight:700">↓ {down_prob:.0%}</span>'
        f'<span style="color:{_UP_COLOR};font-size:0.82rem;font-weight:700">↑ {up_prob:.0%}</span></div>'
        f'<div style="background:#1e293b;border-radius:6px;height:8px;overflow:hidden;display:flex">'
        f'<div style="background:{_DOWN_COLOR};width:{down_prob:.0%};height:100%"></div>'
        f'<div style="background:{_UP_COLOR};width:{up_prob:.0%};height:100%"></div>'
        f'</div></div></div>',
        unsafe_allow_html=True,
    )


def _render_details_card(result: dict) -> None:
    st.markdown(
        f'<div class="glass-card" style="margin-top:8px">'
        f'<div style="display:flex;justify-content:space-between;padding:4px 0">'
        f'<span style="color:{_MUTED};font-size:0.85rem">Market date</span>'
        f'<span style="color:#e2e8f0;font-size:0.85rem;font-weight:500">{result["market_date"]}</span></div>'
        f'<div style="display:flex;justify-content:space-between;padding:4px 0;'
        f'border-top:1px solid #1e293b;margin-top:4px">'
        f'<span style="color:{_MUTED};font-size:0.85rem">Headlines used</span>'
        f'<span style="color:#e2e8f0;font-size:0.85rem;font-weight:500">{result["n_headlines"]}</span></div>'
        f'<div style="display:flex;justify-content:space-between;padding:4px 0;'
        f'border-top:1px solid #1e293b;margin-top:4px">'
        f'<span style="color:{_MUTED};font-size:0.85rem">Horizon</span>'
        f'<span style="color:#60a5fa;font-size:0.85rem;font-weight:500">'
        f'{result.get("horizon", 5)} trading days</span></div></div>',
        unsafe_allow_html=True,
    )


# ── Market context metrics ───────────────────────────────────────────────────

def _render_market_context(market_feat: pd.DataFrame, nlp_feat: pd.Series) -> None:
    """Show 4 key market context metrics below the chart."""
    row = market_feat.iloc[-1]

    rsi = row.get("rsi_14", np.nan)
    vol = row.get("volatility_20d", np.nan)
    vol_ratio = row.get("volume_ratio", np.nan)
    sentiment = nlp_feat.get("finbert_sentiment", 0.0)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        rsi_delta = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
        st.metric("RSI (14)", f"{rsi:.1f}" if not np.isnan(rsi) else "–", delta=rsi_delta,
                  delta_color="inverse" if rsi > 70 else "normal" if rsi < 30 else "off")
    with c2:
        st.metric("20D Volatility", f"{vol:.1%}" if not np.isnan(vol) else "–",
                  delta="High" if vol > 0.025 else "Low" if vol < 0.01 else "Normal",
                  delta_color="inverse" if vol > 0.025 else "off")
    with c3:
        sent_label = "Bullish" if sentiment > 0.05 else "Bearish" if sentiment < -0.05 else "Neutral"
        st.metric("News Sentiment", f"{sentiment:+.3f}", delta=sent_label,
                  delta_color="normal" if sentiment > 0.05 else "inverse" if sentiment < -0.05 else "off")
    with c4:
        vol_label = f"{vol_ratio:.2f}x" if not np.isnan(vol_ratio) else "–"
        st.metric("Vol vs 20D Avg", vol_label,
                  delta="Above avg" if vol_ratio > 1.2 else "Below avg" if vol_ratio < 0.8 else "Normal",
                  delta_color="off")


# ── Feature drivers chart ────────────────────────────────────────────────────

def _render_feature_drivers(predictor, horizon: int, feature_vec: pd.Series) -> None:
    """Show top 5 features driving the prediction as a horizontal bar chart."""
    import plotly.graph_objects as go

    top_imp = _get_feature_importances(predictor, horizon, n=5)
    if top_imp is None:
        return

    names = []
    values = []
    colors = []
    annotations = []

    for feat_name, importance in top_imp.items():
        label = _FEAT_LABELS.get(feat_name, feat_name)
        feat_val = feature_vec.get(feat_name, 0.0)
        direction, color = _feat_direction(feat_name, feat_val)
        names.append(label)
        values.append(importance)
        colors.append(color)
        if direction:
            annotations.append(f"{direction} ({feat_val:+.3f})")
        else:
            annotations.append(f"{feat_val:.3f}")

    # Reverse for Plotly (bottom → top)
    names, values, colors, annotations = names[::-1], values[::-1], colors[::-1], annotations[::-1]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=colors,
        text=annotations,
        textposition="outside",
        textfont=dict(size=11, color="#94a3b8", family="Inter"),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=10, r=100, t=5, b=5),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(gridcolor="#1e293b", tickfont=dict(size=12, color="#e2e8f0")),
        font=dict(family="Inter, sans-serif"),
        bargap=0.35,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Headlines ────────────────────────────────────────────────────────────────

def _render_headline(title: str, source: str, date_str: str, sentiment: float | None = None) -> None:
    safe_title = html_escape(title)
    safe_source = html_escape(source)

    badge_html = ""
    if sentiment is not None:
        s_color = _UP_COLOR if sentiment > 0.1 else _DOWN_COLOR if sentiment < -0.1 else _AMBER
        badge_html = (
            f'<span style="background:{s_color}18;color:{s_color};padding:3px 10px;'
            f'border-radius:20px;font-size:0.78rem;font-weight:600;white-space:nowrap">'
            f'{sentiment:+.2f}</span>'
        )

    st.markdown(
        f'<div class="headline-card">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;gap:12px">'
        f'<div>'
        f'<div style="font-weight:500;font-size:0.9rem;color:#e2e8f0;line-height:1.3">{safe_title}</div>'
        f'<div style="color:{_MUTED};font-size:0.76rem;margin-top:3px">{safe_source} &middot; {date_str}</div>'
        f'</div>{badge_html}</div></div>',
        unsafe_allow_html=True,
    )


# ── Main render ──────────────────────────────────────────────────────────────

def render() -> None:
    predictor = get_predictor()

    # ── Ticker selector row ──────────────────────────────────────────────────
    col_sel, col_horizon, col_btn, _ = st.columns([2, 1.5, 1, 2.5])

    with col_sel:
        ticker = st.selectbox(
            "Ticker", TICKERS_SORTED,
            index=TICKERS_SORTED.index("AAPL") if "AAPL" in TICKERS_SORTED else 0,
            label_visibility="collapsed",
        )

    with col_horizon:
        horizons = predictor.available_horizons
        horizon_labels = {5: "5-Day Forecast", 21: "21-Day Forecast"}
        options = [horizon_labels.get(h, f"{h}-Day") for h in horizons]
        if options:
            sel_label = st.selectbox("Horizon", options, label_visibility="collapsed")
            selected_horizon = horizons[options.index(sel_label)]
        else:
            selected_horizon = 5

    with col_btn:
        run = st.button("Predict", type="primary", use_container_width=True)

    # ── Ticker info header ───────────────────────────────────────────────────
    sector = TICKER_SECTOR_MAP.get(ticker, "")
    info = _fetch_info(ticker)
    name = info["name"]

    parts = [f"<b style='color:#f0f6fc;font-size:1.1rem'>{html_escape(name)}</b>"]
    if sector:
        parts.append(f"<span class='info-tag blue'>{sector}</span>")
    if info["price"]:
        parts.append(f"<span style='color:#f0f6fc;font-weight:700;font-size:1.05rem'>"
                      f"${info['price']:,.2f}</span>")
    if info["change_pct"] is not None:
        chg = info["change_pct"]
        chg_cls = "green" if chg >= 0 else "red"
        parts.append(f"<span class='info-tag {chg_cls}'>{chg:+.2f}%</span>")
    if info["low_52w"] and info["high_52w"]:
        parts.append(f"<span style='color:{_MUTED};font-size:0.82rem'>"
                      f"52W: ${info['low_52w']:,.0f} – ${info['high_52w']:,.0f}</span>")

    st.markdown(
        f"<div style='display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:-0.3rem 0 0.6rem'>"
        f"{''.join(parts)}</div>",
        unsafe_allow_html=True,
    )

    # ── Chart period selector ────────────────────────────────────────────────
    if "chart_period" not in st.session_state:
        st.session_state.chart_period = "3M"

    period_cols = st.columns(len(_CHART_PERIODS) + 4)
    for i, label in enumerate(_CHART_PERIODS):
        with period_cols[i]:
            if st.button(label, key=f"period_{label}",
                         type="primary" if label == st.session_state.chart_period else "secondary",
                         use_container_width=True):
                st.session_state.chart_period = label
                st.rerun()

    # ── Chart ────────────────────────────────────────────────────────────────
    ohlcv = _fetch_ohlcv(ticker)
    if not ohlcv.empty:
        days = _CHART_PERIODS[st.session_state.chart_period]
        _candlestick_chart(ohlcv, days=days)

    # ── Sentiment timeline ───────────────────────────────────────────────────
    st.markdown(
        f"<div style='margin-top:0.3rem;margin-bottom:0.4rem'>"
        f"<span style='color:{_MUTED};font-size:0.9rem;font-weight:600;"
        f"text-transform:uppercase;letter-spacing:0.05em'>"
        f"Sentiment vs Price — Last 30 Days</span></div>",
        unsafe_allow_html=True,
    )
    nlp_pct = _nlp_importance_pct(predictor, selected_horizon)
    _sentiment_timeline(ticker, nlp_pct=nlp_pct)

    # ── Prediction ───────────────────────────────────────────────────────────
    if run:
        results = {}
        market_feat = None
        nlp_feat = None
        cv_feat = None

        progress_bar = st.progress(0)
        progress_text = st.empty()

        progress_text.markdown("📊 **Fetching market data & computing indicators...**")
        progress_bar.progress(15)
        market_feat = predictor.build_market_features(ticker, ohlcv_df=ohlcv)

        progress_text.markdown("📰 **Running sentiment analysis (FinBERT + VADER)...**")
        progress_bar.progress(40)
        nlp_feat = predictor.build_nlp_features(ticker)

        progress_text.markdown("📈 **Processing chart patterns (EfficientNet-B0)...**")
        progress_bar.progress(65)
        cv_feat = predictor.build_cv_features(ticker, ohlcv_df=ohlcv)

        progress_text.markdown(f"🤖 **Running {selected_horizon}-day prediction model...**")
        progress_bar.progress(85)
        results[selected_horizon] = predictor.predict_from_features(
            ticker, market_feat, nlp_feat, cv_feat, horizon=selected_horizon
        )

        # Also run the other horizon if available
        other_horizons = [h for h in horizons if h != selected_horizon]
        for h in other_horizons:
            results[h] = predictor.predict_from_features(
                ticker, market_feat, nlp_feat, cv_feat, horizon=h
            )

        progress_bar.progress(100)
        progress_text.empty()
        progress_bar.empty()

        # ── Market context metrics ───────────────────────────────────────────
        _render_market_context(market_feat, nlp_feat)

        # ── Prediction cards ─────────────────────────────────────────────────
        st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)

        if len(results) == 2:
            h1, h2 = sorted(results.keys())
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                _render_prediction_card(results[h1])
            with col2:
                _render_prediction_card(results[h2])
            with col3:
                _render_details_card(results[selected_horizon])
        elif len(results) == 1:
            col1, col2 = st.columns([1, 1])
            with col1:
                _render_prediction_card(list(results.values())[0])
            with col2:
                _render_details_card(list(results.values())[0])

        # ── Feature drivers ──────────────────────────────────────────────────
        st.markdown(
            f"<div style='margin-top:1.2rem;margin-bottom:0.4rem'>"
            f"<span style='color:{_MUTED};font-size:0.9rem;font-weight:600;"
            f"text-transform:uppercase;letter-spacing:0.05em'>"
            f"What's driving this prediction?</span></div>",
            unsafe_allow_html=True,
        )

        # Build feature vector for annotation
        latest_row = market_feat.iloc[-1]
        feature_vec = pd.concat([latest_row, nlp_feat, cv_feat])
        _render_feature_drivers(predictor, selected_horizon, feature_vec)

        # ── Headlines ────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='margin-top:1rem;margin-bottom:0.6rem'>"
            f"<span style='color:{_MUTED};font-size:0.9rem;font-weight:600;"
            f"text-transform:uppercase;letter-spacing:0.05em'>"
            f"Recent Headlines — {html_escape(ticker)}</span></div>",
            unsafe_allow_html=True,
        )

        try:
            from src.data_collection.news_scraper import load_ticker_news
            news = load_ticker_news(ticker)
            if not news.empty:
                news = news[news["title"].apply(lambda t: _is_relevant_headline(t, ticker))]
                if not news.empty:
                    news["published"] = pd.to_datetime(news["published"]).dt.strftime("%b %d")
                    for _, row in news.head(5).iterrows():
                        _render_headline(row["title"], row.get("source", ""), row["published"])
                else:
                    st.markdown(
                        f'<div class="glass-card" style="text-align:center;color:{_MUTED}">'
                        f'No relevant headlines found for {html_escape(ticker)}.<br>'
                        f'<span style="font-size:0.85rem">Prediction uses sector-level sentiment as fallback.</span>'
                        f'</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="glass-card" style="text-align:center;color:{_MUTED}">'
                    f'No headlines available for {html_escape(ticker)}.<br>'
                    f'<span style="font-size:0.85rem">Prediction uses sector-level sentiment as fallback.</span>'
                    f'</div>', unsafe_allow_html=True)
        except (FileNotFoundError, Exception):
            st.markdown(
                f'<div class="glass-card" style="text-align:center;color:{_MUTED}">'
                f'No saved news data available.<br>'
                f'<span style="font-size:0.85rem">Prediction uses sector-level sentiment as fallback.</span>'
                f'</div>', unsafe_allow_html=True)

    else:
        st.markdown(
            f"<p style='color:{_MUTED};text-align:center;margin-top:1.5rem;font-size:0.95rem'>"
            f"Select a ticker and click <b style='color:#60a5fa'>Predict</b> to generate a forecast</p>",
            unsafe_allow_html=True,
        )

    # ── Disclaimer ───────────────────────────────────────────────────────────
    st.markdown(
        "<p class='disclaimer'>"
        "Research prototype — not financial advice. Predictions are for educational purposes only "
        "and do not constitute investment recommendations. Past patterns do not guarantee future returns.</p>",
        unsafe_allow_html=True,
    )
