"""predictor.py — Live prediction page with Plotly candlestick chart."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from html import escape as html_escape

from src.app.utils import get_predictor, TICKERS_SORTED
from src.config import (
    COMPANY_KEYWORDS, SPAM_KEYWORDS, TICKER_SECTOR_MAP,
    FEATURES_MARKET_PATH, FEATURES_NLP_PATH, FEATURES_CV_PATH,
    STACKING_MODEL_PATH, MODEL_21D_PATH, TEST_START,
)

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


def _load_ticker_news_dates(ticker: str) -> dict[str, str]:
    """Return {date_str: headline} for high-sentiment days (|sentiment| > 0.2)."""
    try:
        from src.data_collection.news_scraper import load_ticker_news
        news = load_ticker_news(ticker)
        if news.empty or "published" not in news.columns:
            return {}
        news["date"] = pd.to_datetime(news["published"]).dt.date.astype(str)
        # Keep only dates with significant news (most recent per day)
        result = {}
        for date_str, grp in news.groupby("date"):
            result[date_str] = grp["title"].iloc[0][:60] + "…"
        return result
    except Exception:
        return {}


def _sentiment_timeline(ticker: str, nlp_pct: float | None = None) -> None:
    """Render a dual-axis chart: closing price (line) + daily sentiment (bars).

    News events with |sentiment| > 0.2 are annotated with vertical markers.
    """
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

    # ── News event annotations (vertical markers on significant days) ─────────
    news_dates = _load_ticker_news_dates(ticker)
    significant_days = data[data["finbert_sentiment"].abs() > 0.2].index
    for date in significant_days:
        date_str = str(date.date()) if hasattr(date, "date") else str(date)[:10]
        headline = news_dates.get(date_str, f"Significant sentiment ({date_str})")
        marker_color = _UP_COLOR if data.loc[date, "finbert_sentiment"] > 0 else _DOWN_COLOR
        fig.add_vline(
            x=date,
            line_color=marker_color,
            line_dash="dot",
            line_width=1.5,
            opacity=0.6,
        )
        fig.add_annotation(
            x=date,
            y=1.0,
            yref="paper",
            text="📰",
            showarrow=False,
            font=dict(size=10),
            hovertext=headline,
            bgcolor="#1e293b",
            bordercolor=marker_color,
            borderpad=2,
            yshift=10,
        )

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

    # Baseline-relative confidence: how far above 50% coin-flip
    baseline_delta = (conf - 0.5) * 100  # pp above random
    baseline_color = _UP_COLOR if baseline_delta >= 0 else _DOWN_COLOR
    # Width of the confidence fill relative to 50%
    fill_pct = min(int((conf - 0.5) * 200), 100)  # 0-100% of right half

    st.markdown(
        f'<div class="pred-card {card_class}">'
        f'<div style="font-size:0.8rem;color:{_MUTED};font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.05em;margin-bottom:8px">{horizon}-Day Forecast</div>'
        f'<div style="font-size:3rem;line-height:1;margin:4px 0">{arrow}</div>'
        f'<div style="font-size:1.8rem;font-weight:800;color:{color};margin:4px 0;'
        f'letter-spacing:-0.02em">{pred}</div>'
        f'<div style="color:{_MUTED};font-size:0.88rem">'
        f'Confidence: <b style="color:{color}">{conf:.0%}</b>'
        f' &nbsp;·&nbsp; <span style="color:{baseline_color};font-size:0.82rem">'
        f'{baseline_delta:+.1f} pp vs 50% baseline</span></div>'
        f'<div style="margin:8px 0 4px"><span style="background:{signal_color}18;color:{signal_color};'
        f'padding:3px 12px;border-radius:20px;font-size:0.78rem;font-weight:700">'
        f'{signal_label}</span></div>'
        f'<div style="margin:8px 0 4px;font-size:0.75rem;color:{_MUTED}">'
        f'Edge above random chance:</div>'
        f'<div style="background:#1e293b;border-radius:6px;height:6px;overflow:hidden;margin-bottom:12px">'
        f'<div style="width:50%;height:100%;background:#1e293b;display:inline-block"></div>'
        f'<div style="width:{fill_pct}%;height:6px;background:{baseline_color};'
        f'border-radius:0 4px 4px 0;display:inline-block;vertical-align:top"></div>'
        f'</div>'
        f'<div>'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
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

def _render_market_context(
    market_feat: pd.DataFrame,
    nlp_feat: pd.Series,
    analyst_feat: pd.Series | None = None,
) -> None:
    """Show 5 key market context metrics below the chart."""
    row = market_feat.iloc[-1]

    rsi = row.get("rsi_14", np.nan)
    vol = row.get("volatility_20d", np.nan)
    vol_ratio = row.get("volume_ratio", np.nan)
    sentiment = nlp_feat.get("finbert_sentiment", 0.0)

    c1, c2, c3, c4, c5 = st.columns(5)

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
    with c5:
        if analyst_feat is not None:
            consensus = float(analyst_feat.get("analyst_consensus", 0.0))
            coverage = int(analyst_feat.get("analyst_coverage_count", 0))
            upside = float(analyst_feat.get("price_target_upside", 0.0))
            if coverage > 0:
                cons_label = "Buy" if consensus >= 1.5 else "Outperform" if consensus >= 0.8 else \
                             "Hold" if consensus >= -0.5 else "Underperform" if consensus >= -1.5 else "Sell"
                cons_delta = f"{coverage} analysts"
                upside_str = f" · {upside:+.1%}" if upside != 0 else ""
                st.metric("Analyst Rating", f"{cons_label}{upside_str}", delta=cons_delta,
                          delta_color="normal" if consensus > 0 else "inverse" if consensus < 0 else "off")
            else:
                st.metric("Analyst Rating", "–", delta="No coverage", delta_color="off")
        else:
            st.metric("Analyst Rating", "–", delta="Not loaded", delta_color="off")


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
        if len(label) > 15:
            label = label[:14] + "…"
        feat_val = feature_vec.get(feat_name, 0.0)
        direction, color = _feat_direction(feat_name, feat_val)
        names.append(label)
        values.append(importance)
        colors.append(color)
        if direction:
            annotations.append(f"{direction} ({feat_val:+.2f})")
        else:
            annotations.append(f"{feat_val:.2f}")

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
        margin=dict(l=10, r=130, t=5, b=5),
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


# ── Backtest ─────────────────────────────────────────────────────────────────

_MODEL_PATHS = {5: STACKING_MODEL_PATH, 21: MODEL_21D_PATH}


@st.cache_data(ttl=3600, show_spinner=False)
def _compute_backtest(ticker: str, horizon: int) -> pd.DataFrame | None:
    """Run model predictions on held-out test set features for a ticker.

    Returns DataFrame with columns: date, actual, predicted, confidence, correct.
    Uses pre-saved feature parquets — no live data fetching.
    """
    import joblib

    model_path = _MODEL_PATHS.get(horizon)
    if model_path is None or not model_path.exists():
        return None

    try:
        artifact = joblib.load(model_path)
        model = artifact["model"]
        feature_cols = artifact["feature_cols"]

        # Load features
        mkt = pd.read_parquet(FEATURES_MARKET_PATH)
        nlp = pd.read_parquet(FEATURES_NLP_PATH)
        cv = pd.read_parquet(FEATURES_CV_PATH)

        # Filter to ticker
        t_mkt = mkt[mkt["ticker"] == ticker].copy()
        t_nlp = nlp[nlp["ticker"] == ticker].drop(columns=["ticker"])
        t_cv = cv[cv["ticker"] == ticker].drop(columns=["ticker"])

        # Join features
        combined = t_mkt.join(t_nlp, how="inner").join(t_cv, how="inner")

        # One-hot encode sector dummies (model expects sector_*)
        if "sector" in combined.columns:
            dummies = pd.get_dummies(combined["sector"], prefix="sector")
            # Ensure all expected sector columns exist
            for col in feature_cols:
                if col.startswith("sector_") and col not in dummies.columns:
                    dummies[col] = 0
            combined = pd.concat([combined, dummies], axis=1)

        # Build target column
        if horizon == 5:
            # Already stored in features_market as 'target'
            if "target" not in combined.columns:
                return None
            combined["actual"] = combined["target"]
        else:
            # Compute forward return for the given horizon
            combined["fwd_return"] = combined["close"].shift(-horizon) / combined["close"] - 1
            combined["actual"] = np.where(combined["fwd_return"] > 0, "UP", "DOWN")
            combined = combined.dropna(subset=["fwd_return"])

        # Filter to test period, last 6 months
        combined = combined[combined.index >= TEST_START]
        # Drop rows where actual outcome isn't yet known
        if horizon == 5:
            combined = combined[combined["actual"].notna()]
        combined = combined.tail(130)  # ~6 months of trading days

        if len(combined) < 10:
            return None

        # Run predictions
        X = combined[feature_cols].fillna(0)
        proba = model.predict_proba(X)
        up_idx = list(model.classes_).index("UP")
        preds = model.predict(X)

        result = pd.DataFrame({
            "date": combined.index,
            "actual": combined["actual"].values,
            "predicted": preds,
            "confidence": np.max(proba, axis=1),
            "up_prob": proba[:, up_idx],
        })
        result["correct"] = result["actual"] == result["predicted"]
        return result.reset_index(drop=True)

    except Exception:
        return None


def _render_backtest(ticker: str, horizon: int) -> None:
    """Render the backtest expander with chart and metrics."""
    import plotly.graph_objects as go

    with st.expander("📊 Historical Prediction Accuracy", expanded=True):
        bt = _compute_backtest(ticker, horizon)
        if bt is None or len(bt) < 10:
            st.markdown(
                f'<p style="color:{_MUTED};text-align:center;font-size:0.9rem">'
                f'Insufficient test data for {html_escape(ticker)} ({horizon}D horizon).</p>',
                unsafe_allow_html=True,
            )
            return

        # ── Scatter: confidence dots colored by correctness ──────────────
        correct = bt[bt["correct"]]
        wrong = bt[~bt["correct"]]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=correct["date"], y=correct["confidence"],
            mode="markers", name="Correct",
            marker=dict(color=_UP_COLOR, size=7, opacity=0.8),
        ))
        fig.add_trace(go.Scatter(
            x=wrong["date"], y=wrong["confidence"],
            mode="markers", name="Wrong",
            marker=dict(color=_DOWN_COLOR, size=7, opacity=0.8),
        ))

        # ── Rolling 30-day accuracy line ─────────────────────────────────
        bt["rolling_acc"] = bt["correct"].rolling(30, min_periods=10).mean()
        fig.add_trace(go.Scatter(
            x=bt["date"], y=bt["rolling_acc"],
            mode="lines", name="30D Rolling Acc",
            line=dict(color="#60a5fa", width=2),
        ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,14,23,0.6)",
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(
                title="Confidence / Accuracy",
                range=[0.35, 1.0],
                gridcolor="#1e293b",
                tickformat=".0%",
            ),
            xaxis=dict(gridcolor="#1e293b"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                font=dict(size=11, color="#94a3b8"),
                bgcolor="rgba(0,0,0,0)",
            ),
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── Metric cards ─────────────────────────────────────────────────
        overall_acc = bt["correct"].mean()
        up_mask = bt["predicted"] == "UP"
        tp = ((bt["predicted"] == "UP") & (bt["actual"] == "UP")).sum()
        fp = ((bt["predicted"] == "UP") & (bt["actual"] == "DOWN")).sum()
        fn = ((bt["predicted"] == "DOWN") & (bt["actual"] == "UP")).sum()
        precision_up = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_up = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Overall Accuracy", f"{overall_acc:.1%}")
        with c2:
            st.metric("Precision (UP)", f"{precision_up:.1%}")
        with c3:
            st.metric("Recall (UP)", f"{recall_up:.1%}")

        st.markdown(
            f'<p style="color:{_MUTED};font-size:0.78rem;margin-top:0.5rem">'
            f'Historical performance shown on the held-out test set (2025). '
            f'Not representative of future performance.</p>',
            unsafe_allow_html=True,
        )


# ── Main render ──────────────────────────────────────────────────────────────

def _render_watchlist() -> str | None:
    """Render watchlist quick-buttons. Returns ticker if quick-button was clicked."""
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []

    watchlist = st.session_state.watchlist
    if not watchlist:
        return None

    st.markdown(
        f"<div style='margin-bottom:0.4rem'>"
        f"<span style='color:{_MUTED};font-size:0.78rem;font-weight:600;'"
        f"text-transform:uppercase;letter-spacing:0.05em'>Recent:</span></div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(min(len(watchlist), 6))
    for i, t in enumerate(watchlist[:6]):
        with cols[i]:
            if st.button(t, key=f"wl_{t}", use_container_width=True):
                return t
    return None


def _add_to_watchlist(ticker: str) -> None:
    """Add ticker to watchlist (max 6, deduplicated, most-recent first)."""
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []
    wl = st.session_state.watchlist
    if ticker in wl:
        wl.remove(ticker)
    wl.insert(0, ticker)
    st.session_state.watchlist = wl[:6]


def render() -> None:
    predictor = get_predictor()

    # ── Watchlist quick-access ───────────────────────────────────────────────
    clicked_ticker = _render_watchlist()

    # ── Ticker selector row ──────────────────────────────────────────────────
    col_sel, col_horizon, col_btn, _ = st.columns([2, 1.5, 1, 2.5])

    with col_sel:
        default_idx = TICKERS_SORTED.index(clicked_ticker) if clicked_ticker and clicked_ticker in TICKERS_SORTED \
            else (TICKERS_SORTED.index("AAPL") if "AAPL" in TICKERS_SORTED else 0)
        ticker = st.selectbox(
            "Ticker", TICKERS_SORTED,
            index=default_idx,
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
        progress_bar.progress(35)
        nlp_feat = predictor.build_nlp_features(ticker)

        progress_text.markdown("🏦 **Fetching analyst ratings & price targets...**")
        progress_bar.progress(55)
        current_price = float(ohlcv["Close"].iloc[-1]) if not ohlcv.empty else None
        analyst_feat = predictor.build_analyst_features(ticker, current_price=current_price)

        progress_text.markdown("📈 **Processing chart patterns (EfficientNet-B0)...**")
        progress_bar.progress(70)
        cv_feat = predictor.build_cv_features(ticker, ohlcv_df=ohlcv)

        progress_text.markdown(f"🤖 **Running {selected_horizon}-day prediction model...**")
        progress_bar.progress(88)
        results[selected_horizon] = predictor.predict_from_features(
            ticker, market_feat, nlp_feat, cv_feat, horizon=selected_horizon,
            analyst_feat=analyst_feat,
        )

        # Also run the other horizon if available
        other_horizons = [h for h in horizons if h != selected_horizon]
        for h in other_horizons:
            results[h] = predictor.predict_from_features(
                ticker, market_feat, nlp_feat, cv_feat, horizon=h,
                analyst_feat=analyst_feat,
            )

        progress_bar.progress(100)
        progress_text.empty()
        progress_bar.empty()

        # Add to watchlist after successful prediction
        _add_to_watchlist(ticker)

        # ── Market context metrics ───────────────────────────────────────────
        _render_market_context(market_feat, nlp_feat, analyst_feat=analyst_feat)

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

    # ── Backtest ──────────────────────────────────────────────────────────────
    _render_backtest(ticker, selected_horizon)

    # ── Disclaimer ───────────────────────────────────────────────────────────
    st.markdown(
        "<p class='disclaimer'>"
        "Research prototype — not financial advice. Predictions are for educational purposes only "
        "and do not constitute investment recommendations. Past patterns do not guarantee future returns.</p>",
        unsafe_allow_html=True,
    )


# ── Compare tab ───────────────────────────────────────────────────────────────

def render_compare() -> None:
    """Render side-by-side two-ticker comparison page."""
    p = get_predictor()

    st.markdown("<h2 style='margin-bottom:4px'>Ticker Comparison</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{_MUTED};font-size:0.9rem;margin-bottom:1rem'>"
        "Compare 5-day predictions for two tickers side by side.</p>",
        unsafe_allow_html=True,
    )

    col_a, col_b, col_h, col_run = st.columns([2, 2, 1.5, 1])
    with col_a:
        ticker_a = st.selectbox(
            "Ticker A", TICKERS_SORTED,
            index=TICKERS_SORTED.index("AAPL") if "AAPL" in TICKERS_SORTED else 0,
            label_visibility="collapsed", key="cmp_a",
        )
    with col_b:
        ticker_b = st.selectbox(
            "Ticker B", TICKERS_SORTED,
            index=TICKERS_SORTED.index("MSFT") if "MSFT" in TICKERS_SORTED else 1,
            label_visibility="collapsed", key="cmp_b",
        )
    with col_h:
        horizons = p.available_horizons
        horizon_labels = {5: "5-Day", 21: "21-Day"}
        opts = [horizon_labels.get(h, f"{h}-Day") for h in horizons]
        sel = st.selectbox("Horizon", opts, label_visibility="collapsed", key="cmp_h")
        selected_h = horizons[opts.index(sel)]
    with col_run:
        run_cmp = st.button("Compare", type="primary", use_container_width=True, key="cmp_run")

    if run_cmp:
        results_map = {}
        for ticker in [ticker_a, ticker_b]:
            with st.spinner(f"Predicting {ticker}…"):
                try:
                    ohlcv = _fetch_ohlcv(ticker)
                    mf = p.build_market_features(ticker, ohlcv_df=ohlcv)
                    nf = p.build_nlp_features(ticker)
                    cf = p.build_cv_features(ticker, ohlcv_df=ohlcv)
                    price = float(ohlcv["Close"].iloc[-1]) if not ohlcv.empty else None
                    af = p.build_analyst_features(ticker, current_price=price)
                    result = p.predict_from_features(ticker, mf, nf, cf, horizon=selected_h, analyst_feat=af)
                    results_map[ticker] = (result, ohlcv, mf, nf)
                except Exception as exc:
                    st.error(f"{ticker}: {exc}")

        if len(results_map) == 2:
            col1, col2 = st.columns(2)
            for col, ticker in zip([col1, col2], [ticker_a, ticker_b]):
                result, ohlcv, mf, nf = results_map[ticker]
                with col:
                    # Ticker header
                    info = _fetch_info(ticker)
                    sector = TICKER_SECTOR_MAP.get(ticker, "")
                    st.markdown(
                        f"<div style='margin-bottom:0.4rem'>"
                        f"<b style='color:#f0f6fc;font-size:1.05rem'>{html_escape(info['name'])}</b>"
                        f"<span class='info-tag blue' style='margin-left:8px'>{sector}</span>"
                        + (f"<span style='color:#f0f6fc;font-weight:700;margin-left:8px'>${info['price']:,.2f}</span>"
                           if info.get('price') else "")
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                    _render_prediction_card(result)
                    # Compact candlestick
                    if not ohlcv.empty:
                        _candlestick_chart(ohlcv, days=60)
                    # Market context
                    rsi = float(mf.iloc[-1].get("rsi_14", 0))
                    vol = float(mf.iloc[-1].get("volatility_20d", 0))
                    sent = float(nf.get("finbert_sentiment", 0)) if hasattr(nf, 'get') else 0.0
                    m1, m2, m3 = st.columns(3)
                    m1.metric("RSI", f"{rsi:.1f}", delta="Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral",
                              delta_color="inverse" if rsi > 70 else "normal" if rsi < 30 else "off")
                    m2.metric("Vol", f"{vol:.1%}", delta="High" if vol > 0.025 else "Normal", delta_color="inverse" if vol > 0.025 else "off")
                    m3.metric("Sentiment", f"{sent:+.3f}", delta="Bullish" if sent > 0.05 else "Bearish" if sent < -0.05 else "Neutral",
                              delta_color="normal" if sent > 0.05 else "inverse" if sent < -0.05 else "off")

    else:
        st.markdown(
            f"<p style='color:{_MUTED};text-align:center;margin-top:2rem'>"
            f"Select two tickers and click <b style='color:#60a5fa'>Compare</b>.</p>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<p class='disclaimer'>Research prototype — not financial advice.</p>",
        unsafe_allow_html=True,
    )
