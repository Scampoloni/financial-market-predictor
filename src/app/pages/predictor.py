"""predictor.py — Live prediction page with Plotly candlestick chart."""

from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta

from src.app.utils import get_predictor, TICKERS_SORTED
from src.config import TICKER_SECTOR_MAP

# ── Colour palette ───────────────────────────────────────────────────────────
_PRED_COLOR = {"UP": "#22c55e", "DOWN": "#ef4444"}
_PRED_ICON  = {"UP": "▲", "DOWN": "▼"}
_PRED_LABEL = {"UP": "Bullish — next 5 trading days", "DOWN": "Bearish — next 5 trading days"}


# ── Cached data fetchers ────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_market_data(ticker: str) -> pd.DataFrame:
    """Fetch 6 months OHLCV, cached for 1 hour."""
    end = datetime.today()
    start = end - timedelta(days=180)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_ticker_info(ticker: str) -> dict:
    """Fetch basic ticker info (name, price, change), cached for 1 hour."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        return {
            "name": info.get("shortName", info.get("longName", ticker)),
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "change_pct": info.get("regularMarketChangePercent"),
            "high_52w": info.get("fiftyTwoWeekHigh"),
            "low_52w": info.get("fiftyTwoWeekLow"),
        }
    except Exception:
        return {"name": ticker, "price": None, "change_pct": None, "high_52w": None, "low_52w": None}


def _candlestick_chart(df: pd.DataFrame) -> None:
    """Plotly candlestick chart with volume subplot."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Take last 90 days
    df = df.tail(90).copy()
    if df.empty:
        return

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.02,
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        increasing_fillcolor="#22c55e", decreasing_fillcolor="#ef4444",
        name="OHLC",
    ), row=1, col=1)

    colors = ["#22c55e" if c >= o else "#ef4444"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], marker_color=colors, opacity=0.5,
        name="Volume", showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False,
        height=420,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    fig.update_yaxes(gridcolor="#21262d", row=1, col=1)
    fig.update_yaxes(gridcolor="#21262d", row=2, col=1)
    fig.update_xaxes(gridcolor="#21262d")

    st.plotly_chart(fig, use_container_width=True)


def _prediction_card(pred: str, conf: float, probs: dict) -> None:
    """Render the prediction result as styled HTML."""
    color = _PRED_COLOR[pred]
    arrow = _PRED_ICON[pred]
    up_prob = probs.get("UP", 0)
    down_prob = probs.get("DOWN", 0)

    st.markdown(f"""
    <div style="background:{color}12;border:1px solid {color}40;border-radius:12px;
                padding:24px;text-align:center;margin-bottom:1rem">
        <div style="font-size:2.8rem;line-height:1">{arrow}</div>
        <div style="font-size:1.6rem;font-weight:700;color:{color};margin:4px 0">{pred}</div>
        <div style="color:#8b949e;font-size:0.85rem">Confidence: {conf:.0%}</div>
    </div>
    """, unsafe_allow_html=True)

    # Probability bar
    st.markdown(f"""
    <div style="margin-top:8px">
        <div style="display:flex;justify-content:space-between;margin-bottom:4px">
            <span style="color:#ef4444;font-size:0.85rem;font-weight:600">DOWN {down_prob:.0%}</span>
            <span style="color:#22c55e;font-size:0.85rem;font-weight:600">UP {up_prob:.0%}</span>
        </div>
        <div style="background:#21262d;border-radius:4px;height:10px;overflow:hidden;display:flex">
            <div style="background:#ef4444;width:{down_prob:.0%};height:100%"></div>
            <div style="background:#22c55e;width:{up_prob:.0%};height:100%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _headline_card(title: str, source: str, date_str: str, sentiment: float | None = None) -> None:
    """Render a single headline as a card."""
    if sentiment is not None:
        s_color = "#22c55e" if sentiment > 0.1 else "#ef4444" if sentiment < -0.1 else "#f59e0b"
        badge = (f'<span style="background:{s_color}20;color:{s_color};padding:2px 10px;'
                 f'border-radius:12px;font-size:0.8rem;font-weight:600;white-space:nowrap">'
                 f'{sentiment:+.2f}</span>')
    else:
        badge = ""

    st.markdown(f"""
    <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;
                padding:10px 14px;margin-bottom:6px">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:12px">
            <div>
                <div style="font-weight:500;font-size:0.92rem;color:#e6edf3">{title}</div>
                <div style="color:#8b949e;font-size:0.78rem;margin-top:2px">{source} · {date_str}</div>
            </div>
            {badge}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render() -> None:
    # ── Ticker selector row ───────────────────────────────────────────────────
    col_sel, col_btn, _ = st.columns([2, 1, 3])

    with col_sel:
        ticker = st.selectbox(
            "Ticker", TICKERS_SORTED,
            index=TICKERS_SORTED.index("AAPL") if "AAPL" in TICKERS_SORTED else 0,
            label_visibility="collapsed",
        )

    with col_btn:
        run = st.button("Predict", type="primary", use_container_width=True)

    # ── Ticker info header ────────────────────────────────────────────────────
    sector = TICKER_SECTOR_MAP.get(ticker, "")
    info = _fetch_ticker_info(ticker)
    name = info["name"]
    price = info["price"]
    change = info["change_pct"]

    info_parts = [f"<b style='color:#e6edf3;font-size:1.1rem'>{name}</b>"]
    info_parts.append(f"<span style='color:#8b949e'>{sector}</span>")
    if price:
        info_parts.append(f"<span style='color:#e6edf3;font-weight:600'>${price:,.2f}</span>")
    if change is not None:
        c_color = "#22c55e" if change >= 0 else "#ef4444"
        info_parts.append(f"<span style='color:{c_color};font-weight:600'>{change:+.2f}%</span>")
    if info["low_52w"] and info["high_52w"]:
        info_parts.append(
            f"<span style='color:#8b949e;font-size:0.85rem'>52W: ${info['low_52w']:,.0f} — ${info['high_52w']:,.0f}</span>"
        )

    st.markdown(
        f"<p style='margin:0 0 0.5rem;font-size:0.95rem'>{' &nbsp;·&nbsp; '.join(info_parts)}</p>",
        unsafe_allow_html=True,
    )

    # ── Layout: chart + prediction side by side ───────────────────────────────
    if run:
        col_chart, col_pred = st.columns([3, 1])

        with col_chart:
            with st.spinner("Loading chart..."):
                df = _fetch_market_data(ticker)
            if not df.empty:
                _candlestick_chart(df)

        with col_pred:
            with st.spinner("Running model pipeline..."):
                try:
                    result = get_predictor().predict(ticker)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    return

            pred = result["prediction"]
            conf = result["confidence"]
            probs = result["probabilities"]

            _prediction_card(pred, conf, probs)

            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;
                        padding:12px;margin-top:8px">
                <div style="display:flex;justify-content:space-between">
                    <span style="color:#8b949e;font-size:0.85rem">Market date</span>
                    <span style="color:#e6edf3;font-size:0.85rem">{result['market_date']}</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:4px">
                    <span style="color:#8b949e;font-size:0.85rem">Headlines used</span>
                    <span style="color:#e6edf3;font-size:0.85rem">{result['n_headlines']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Headlines section ─────────────────────────────────────────────────
        st.markdown("<hr style='margin:1rem 0 0.8rem;border-color:#21262d'>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='color:#8b949e;font-size:0.9rem;font-weight:600;margin:0 0 0.5rem'>"
            f"Recent Headlines — {ticker}</p>",
            unsafe_allow_html=True,
        )

        try:
            from src.data_collection.news_scraper import load_ticker_news
            news = load_ticker_news(ticker)
            if not news.empty:
                news["published"] = pd.to_datetime(news["published"]).dt.strftime("%b %d")
                for _, row in news.head(6).iterrows():
                    _headline_card(
                        row["title"],
                        row.get("source", ""),
                        row["published"],
                    )
            else:
                st.caption("No saved headlines for this ticker.")
        except FileNotFoundError:
            st.caption("No saved news file found.")

    else:
        # Show chart only (no prediction yet)
        with st.spinner("Loading chart..."):
            df = _fetch_market_data(ticker)
        if not df.empty:
            _candlestick_chart(df)
        st.markdown(
            "<p style='color:#8b949e;text-align:center;margin-top:1rem'>"
            "Click <b>Predict</b> to generate a 5-day forecast</p>",
            unsafe_allow_html=True,
        )

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown(
        "<p style='color:#484f58;font-size:0.75rem;margin-top:1.5rem'>"
        "Research prototype — not financial advice. Past patterns do not guarantee future returns.</p>",
        unsafe_allow_html=True,
    )
