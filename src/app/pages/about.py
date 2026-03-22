"""about.py — Project info page."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.markdown("""
<div style='max-width:760px;margin:0 auto'>

<h1 style='font-size:2rem !important;margin-bottom:4px'>Financial Market Predictor</h1>
<p style='color:#64748b;font-size:0.95rem;margin-bottom:1.5rem'>
    ZHAW AI Applications Module &middot; 2026
</p>

<div class="glass-card">
<h3 style='color:#60a5fa;margin-top:0'>What it does</h3>
<p style='color:#94a3b8;line-height:1.7'>
Forecasts short-term and medium-term price direction (<b style="color:#10b981">UP</b> /
<b style="color:#ef4444">DOWN</b>) for 67 S&amp;P 500 stocks by fusing three independent
signal sources into a unified ML model.
</p>

<table style='width:100%;border-collapse:collapse;margin:0.8rem 0 0'>
<tr style='border-bottom:1px solid #1e293b'>
  <td style='padding:12px 14px;color:#10b981;font-weight:700;font-size:0.95rem;width:90px'>Market</td>
  <td style='padding:12px 14px;color:#94a3b8;font-size:0.9rem;line-height:1.5'>
    28 technical indicators from 5+ years of daily OHLCV data<br>
    <span style='color:#475569;font-size:0.82rem'>RSI, MACD, Bollinger Bands, ATR, VIX level, sector dummies, cyclical time features</span>
  </td>
</tr>
<tr style='border-bottom:1px solid #1e293b'>
  <td style='padding:12px 14px;color:#f59e0b;font-weight:700;font-size:0.95rem'>NLP</td>
  <td style='padding:12px 14px;color:#94a3b8;font-size:0.9rem;line-height:1.5'>
    23 sentiment features from 6,111 financial news headlines<br>
    <span style='color:#475569;font-size:0.82rem'>FinBERT + VADER sentiment, PCA embeddings, dynamic features
    (momentum, surprise, dispersion), sector/market fallback</span>
  </td>
</tr>
<tr>
  <td style='padding:12px 14px;color:#60a5fa;font-weight:700;font-size:0.95rem'>Chart CV</td>
  <td style='padding:12px 14px;color:#94a3b8;font-size:0.9rem;line-height:1.5'>
    10 visual features from 41,000+ candlestick chart images<br>
    <span style='color:#475569;font-size:0.82rem'>EfficientNet-B0 transfer learning embeddings + PCA reduction (every 2nd trading day)</span>
  </td>
</tr>
</table>
</div>

<div class="glass-card">
<h3 style='color:#60a5fa;margin-top:0'>Prediction horizons</h3>
<div style='display:flex;gap:12px;flex-wrap:wrap'>
    <div style='flex:1;min-width:200px;background:#111827;border:1px solid #1e293b;
                border-radius:10px;padding:16px;text-align:center'>
        <div style='font-size:2rem;font-weight:800;color:#3b82f6'>5D</div>
        <div style='color:#94a3b8;font-size:0.85rem;margin-top:4px'>Short-term<br>5 trading days</div>
    </div>
    <div style='flex:1;min-width:200px;background:#111827;border:1px solid #1e293b;
                border-radius:10px;padding:16px;text-align:center'>
        <div style='font-size:2rem;font-weight:800;color:#8b5cf6'>21D</div>
        <div style='color:#94a3b8;font-size:0.85rem;margin-top:4px'>Medium-term<br>~1 month</div>
    </div>
</div>
<p style='color:#475569;font-size:0.85rem;margin-top:12px;margin-bottom:0'>
    Binary classification &mdash; <span style='color:#10b981'>UP</span> = forward return &gt; 0%
    &nbsp;|&nbsp; <span style='color:#ef4444'>DOWN</span> = forward return &le; 0%
</p>
</div>

<div class="glass-card">
<h3 style='color:#60a5fa;margin-top:0'>Models</h3>
<p style='color:#94a3b8;line-height:1.7;margin-bottom:0'>
    <b>RandomForest</b> (300 trees, balanced) and <b>LightGBM</b> (Optuna-tuned, 40 trials)
    are trained per configuration. A <b>StackingClassifier</b> (RF + XGBoost + LightGBM &rarr;
    LogisticRegression meta-learner) is also evaluated.<br>
    <span style='color:#475569'>5-fold TimeSeriesSplit cross-validation &middot; Temporal train/val/test split (no leakage)</span>
</p>
</div>

<div class="glass-card">
<h3 style='color:#60a5fa;margin-top:0'>Data sources</h3>
<ul style='color:#94a3b8;padding-left:1.2rem;line-height:1.8;margin-bottom:0'>
    <li>Market data: <b>Yahoo Finance</b> via yfinance (2020&ndash;2026)</li>
    <li>News: <b>RSS feeds</b> (Reuters, Yahoo Finance, MarketWatch) + <b>NewsAPI</b></li>
    <li>Pre-trained models: <b>ProsusAI/finbert</b> (HuggingFace), <b>EfficientNet-B0</b> (torchvision)</li>
</ul>
</div>

<div class="glass-card">
<h3 style='color:#60a5fa;margin-top:0'>Ethical considerations</h3>
<ul style='color:#94a3b8;padding-left:1.2rem;line-height:1.8;margin-bottom:0'>
    <li>Stock prediction is inherently uncertain &mdash; models exploit statistical patterns, not causal knowledge</li>
    <li>Past performance does not guarantee future results</li>
    <li>Survivorship bias: only currently listed S&amp;P 500 stocks are included</li>
    <li>NLP sentiment from public news may reflect already-priced-in information</li>
    <li>This system should <b>not</b> be used for real investment decisions</li>
</ul>
</div>

<p class="disclaimer">
Research prototype &mdash; not financial advice. Predictions are for educational purposes only
and do not constitute investment recommendations.
</p>

</div>
""", unsafe_allow_html=True)
