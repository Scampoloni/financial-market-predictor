"""about.py — Project info page."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.markdown("""
<div style='max-width:720px;margin:0 auto'>

<h2 style='color:#e6edf3'>Financial Market Predictor</h2>
<p style='color:#8b949e'>ZHAW AI Applications Module · 2026</p>

<hr style='border-color:#30363d;margin:1rem 0'>

<h4 style='color:#58a6ff'>What it does</h4>
<p>
Forecasts 5-day price direction (<b>UP / DOWN</b>) for 67 S&amp;P 500 stocks
by combining three signal sources:
</p>

<table style='width:100%;border-collapse:collapse;margin:0.5rem 0 1.2rem'>
<tr style='border-bottom:1px solid #30363d'>
  <td style='padding:0.5rem 0.8rem;color:#3fb950;font-weight:600'>Market</td>
  <td style='padding:0.5rem 0.8rem;color:#8b949e'>28 technical indicators from 5 years of OHLCV data<br>
  <small>RSI, MACD, Bollinger Bands, ATR, VIX level, sector dummies</small></td>
</tr>
<tr style='border-bottom:1px solid #30363d'>
  <td style='padding:0.5rem 0.8rem;color:#d29922;font-weight:600'>NLP</td>
  <td style='padding:0.5rem 0.8rem;color:#8b949e'>23 sentiment features from 6,111 news headlines<br>
  <small>FinBERT + VADER sentiment, PCA embeddings, dynamic features (shifts, surprise, dispersion)</small></td>
</tr>
<tr>
  <td style='padding:0.5rem 0.8rem;color:#388bfd;font-weight:600'>Chart CV</td>
  <td style='padding:0.5rem 0.8rem;color:#8b949e'>10 visual features from 41,000+ candlestick chart images<br>
  <small>EfficientNet-B0 embeddings + PCA (every 2nd trading day)</small></td>
</tr>
</table>

<h4 style='color:#58a6ff'>Target definition</h4>
<p style='color:#8b949e'>
<b>Binary classification</b> — 5-day forward return:<br>
<span style='color:#22c55e'>UP</span> = return &gt; 0% &nbsp;|&nbsp;
<span style='color:#ef4444'>DOWN</span> = return &le; 0%<br>
This replaces the earlier 3-class (UP/DOWN/SIDEWAYS) target which proved too noisy.
</p>

<h4 style='color:#58a6ff'>Models</h4>
<p style='color:#8b949e'>
Ensemble of <b>RandomForest + XGBoost + LightGBM</b> (Optuna-tuned) via StackingClassifier
with LogisticRegression meta-learner. 5-fold TimeSeriesSplit cross-validation.
</p>

<h4 style='color:#58a6ff'>Data sources</h4>
<ul style='color:#8b949e;padding-left:1.2rem'>
<li>Market data: <b>Yahoo Finance</b> via yfinance</li>
<li>News: <b>RSS feeds</b> (Reuters, Yahoo Finance, MarketWatch) + <b>NewsAPI</b></li>
<li>Pre-trained models: <b>ProsusAI/finbert</b> (HuggingFace), <b>EfficientNet-B0</b> (torchvision)</li>
</ul>

<h4 style='color:#58a6ff'>Ethical considerations</h4>
<ul style='color:#8b949e;padding-left:1.2rem'>
<li>Stock prediction is inherently uncertain — models exploit statistical patterns, not causal knowledge</li>
<li>Past performance does not guarantee future results</li>
<li>Survivorship bias: only currently listed S&amp;P 500 stocks are included</li>
<li>NLP sentiment from public news may reflect already-priced-in information</li>
<li>This system should not be used for real investment decisions</li>
</ul>

<hr style='border-color:#30363d;margin:1rem 0'>

<p style='color:#484f58;font-size:0.8rem'>
Research prototype — not financial advice. Predictions are for educational purposes only
and do not constitute investment recommendations.
</p>

</div>
""", unsafe_allow_html=True)
