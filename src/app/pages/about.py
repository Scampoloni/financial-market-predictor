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
Forecasts next-day price direction (<b>UP / DOWN / SIDEWAYS</b>) for 67 S&amp;P 500 stocks
by combining three signal sources:
</p>

<table style='width:100%;border-collapse:collapse;margin:0.5rem 0 1.2rem'>
<tr style='border-bottom:1px solid #30363d'>
  <td style='padding:0.5rem 0.8rem;color:#3fb950;font-weight:600'>Market</td>
  <td style='padding:0.5rem 0.8rem;color:#8b949e'>28 technical indicators from 5 years of OHLCV data<br>
  <small>RSI, MACD, Bollinger Bands, ATR, VIX regime, sector</small></td>
</tr>
<tr style='border-bottom:1px solid #30363d'>
  <td style='padding:0.5rem 0.8rem;color:#d29922;font-weight:600'>NLP</td>
  <td style='padding:0.5rem 0.8rem;color:#8b949e'>18 sentiment features from 6,111 news headlines<br>
  <small>FinBERT (financial domain), VADER, PCA embeddings</small></td>
</tr>
<tr>
  <td style='padding:0.5rem 0.8rem;color:#388bfd;font-weight:600'>Chart CV</td>
  <td style='padding:0.5rem 0.8rem;color:#8b949e'>10 visual features from 2,788 candlestick chart images<br>
  <small>EfficientNet-B0 embeddings + PCA</small></td>
</tr>
</table>

<h4 style='color:#58a6ff'>Model performance</h4>
<p style='color:#8b949e;font-size:0.9rem'>
RandomForest · 5-fold TimeSeriesSplit CV · Macro F1 on 2025 held-out test set
</p>

<table style='width:100%;border-collapse:collapse;margin:0.5rem 0 1.2rem'>
<tr style='border-bottom:1px solid #30363d;color:#8b949e;font-size:0.85rem'>
  <th style='text-align:left;padding:0.4rem 0.8rem'>Config</th>
  <th style='padding:0.4rem 0.8rem'>Features</th>
  <th style='padding:0.4rem 0.8rem'>Test F1</th>
  <th style='padding:0.4rem 0.8rem'>vs baseline</th>
</tr>
<tr style='border-bottom:1px solid #21262d'>
  <td style='padding:0.4rem 0.8rem'>A — Market only</td>
  <td style='padding:0.4rem 0.8rem;text-align:center'>28</td>
  <td style='padding:0.4rem 0.8rem;text-align:center'>0.3415</td>
  <td style='padding:0.4rem 0.8rem;text-align:center;color:#8b949e'>baseline</td>
</tr>
<tr style='border-bottom:1px solid #21262d'>
  <td style='padding:0.4rem 0.8rem'>B — Market + NLP</td>
  <td style='padding:0.4rem 0.8rem;text-align:center'>46</td>
  <td style='padding:0.4rem 0.8rem;text-align:center'>0.3430</td>
  <td style='padding:0.4rem 0.8rem;text-align:center;color:#3fb950'>+0.0015</td>
</tr>
<tr>
  <td style='padding:0.4rem 0.8rem'>C — Market + NLP + CV</td>
  <td style='padding:0.4rem 0.8rem;text-align:center'>56</td>
  <td style='padding:0.4rem 0.8rem;text-align:center'>0.3443</td>
  <td style='padding:0.4rem 0.8rem;text-align:center;color:#3fb950'>+0.0028</td>
</tr>
</table>

<h4 style='color:#58a6ff'>Data sources</h4>
<ul style='color:#8b949e;padding-left:1.2rem'>
<li>Market data: <b>Yahoo Finance</b> via yfinance</li>
<li>News: <b>RSS feeds</b> (Reuters, Yahoo Finance, MarketWatch) + <b>NewsAPI</b></li>
<li>Pre-trained models: <b>ProsusAI/finbert</b> (HuggingFace), <b>EfficientNet-B0</b> (torchvision)</li>
</ul>

<hr style='border-color:#30363d;margin:1rem 0'>

<p style='color:#484f58;font-size:0.8rem'>
Research prototype — not financial advice. Predictions are for educational purposes only
and do not constitute investment recommendations.
</p>

</div>
""", unsafe_allow_html=True)
