"""about.py — Project description, data sources, methodology, ethics."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.header("About — Financial Market Predictor")

    st.markdown("""
## Project Overview

This is a **multi-modal stock movement predictor** built as part of the ZHAW AI Applications module.
The system forecasts next-day price direction (UP / SIDEWAYS / DOWN) for 67 S&P 500 stocks
using three complementary signal sources:

| Feature Block | Source | Method |
|---|---|---|
| **Market** | yfinance OHLCV (2020–2025) | Technical indicators, VIX, sector |
| **NLP** | RSS feeds + NewsAPI headlines | FinBERT sentiment + VADER + PCA embeddings |
| **CV** | Candlestick chart images | EfficientNet-B0 embeddings + PCA |

---

## Ablation Study Results

Three model configurations are trained to quantify each block's contribution:

| Config | Features | Test F1 | Delta |
|---|---|---|---|
| A — Market only | 28 | 0.3415 | baseline |
| B — Market + NLP | 46 | 0.3430 | +0.0015 |
| C — Market + NLP + CV | 56 | 0.3443 | +0.0028 |

The primary metric is **macro-averaged F1** on the held-out 2025 test set.
A three-class random baseline would achieve F1 ≈ 0.33, so all configs beat chance.

---

## Methodology

### Data Pipeline
- **Market data**: `yfinance` downloads daily OHLCV for 2020-01-01 to 2025-03-01
- **Feature engineering**: RSI, MACD, Bollinger Bands, ATR, VIX regime, month/weekday cyclical encoding
- **News collection**: RSS feeds (Reuters, Yahoo Finance, MarketWatch) + NewsAPI (67 tickers)
- **Chart generation**: 30-day rolling candlestick PNGs at 224×224px (mplfinance, headless)

### Model
- **Classifier**: RandomForestClassifier (300 trees, max_depth=10, class_weight=balanced)
- **Validation**: 5-fold TimeSeriesSplit cross-validation
- **Temporal split**: train ≤ 2024-06-30 | val 2024-07–2024-12 | test 2025

### NLP Pipeline
- **FinBERT** (ProsusAI/finbert): financial domain sentiment (positive/neutral/negative)
- **VADER**: rule-based compound score as baseline/complement
- **PCA**: 10 principal components from 768-dim FinBERT CLS embeddings (62.7% variance)

### CV Pipeline
- **EfficientNet-B0** (ImageNet weights, frozen): 1280-dim penultimate layer embeddings
- **PCA**: 10 principal components (48.6% variance on 20,569 chart-covered rows)

---

## Data Sources

- **Market data**: [Yahoo Finance](https://finance.yahoo.com) via `yfinance`
- **News**: RSS feeds from Reuters, Yahoo Finance, MarketWatch; NewsAPI.org
- **Pre-trained models**: HuggingFace Hub (`ProsusAI/finbert`), torchvision (`efficientnet_b0`)
- **VIX**: CBOE Volatility Index (`^VIX`) via yfinance

---

## Ethical Considerations

- **Not financial advice**: This system is a research prototype for educational purposes only.
  Past patterns in the training data do not guarantee future returns.
- **Market impact**: Predictions should not be used for automated trading at scale.
- **Data recency**: NewsAPI free tier returns only the most recent ~100 articles per ticker.
  Historical NLP coverage is limited, which contributes to the small Config B/C deltas.
- **Model fairness**: The model is trained on large-cap US equities only.
  Predictions for other markets or asset classes are not supported.

---

## Tech Stack

```
Python 3.13 · Streamlit · scikit-learn · XGBoost · PyTorch · torchvision
transformers (FinBERT) · mplfinance · yfinance · feedparser · rapidfuzz
pandas · numpy · matplotlib · seaborn
```
    """)
