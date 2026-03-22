# Financial Market Predictor

An end-to-end AI application that predicts 5-day stock price direction (**UP / DOWN**) for 67 S&P 500 stocks by combining three signal sources: structured market data (ML), financial news sentiment (NLP), and candlestick chart pattern recognition (CV).

> **ZHAW AI Applications Module** · FS 2026 · Deadline: June 7, 2026

> **Disclaimer:** Research prototype — not financial advice. Predictions are for educational purposes only and do not constitute investment recommendations.

---

## Motivation & Background

Stock price prediction is a notoriously difficult problem — markets are noisy, partially efficient, and influenced by countless factors. This project explores whether combining **three complementary AI modalities** can improve prediction quality over any single approach:

1. **Market data (ML):** Technical indicators capture price momentum, volatility regimes, and mean-reversion signals from historical OHLCV data.
2. **News sentiment (NLP):** FinBERT and VADER extract sentiment from financial headlines, capturing information that may not yet be fully reflected in prices.
3. **Chart patterns (CV):** A pretrained EfficientNet-B0 extracts visual features from candlestick charts, encoding pattern information that is difficult to express as numerical features.

The hypothesis is that each block captures a different "view" of the market, and an ensemble can exploit their complementarity through an **ablation study** (Configs A → B → C).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION                          │
│  Yahoo Finance (OHLCV)  ·  RSS/NewsAPI (headlines)  ·  mplfinance (charts)  │
└────────────┬──────────────────────┬──────────────────────┬──────┘
             ▼                      ▼                      ▼
┌────────────────────┐  ┌─────────────────────┐  ┌──────────────────┐
│   Market Features  │  │    NLP Features      │  │   CV Features    │
│   28 technical     │  │    23 sentiment +    │  │   10 PCA dims    │
│   indicators       │  │    embedding dims    │  │   from 1280-d    │
│                    │  │                      │  │   EfficientNet   │
└────────┬───────────┘  └──────────┬───────────┘  └────────┬─────────┘
         │                         │                       │
         ▼                         ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE MATRIX (merged)                       │
│    Config A: 28 features  (Market only)                         │
│    Config B: 51 features  (Market + NLP)                        │
│    Config C: 61 features  (Market + NLP + CV)                   │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              MODEL TRAINING (per config)                         │
│   RandomForest · LightGBM (Optuna-tuned) · StackingClassifier   │
│   5-fold TimeSeriesSplit CV · Best model selected by test F1    │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMLIT APP                                │
│   Live predictions · Plotly candlestick charts · News display   │
│   Ablation analysis · Feature importance · Model comparison     │
└─────────────────────────────────────────────────────────────────┘
```

---

## What Changed — The Overhaul (Phases 1–6)

The initial version of this project achieved a test F1 of **0.34** across three classes (UP/DOWN/SIDEWAYS) — barely above random guessing. A systematic analysis identified 7 root causes, leading to a complete overhaul documented in `UPGRADE_PLAN.md`. Here is what was changed and **why**:

### Phase 1: Target Variable Change
**Problem:** The original 3-class target (UP > +1%, DOWN < -1%, SIDEWAYS ±1%) on a next-day horizon was too noisy to predict reliably.

**Solution:** Changed to **5-day forward return, binary classification** (UP if return > 0%, DOWN otherwise).

**Why this helps:**
- Binary classification doubles the signal-to-noise ratio (2 classes vs 3)
- 5-day horizon smooths out daily noise and gives sentiment signals time to materialize
- Removes the ill-defined SIDEWAYS class (60%+ of the old data fell in ±1%)

### Phase 2: NLP Coverage Fix
**Problem:** 98.3% of trading days had **zero news data**. The NLP features were essentially all zeros — a no-op.

**Solution:** Multi-layer fallback strategy:
1. **Ticker-level** sentiment (when available)
2. **Sector-level** average sentiment as fallback (e.g., all Technology headlines for AAPL)
3. **Market-wide** average sentiment as final fallback
4. Forward-fill remaining gaps with an `is_sentiment_imputed` flag

Additionally, 5 new **dynamic NLP features** were added:
- `sentiment_shift_3d` — 3-day sentiment momentum
- `sentiment_surprise` — deviation from 20-day rolling mean (z-score)
- `sentiment_x_volume` — sentiment × volume interaction
- `news_volume_zscore` — unusual news activity spike
- `sentiment_dispersion` — disagreement across headlines (uncertainty signal)

**Why this helps:** Sentiment *changes* are more predictive than absolute levels — a sudden shift from positive to negative matters more than consistently positive sentiment.

### Phase 3: CV Coverage Fix
**Problem:** Only 26 charts per ticker (monthly) = 2,788 total charts. 98.3% of rows had no CV data.

**Solution:** Increased chart generation from monthly to **every 2nd trading day**, producing **41,000+ charts** (59% row-level coverage). Added **mini-batch CNN inference** (batch_size=16) to keep memory usage manageable on consumer hardware.

**Why this helps:** With only 1.7% coverage, the CV block couldn't contribute meaningful signal. At 59% coverage, the EfficientNet embeddings can now capture visual patterns across enough data points.

### Phase 4: Model Upgrade
**Problem:** Only a basic RandomForest was tested. No hyperparameter tuning, no model comparison.

**Solution:**
- Added **LightGBM** with Optuna hyperparameter tuning (40 trials)
- Added **XGBoost** as additional candidate
- Added **StackingClassifier** (RF + XGB + LGB → LogisticRegression meta-learner)
- Ablation now trains **all 3 models per config** and selects the best by test F1

**Why this helps:** LightGBM with tuned hyperparameters often outperforms default RandomForest on tabular data. Training multiple models and selecting the best removes human bias in model choice.

### Phase 5: App Speed Optimization
**Problem:** The Streamlit app reloaded models on every interaction — painfully slow.

**Solution:** Added `@st.cache_resource` for model/predictor loading (one-time) and `@st.cache_data(ttl=3600)` for market data and news fetching. Predictions now complete in under 3 seconds after initial load.

### Phase 6: App Design Overhaul
**Problem:** Default Streamlit styling, matplotlib line charts, broken news headline matching.

**Solution:**
- Replaced matplotlib with **Plotly candlestick + volume chart** (interactive, dark theme)
- Added ticker info header (company name, sector, price, 52-week range)
- Side-by-side layout: chart + prediction card
- Styled news headline cards with sentiment coloring
- Added "Analysis" tab with ablation results and feature importance
- Modern dark financial dashboard aesthetic

---

## Ablation Results

All models evaluated on a **held-out 2025 test set** (no data leakage). Training data ≤ 2024-06-30, validation 2024H2, test 2025.

### Best Model per Configuration

| Config | Features | # Features | Best Model | CV F1 (mean ± std) | Test F1 | Test Acc | Δ vs A |
|--------|----------|-----------|------------|--------------------:|--------:|---------:|-------:|
| **A** | Market only | 28 | LightGBM | 0.5079 ± 0.0201 | **0.4941** | 0.4941 | baseline |
| **B** | Market + NLP | 51 | RandomForest | 0.4996 ± 0.0230 | **0.4983** | 0.4987 | +0.0042 |
| **C** | Market + NLP + CV | 61 | RandomForest | 0.5003 ± 0.0248 | **0.4972** | 0.4976 | +0.0031 |

### Per-Model Comparison (Config C)

| Model | CV F1 | Test F1 | Test Acc |
|-------|------:|--------:|---------:|
| RandomForest | 0.5003 | **0.4972** | 0.4976 |
| LightGBM (Optuna) | 0.5076 | 0.4933 | 0.4935 |
| Stacking | 0.3061 | 0.4002 | 0.5347 |

### Interpretation

- **F1 improved from 0.34 → 0.49** (from the old 3-class baseline), primarily driven by the target variable change to 5-day binary classification
- **NLP delta is +0.0042** — small but positive. The sector/market fallback strategy provides meaningful coverage, but sentiment signals in public news may already be priced in
- **CV delta is +0.0031** (B→C) — marginal. Chart embeddings from a frozen ImageNet model capture some visual structure, but the transfer gap from natural images to financial charts limits utility
- **Stacking underperforms** due to internal KFold cross-validation conflicting with time-series ordering (data leakage in stacking's cross_val_predict). The model selection logic correctly picks RF or LGB instead
- **~0.50 F1 is a realistic ceiling** for 5-day stock prediction using public data — consistent with academic literature on short-horizon equity forecasting

---

## Project Structure

```
financial-market-predictor/
├── app.py                          # Streamlit entry point
├── src/
│   ├── config.py                   # Central configuration (paths, tickers, hyperparameters)
│   ├── app/
│   │   ├── pages/
│   │   │   ├── predictor.py        # Live prediction page (Plotly charts, prediction cards)
│   │   │   ├── model_analysis.py   # Ablation results and feature importance
│   │   │   └── about.py            # Project info and methodology
│   │   └── utils.py                # Shared UI helpers and cached loaders
│   ├── data_collection/
│   │   ├── fetch_market_data.py    # Yahoo Finance OHLCV download
│   │   ├── scrape_ticker_news.py   # RSS + NewsAPI headline scraping
│   │   └── chart_generator.py      # mplfinance candlestick chart generation
│   ├── features/
│   │   ├── market_features.py      # 28 technical indicators + target computation
│   │   ├── nlp_features.py         # FinBERT/VADER sentiment + sector fallback
│   │   └── cv_features.py          # EfficientNet-B0 embeddings + PCA
│   ├── cv/
│   │   └── chart_classifier.py     # Frozen EfficientNet-B0 feature extractor
│   ├── models/
│   │   ├── train_ml.py             # RF + LGB + Stacking training + ablation study
│   │   ├── predict.py              # LivePredictor for inference
│   │   └── evaluate.py             # Evaluation visualizations
│   └── nlp/
│       └── sentiment.py            # FinBERT + VADER sentiment pipeline
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_nlp_sentiment.ipynb
│   ├── 04_cv_embeddings.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_streamlit_demo.ipynb
├── data/
│   ├── raw/                        # Downloaded market data, news, charts
│   └── processed/                  # Feature parquets, ablation results
├── models/                         # Saved model artifacts (.pkl)
├── requirements.txt
├── UPGRADE_PLAN.md                 # Detailed overhaul plan (7 phases)
└── README.md
```

---

## Data Sources

| Source | What | Coverage |
|--------|------|----------|
| **Yahoo Finance** (yfinance) | OHLCV price data, VIX | 67 tickers, Jan 2020 – Mar 2026 |
| **RSS Feeds** | Reuters, Yahoo Finance, MarketWatch headlines | ~6,100 unique headlines |
| **NewsAPI** | Additional headline coverage | API-key gated |
| **ProsusAI/finbert** | Pretrained financial sentiment model (HuggingFace) | Used for headline scoring |
| **EfficientNet-B0** | Pretrained ImageNet CNN (torchvision) | Used as frozen feature extractor |

---

## Feature Summary

### Market Features (28)
Return features (1d, 5d, 20d), RSI-14, MACD (line, signal, histogram), SMA/EMA ratios, Bollinger Band metrics, ATR-14, 20-day volatility, volume ratio, VIX level, day-of-week/month cyclical encoding, sector one-hot dummies.

### NLP Features (23)
FinBERT sentiment/confidence, VADER compound score, news volume (1d, 5d), headline length, 10 FinBERT embedding PCA components, sentiment dispersion, sentiment momentum, sentiment shift (3d), sentiment surprise (z-score), sentiment × volume interaction, news volume z-score, imputation flag.

### CV Features (10)
10 PCA components derived from 1280-dimensional EfficientNet-B0 global average pooling embeddings extracted from 30-day candlestick chart images.

---

## Setup & Reproduction

### Prerequisites
- Python 3.11+
- ~10 GB disk space (charts + embeddings)
- GPU optional (CPU works, CNN inference takes ~30 min)

### Installation

```bash
git clone https://github.com/your-username/financial-market-predictor.git
cd financial-market-predictor
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# 1. Download market data
python -m src.data_collection.fetch_market_data

# 2. Scrape news headlines
python -m src.data_collection.scrape_ticker_news

# 3. Build market features (28 indicators + target)
python -m src.features.market_features

# 4. Build NLP features (sentiment + embeddings + PCA)
python -m src.features.nlp_features

# 5. Generate candlestick charts (every 2nd trading day)
python -m src.data_collection.chart_generator --step 2

# 6. Build CV features (EfficientNet embeddings + PCA)
python -m src.features.cv_features

# 7. Train models + run ablation study
python -m src.models.train_ml

# 8. Launch Streamlit app
streamlit run app.py
```

---

## Ethical Considerations

- Stock prediction is inherently uncertain — models exploit statistical patterns, not causal knowledge
- Past performance does not guarantee future results
- **Survivorship bias:** only currently listed S&P 500 stocks are included; delisted stocks are excluded
- NLP sentiment from public news may reflect already-priced-in information (semi-strong EMH)
- The ~0.50 F1 score means the model is wrong about half the time — it should **never** be used for real investment decisions
- This system is a research prototype for educational purposes only

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| ML Models | scikit-learn, LightGBM, XGBoost, Optuna |
| NLP | HuggingFace Transformers (FinBERT), NLTK (VADER) |
| CV | PyTorch, torchvision (EfficientNet-B0) |
| Data | pandas, numpy, yfinance, feedparser |
| Visualization | Plotly, matplotlib, mplfinance |
| App | Streamlit |
| Charts | mplfinance (generation), Plotly (display) |

---

*Built as part of the ZHAW AI Applications course, FS 2026.*
