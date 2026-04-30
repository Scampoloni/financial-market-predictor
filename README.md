# Financial Market Predictor

An end-to-end AI application that predicts **5-day stock price direction (UP/DOWN)** for 67 S&P 500 stocks by fusing three complementary signal sources: structured market data (ML), financial news sentiment (NLP), and candlestick chart pattern recognition (CV).

> **Disclaimer:** Research prototype — not financial advice.

**Live Demo:** [financial-market-predictorr.streamlit.app](https://financial-market-predictorr.streamlit.app/)

---

## Screenshots

| Prediction | Model Analysis | NLP + CV |
|:---:|:---:|:---:|
| ![Prediction](docs/screenshots/01_prediction_flow.png) | ![Model Analysis](docs/screenshots/02_model_analysis.png) | ![NLP + CV](docs/screenshots/03_nlp_cv_integration.png) |

---

## What it does

The system tests the hypothesis that combining three independent "views" of the market — technical indicators, language-based sentiment, and visual chart patterns — yields more robust predictions than any single modality alone. An ablation study (Configs A → B → C) quantifies each block's incremental contribution on a held-out 2025 test set.

**Streamlit app includes:**
- Live predictions for any of the 67 tracked tickers
- Interactive Plotly candlestick charts
- Ablation results and per-block feature importance
- RAG-powered news Q&A chatbot

---

## Key Results

All models evaluated on held-out **2025 test data** (temporal split, no leakage).

| Config | Features | # Features | Best Model | CV F1 ± std | Test F1 | Test Acc | Δ vs Baseline |
|--------|----------|-----------|------------|:-----------:|:-------:|:--------:|:-------------:|
| **A** | Market only | 28 | LightGBM | 0.509 ± 0.022 | 0.4892 | 0.4898 | — |
| **B** | Market + NLP | 56 | LightGBM | 0.513 ± 0.015 | 0.4852 | 0.4861 | -0.0040 |
| **C** | Market + NLP + CV | 66 | LightGBM | 0.505 ± 0.023 | **0.4908** | 0.4911 | +0.0016 |

**Interpretation:** ~0.50 F1 remains a realistic ceiling for direction prediction on public data — consistent with the semi-strong Efficient Market Hypothesis. NLP shows mixed impact under sparse coverage; CV adds a modest positive lift when combined with market features.

**Selection protocol:** Best model per config is chosen by **validation F1 only** (2024H2). The **test set (2025)** is evaluated once for final reporting.

| Modality | Contribution | Why it works |
|----------|-------------|--------------|
| Market (ML) | Baseline | Technical indicators capture momentum, volatility, mean-reversion regimes |
| NLP | -0.0040 F1 vs A | Sentiment *changes* can lead price, but coverage remains sparse even with fallback |
| CV | +0.0016 F1 vs A (+0.0056 vs B) | Fine-tuned EfficientNet-B0 captures visual patterns complementary to indicators |

### Multi-Horizon Comparison

The app also supports a **21-day prediction horizon** (Config C equivalent, 61 features), evaluated on the same held-out 2025 test set:

| Horizon | Features | Test F1 | Test Acc | Test Rows |
|---------|----------|:-------:|:--------:|:---------:|
| **5-day** | 66 (Config C) | **0.4908** | 0.4911 | 20,033 |
| **21-day** | 61 (Config C) | 0.4961 | 0.4989 | 18,961 |

**Finding:** The EMH ceiling (~0.50 F1) holds consistently across both prediction horizons, confirming that the signal limitation is structural rather than specific to the 5-day window. The 21-day model offers slightly higher recall on DOWN predictions (0.65 vs 0.59), suggesting chart patterns carry more signal over longer windows.
**Note:** 5-day numbers reflect the validation-based selection protocol update; re-train the 21-day model to align metrics if required.

---

## Architecture

```
DATA COLLECTION
├── Yahoo Finance → OHLCV (69 CSV files, 67 tickers + 2 indices)
├── RSS / NewsAPI  → Financial headlines (8,552 rows across 67 tickers)
└── mplfinance    → Candlestick chart images (61,640+ PNGs, bi-daily)

FEATURE EXTRACTION
├── Market block  : 28 technical indicators + sector encoding       → 28 features
├── NLP block     : FinBERT + VADER + embedding PCA + analyst data  → 24 features
└── CV block      : Fine-tuned EfficientNet-B0 → 1280-dim → PCA    → 10 features

UNIFIED FEATURE MATRIX (per ticker-date)
├── Config A: 28 features  (market only)
├── Config B: 56 features  (+ NLP)
└── Config C: 66 features  (+ CV)      ← best performing

MODEL TRAINING (identical split across all configs)
├── RandomForest       (GridSearch-tuned)
├── LightGBM           (Optuna, 40 trials)
└── StackingClassifier (RF + XGB + LGB meta-ensemble)
    5-fold TimeSeriesSplit · Train ≤ 2024-06, Val 2024H2, Test 2025

STREAMLIT APP
└── Live predictions · Ablation analysis · RAG news chatbot
```

---

## Project Structure

```
financial-market-predictor/
├── app.py                          # Streamlit entry point
├── requirements.txt
├── .env.example                    # API key template
├── src/
│   ├── config.py                   # Central config (paths, tickers, hyperparameters)
│   ├── app/
│   │   ├── pages/
│   │   │   ├── predictor.py        # Live prediction UI (Plotly charts + cards)
│   │   │   ├── model_analysis.py   # Ablation results and feature importance
│   │   │   ├── rag_chat.py         # RAG news Q&A chatbot
│   │   │   └── about.py            # Project overview page
│   │   └── utils.py                # Cached loaders and UI helpers
│   ├── data_collection/
│   │   ├── market_collector.py     # Yahoo Finance OHLCV downloader
│   │   ├── news_scraper.py         # RSS + NewsAPI headline scraper
│   │   └── chart_generator.py      # mplfinance candlestick image generator
│   ├── features/
│   │   ├── market_features.py      # 28 technical indicators + target
│   │   ├── nlp_features.py         # FinBERT/VADER sentiment + sector fallback
│   │   └── cv_features.py          # EfficientNet-B0 embeddings + PCA
│   ├── cv/
│   │   └── chart_classifier.py     # EfficientNet-B0 feature extractor
│   ├── models/
│   │   ├── train_ml.py             # Ablation training pipeline
│   │   ├── predict.py              # LivePredictor (inference)
│   │   └── evaluate.py             # Evaluation visualizations
│   └── nlp/
│       ├── finbert_sentiment.py    # FinBERT sentiment pipeline
│       ├── vader_sentiment.py      # VADER lexicon pipeline
│       └── rag_chatbot.py          # Retrieval-augmented Q&A
├── notebooks/                      # Development records (v1 exploratory phase)
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_ml_baseline.ipynb        # Feature engineering and baseline (3-class, 28 features)
│   ├── 03_nlp_pipeline.ipynb       # NLP sentiment extraction
│   ├── 04_cv_pipeline.ipynb        # Chart embeddings (EfficientNet + PCA)
│   ├── 05_integrated_model.ipynb   # End-to-end Config A/B/C training (3-class)
│   └── 06_evaluation_ablation.ipynb # Full ablation study and error analysis
├── scripts/
│   ├── finetune_cnn.py             # Domain-adapt EfficientNet-B0 on chart labels
│   └── train_21d.py                # 21-day horizon model training
├── data/
│   ├── raw/                        # market_data/, news/, charts/ (gitignored — too large)
│   └── processed/                  # Feature parquets + ablation results
├── models/                         # Saved model artifacts (pkl + pth, tracked in git)
└── tests/                          # pytest test suite
```

---

## Feature Summary

### Market Block (28 features)
Returns (1d/5d/20d), RSI-14, MACD (line/signal/histogram), SMA-20/SMA-50/EMA-12 ratios, Bollinger Bands (upper/lower/width), ATR-14, 20-day volatility, volume ratio, VIX level, day-of-week and month cyclical encoding (sin/cos), sector one-hot dummies.

### NLP Block (24 features)
FinBERT compound score + confidence, VADER compound score, news volume (1d/5d rolling), headline length, 10 FinBERT embedding PCA components, sentiment momentum, sentiment dispersion, 3-day sentiment shift, sentiment surprise (z-score vs 20-day baseline), sentiment × volume interaction, news volume z-score, imputation flag.

**Coverage strategy:** ticker-level → sector-average fallback → market-average fallback → forward-fill. Raises raw 1.7% coverage to ~59%.

### CV Block (10 features)
10 PCA components from 1280-dim EfficientNet-B0 embeddings. Model fine-tuned on chart→direction labels (`scripts/finetune_cnn.py`) rather than using frozen ImageNet weights — this was the key step enabling a positive CV contribution.

---

## Data Sources

| Source | Type | Scale |
|--------|------|-------|
| **Yahoo Finance** (yfinance) | OHLCV + VIX | 67 tickers + 2 indices, 2020–2026 |
| **RSS feeds + NewsAPI** | Reuters, MarketWatch, Yahoo Finance headlines | 8,552 scraped rows across 67 tickers |
| **ProsusAI/finbert** | Pre-trained financial sentiment model | HuggingFace Hub |
| **EfficientNet-B0** | CNN backbone (torchvision) → domain fine-tuned | 61,640 generated chart images |

---

## Ticker Selection Rationale

- **Liquidity and data quality:** focus on large-cap S&P 500 names with reliable OHLCV history and consistent trading calendars.
- **Sector coverage:** include tickers across 7 sectors to avoid sector-specific overfitting and enable sector-level error analysis.
- **Signal diversity:** mix growth (Tech) and defensive sectors (Consumer, Healthcare) to capture different regimes.
- **Practical constraints:** exclude thinly traded or short-history tickers to avoid missing data and unstable indicators.

## NLP Approach Comparison

| Approach | Type | Strengths | Role |
|----------|------|-----------|------|
| VADER | Lexicon/rule-based | Fast, deterministic, robust on short headlines | Baseline signal + fallback |
| FinBERT | Transformer (finance-tuned) | Finance-domain context on earnings/macro language | Primary sentiment + confidence features |
| FinBERT + VADER combined | Ensemble feature fusion | More stable across coverage gaps | Final NLP feature block (Config B/C) |

---

## Development Journey

The initial version reached F1 = 0.34 across three classes (UP/DOWN/SIDEWAYS) on a next-day horizon — barely above random. Seven root-cause fixes drove the final result:

| Phase | Change | Why |
|-------|--------|-----|
| 1 | 3-class next-day → 5-day binary | Doubles signal-to-noise; removes ill-defined SIDEWAYS band (60%+ of data) |
| 2 | NLP fallback strategy | Raw 1.7% coverage makes NLP features a no-op; sector/market fallback gets to 59% |
| 3 | Chart generation: monthly → bi-daily | 1.7% CV coverage → 59%; 2,788 → 61,640 images |
| 4 | LightGBM + Optuna + Stacking | Single default RF leaves F1 on the table; multi-model comparison removes selection bias |
| 5 | `st.cache_resource` / `st.cache_data` | App reloaded models on every click; now <3s after initial load |
| 6 | Plotly dark-theme UI | matplotlib line charts + default Streamlit styling; replaced with interactive financial dashboard |
| 7 | CNN fine-tuning + RAG chatbot | Frozen ImageNet weights caused CV regression; domain adaptation on chart→direction labels turned it positive |

---

## Notebook vs. Production Pipeline

The notebooks (`01`–`06`) document the **iterative development process** and contain saved outputs from the exploratory phase (v1: 3-class UP/DOWN/SIDEWAYS, 28 features, F1≈0.34). The production pipeline in `src/` implements the final version (v2: binary UP/DOWN, 28–66 features depending on config, F1≈0.50). The key changes are captured in the [Development Journey](#development-journey) section above. Both versions are intentionally preserved to show the full research arc.

---

## Local Setup

### Prerequisites
- Python 3.11+
- ~10 GB disk space (charts + embeddings)
- GPU optional (CPU inference works; CNN batch takes ~30 min)

### Installation

```bash
git clone https://github.com/Scampoloni/financial-market-predictor.git
cd financial-market-predictor
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # add NEWS_API_KEY and optionally GEMINI_API_KEY
```

### Run the full pipeline

```bash
# 1. Download market data (OHLCV)
python -m src.data_collection.market_collector

# 2. Scrape news headlines
python -m src.data_collection.news_scraper

# 3. Build market features
python -m src.features.market_features

# 4. Build NLP features (FinBERT + VADER + PCA)
python -m src.features.nlp_features

# 5. Generate candlestick charts (bi-daily)
python -m src.data_collection.chart_generator --step 2

# 6. Build CV features (EfficientNet + PCA)
python -m src.features.cv_features

# 7. Train models + run ablation study
python -m src.models.train_ml

# 8. (Optional) Fine-tune EfficientNet-B0 on chart labels
python scripts/finetune_cnn.py --epochs 10

# 9. Launch the app
streamlit run app.py
```

### Tests

```bash
pytest tests/ -q
```

### Smoke-test dataset (reproducible)

For a fast, fully reproducible run without large downloads, use the bundled smoke dataset
(AAPL/MSFT/NVDA + indices, 3 months of data) and the existing `--test` flags:

```bash
# Build smoke dataset from current raw data
python scripts/build_smoke_dataset.py

# Activate smoke dataset (backs up data/raw to data/raw_full)
python scripts/use_smoke_data.py --activate

# Run the smoke pipeline
python -m src.features.market_features --test
python -m src.features.nlp_features --test
python -m src.data_collection.chart_generator --test --step 2
python -m src.features.cv_features --test
python -m src.models.train_ml --config C

# Restore full dataset
python scripts/use_smoke_data.py --restore
```

---

## Tech Stack

| Layer | Libraries |
|-------|-----------|
| ML | scikit-learn, LightGBM, XGBoost, Optuna |
| NLP | HuggingFace Transformers (FinBERT), NLTK (VADER), sentence-transformers (RAG) |
| CV | PyTorch, torchvision (EfficientNet-B0) |
| Data | pandas, numpy, pyarrow, yfinance, feedparser |
| Visualization | Plotly, matplotlib, mplfinance |
| App | Streamlit |
| Evaluation | SHAP, pytest |

---

## Ethical Considerations

- Predictions are uncertain by nature — the model is wrong roughly half the time
- **Not investment advice.** Never use this for real capital allocation
- **Survivorship bias:** only currently-listed S&P 500 stocks; delisted companies are excluded
- **Source bias:** news data is English-only and limited to select RSS/NewsAPI sources, which may skew sentiment.
- **Information leakage risk:** widely disseminated signals can reduce edge (self-fulfilling/decaying alpha).
- **Access inequality:** data availability and compute resources may advantage larger market participants.
- Public news may reflect already-priced-in information (semi-strong EMH)
- Past performance on the 2025 test set does not guarantee future performance
