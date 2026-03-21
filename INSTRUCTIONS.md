# INSTRUCTIONS.md — Financial News & Market Predictor
# Master Context File for AI Coding Assistants (Claude Code / GitHub Copilot)
# ALWAYS READ THIS FILE COMPLETELY BEFORE MAKING ANY CHANGES

---

## 1. PROJECT OVERVIEW

### What We're Building
An end-to-end AI application that predicts short-term stock price movements
by combining three AI blocks:
- ML on structured market data (Yahoo Finance)
- NLP on financial news sentiment (FinBERT + RSS feeds)
- CV on candlestick chart pattern recognition (CNN on rendered charts)

### Core Scientific Question
"Do NLP features from financial news sentiment and CV features from
candlestick chart pattern recognition measurably improve the prediction
of short-term stock price movements compared to a model using only
structured market data?"

This is answered through a formal Ablation Study:
  Model A: Market features only (baseline)
  Model B: Market + NLP features
  Model C: Market + NLP + CV features (bonus)
  → Each model is evaluated identically. Deltas are reported.

### Academic Context
- Institution: ZHAW School of Management and Law
- Module: AI Applications (FS 2026)
- Deadline: June 7, 2026, 18:00 CET
- Target Grade: 6.0 (maximum in Swiss system)
- ECTS: 3 (≈ 90 hours total budget)
- Graders: Jasmin Heierli (jasminh), Benjamin Kühnis (bkuehnis)
  → Both must be added as collaborators to the GitHub repository.

### Why This Project (CV Narrative)
This project demonstrates applied AI at the intersection of finance,
data science, and multi-modal prediction — combining structured financial
data, NLP on real-time news, and computer vision on market charts.
It reflects genuine interest in data-driven finance and showcases
the ability to build end-to-end ML systems with real-world data sources.

---

## 2. BLOCK REQUIREMENTS (NON-NEGOTIABLE)

These are the official grading criteria. Every requirement below must be
demonstrably fulfilled.

### Block 1: ML Numeric Data (REQUIRED)
- [ ] At least one structured/numeric dataset used (Yahoo Finance market data)
- [ ] EDA with visualizations (distributions, correlations, anomalies)
- [ ] Feature engineering and feature selection
- [ ] At least TWO different models trained and compared
      (e.g. XGBoost vs. Random Forest, plus optional LSTM or Stacking)
- [ ] Quantitative evaluation: Accuracy, F1-Score, AUC-ROC (classification)
      or RMSE, MAE, R² (regression) on held-out test set
- [ ] 5-fold cross-validation (time-series aware: no future leakage)
- [ ] Error analysis (where does the model fail? Which sectors? Which volatility regimes?)
- [ ] Clear explanation of how ML uses NLP/CV outputs as features

### Block 2: NLP (REQUIRED)
- [ ] Clear definition of text data: financial news articles and headlines
- [ ] NLP-specific preprocessing: cleaning, tokenization, headline extraction
- [ ] At least ONE NLP approach implemented:
      Option A: FinBERT (ProsusAI/finbert — finance-specific sentiment)
      Option B: Zero-shot classification via Gemini or similar LLM
      Option C: TF-IDF + classical sentiment baseline (VADER)
      → At least TWO approaches compared (this is the required comparison)
- [ ] Features extracted: sentiment_score, sentiment_confidence,
      news_volume (count per ticker per day), headline_subjectivity,
      avg_headline_length, entity_mention_count
- [ ] NLP output becomes ML input feature vector
- [ ] Qualitative examples: show 5 events where NLP improved prediction,
      5 where it didn't
- [ ] RAG component: chatbot that answers contextual market questions
      using retrieved news articles (Gemini API or similar)

### Block 3: Computer Vision (BONUS — implement for all-3-blocks bonus)
- [ ] Candlestick chart images generated from OHLCV data using mplfinance
- [ ] Image preprocessing: standardized size (224x224), consistent styling
- [ ] Pretrained model applied: ResNet18 or EfficientNet (fine-tuned)
      OR custom CNN trained on chart patterns
- [ ] Classification task: predict chart pattern category
      (uptrend / downtrend / sideways / reversal) from 30-day chart windows
- [ ] CV output (predicted pattern class + confidence) becomes ML input feature
- [ ] Evaluation: confusion matrix on pattern classification +
      ablation delta when added to ML model

### Integration Requirements (CRITICAL for grade)
The blocks MUST interact — they cannot run independently:
- NLP features → fed as columns into the same ML training dataframe
- CV features → fed as columns into the same ML training dataframe
- Final model sees: market_features + nlp_features + cv_features together
- Ablation study proves each block's added value with a number

---

## 3. DATA SOURCES

### Primary: Yahoo Finance (via yfinance library)
- Library: yfinance (pip install yfinance)
- Auth: None needed (free, no API key)
- Data: OHLCV (Open, High, Low, Close, Volume) daily data
- Tickers: 50-100 tickers from S&P 500 across sectors
  → Technology: AAPL, MSFT, NVDA, GOOGL, META, AMZN, TSLA
  → Finance: JPM, GS, BAC, MS, V, MA, BRK-B
  → Healthcare: JNJ, PFE, UNH, ABBV, MRK
  → Consumer: KO, PEP, MCD, NKE, SBUX
  → Energy: XOM, CVX, COP
  → Insurance: AIG, MET, PRU, ALL, TRV (personal relevance)
  → + 20-30 more for diversity
- Time range: 2020-01-01 to 2025-12-31 (5 years, captures COVID + recovery)
- Store raw data in: data/raw/market_data/

### Secondary: Financial News (RSS feeds + NewsAPI)
- Sources:
  * Yahoo Finance RSS: https://feeds.finance.yahoo.com/rss/2.0/headline
  * Reuters RSS: various sector feeds
  * MarketWatch RSS
  * Optional: NewsAPI.org (free tier: 100 requests/day, 1 month history)
  * Optional: Reddit r/stocks, r/wallstreetbets via PRAW
- For each ticker: collect headlines + article snippets from news date
- Match news to ticker by entity extraction (company name / ticker symbol)
- Store in: data/raw/news/

### Tertiary: Candlestick Chart Images (generated, not downloaded)
- Source: Generated from OHLCV data using mplfinance library
- For each ticker × each 30-day window: render a candlestick chart
- Standardize: same dimensions, same color scheme, no axis labels
  (model should learn from patterns, not read numbers)
- Store in: data/raw/charts/{ticker}_{date}.png
- Label each chart: manually or rule-based using price movement
  in the following 5 days (up >2% = uptrend, down >2% = downtrend, else sideways)

### Data Quality Rules
- Minimum: 5,000 ticker-day observations with both market data AND news
- Exclude: days with zero volume (market closed, data errors)
- Exclude: penny stocks (price < $5) — too volatile, different dynamics
- Deduplicate news by headline similarity (rapidfuzz, threshold 90%)
- Time-series integrity: NO future data leakage in any feature
  → All features use ONLY past/current data relative to prediction date
  → Train/test split must be TEMPORAL (not random)

---

## 4. FEATURE ENGINEERING

### Market Features (from Yahoo Finance — structured numeric data)
| Feature | Type | Notes |
|---------|------|-------|
| return_1d | float | (close - prev_close) / prev_close |
| return_5d | float | 5-day rolling return |
| return_20d | float | 20-day rolling return |
| volatility_20d | float | 20-day rolling std of returns |
| volume_ratio | float | volume / 20-day avg volume |
| rsi_14 | float 0-100 | Relative Strength Index |
| macd | float | MACD line value |
| macd_signal | float | MACD signal line |
| bb_upper_dist | float | distance to upper Bollinger Band |
| bb_lower_dist | float | distance to lower Bollinger Band |
| sma_20_ratio | float | close / SMA(20) |
| sma_50_ratio | float | close / SMA(50) |
| ema_12_ratio | float | close / EMA(12) |
| atr_14 | float | Average True Range (volatility) |
| sector | cat | one-hot encoded sector |
| day_of_week | int 0-4 | cyclical encoding (sin/cos) |
| month | int 1-12 | cyclical encoding (sin/cos) |
| vix_level | float | market fear index (from ^VIX) |

### NLP Features (derived from news, become ML input columns)
| Feature | How | Library |
|---------|-----|---------|
| finbert_sentiment | FinBERT compound score (-1 to 1) | transformers (ProsusAI/finbert) |
| finbert_confidence | FinBERT max class probability | transformers |
| vader_sentiment | VADER compound score | nltk |
| news_volume_1d | count of news articles for ticker on day | manual |
| news_volume_5d | rolling 5-day news count | manual |
| headline_avg_length | mean word count of headlines | manual |
| sentiment_momentum | sentiment today - sentiment 5d ago | manual |
| sentiment_dispersion | std of sentiment scores across articles | manual |
| finbert_embed_pca_{1..10} | 10 PCA dims of FinBERT [CLS] embeddings | transformers + sklearn |

### CV Features (derived from chart images, become ML input columns)
| Feature | How | Library |
|---------|-----|---------|
| chart_pattern | predicted class (uptrend/down/side/reversal) | CNN model |
| chart_pattern_confidence | softmax probability of predicted class | CNN model |
| cnn_embed_pca_{1..10} | 10 PCA dims of CNN penultimate layer | torchvision + sklearn |

---

## 5. MODELS

### Prediction Task
**Primary: Classification** — predict whether a stock will go UP (>1%),
DOWN (<-1%), or SIDEWAYS (-1% to +1%) in the next 5 trading days.

Why classification over regression:
- Stock returns are extremely noisy; regression on exact returns gives poor R²
- Classification into directional buckets is more actionable
- Evaluation metrics (F1, AUC) are more interpretable for this domain
- Grading rubric supports both approaches

**Secondary (optional):** Also report regression results (predict exact 5-day return)
for completeness. Use RMSE, MAE, R².

### Models to Train and Compare (minimum requirement: 2)
Implement all 4 for maximum depth:

1. Logistic Regression — baseline linear model
   - Hyperparams: C in [0.01, 0.1, 1, 10] → GridSearchCV
   - Serves as interpretability benchmark

2. Random Forest Classifier
   - Hyperparams: n_estimators=300, max_depth=[5,10,None], min_samples_leaf=[1,5,10]
   - Good for capturing non-linear interactions

3. XGBoost Classifier (likely best performer)
   - Hyperparams: n_estimators=500, learning_rate=[0.01, 0.05], max_depth=[3,5,7]
   - Use early stopping on temporal validation set
   - Use scale_pos_weight if classes are imbalanced

4. Stacking Ensemble
   - Base estimators: Logistic Regression + Random Forest + XGBoost
   - Meta-learner: Logistic Regression
   - This is the final production model

### Ablation Study (CORE scientific contribution)
Train each model configuration on the SAME temporal train/test split:

| Config | Features Used | Accuracy | F1 (macro) | AUC-ROC |
|--------|--------------|----------|------------|---------|
| A — Market only | 18+ market features | TBD | TBD | TBD |
| B — Market + NLP | Market + 10 NLP features | TBD | TBD | TBD |
| C — Market + NLP + CV | Market + NLP + 12 CV features | TBD | TBD | TBD |

Report deltas. Interpret results. This IS the scientific finding.

### CRITICAL: Time-Series Evaluation (no data leakage)
- Train: 2020-01 to 2024-06
- Validation: 2024-07 to 2024-12 (for hyperparameter tuning)
- Test: 2025-01 to 2025-12 (NEVER seen during development)
- Cross-validation: TimeSeriesSplit with 5 folds (sklearn)
- NEVER shuffle data randomly — temporal order must be preserved

---

## 6. GIT WORKFLOW (MANDATORY — graders will check commit history)

### Repository Setup
1. Create new repo on GitHub: `financial-market-predictor`
2. Clone locally
3. Set up branch protection on main
4. Add graders as collaborators immediately

### Branching Strategy
main          → protected, only merge from dev via PR
dev           → integration branch
feature/data-pipeline      → Yahoo Finance + News collection
feature/eda                → exploratory data analysis notebooks
feature/ml-baseline        → ML models with market features only
feature/nlp-pipeline       → news scraping + FinBERT + sentiment features
feature/cv-pipeline        → chart generation + CNN pattern recognition
feature/rag-chatbot        → RAG component for news Q&A
feature/integration        → combining all features + ablation study
feature/streamlit-app      → deployment UI
feature/evaluation         → ablation study + error analysis

### Commit Message Convention (follow this exactly)
Format: <type>(<scope>): <short description>

Types:
  feat     → new feature
  fix      → bug fix
  data     → data collection or processing
  model    → model training or evaluation
  docs     → documentation
  test     → adding tests
  refactor → code cleanup, no behavior change
  ci       → GitHub Actions

Examples:
  data(market): collect 5yr OHLCV data for 80 S&P 500 tickers
  data(news): implement Yahoo Finance RSS scraper with entity matching
  feat(nlp): add FinBERT sentiment pipeline with batch processing
  model(xgboost): train baseline with market features — F1=0.48
  model(ablation): add NLP features, F1 improves to 0.55 (+0.07)
  feat(cv): generate 30-day candlestick chart images for CNN training
  feat(streamlit): add real-time ticker analysis with SHAP explanations
  docs: update README with deployment instructions and results table

### Pull Request Rules
- Every feature branch gets a PR into dev
- PR description must explain: what was built, what metrics were achieved
- Merge only when tests pass

---

## 7. PROJECT STRUCTURE

```
financial-market-predictor/
├── .github/
│   └── workflows/
│       └── tests.yml          # CI: run pytest on push to dev
├── .env.example               # Template for API keys (NewsAPI, Gemini)
├── .gitignore
├── INSTRUCTIONS.md            # THIS FILE
├── README.md                  # Project overview, results, how to run
├── requirements.txt           # Pinned dependencies
├── app.py                     # Streamlit entry point
│
├── data/
│   ├── raw/
│   │   ├── market_data/       # OHLCV CSVs per ticker
│   │   ├── news/              # News headlines + articles
│   │   └── charts/            # Generated candlestick images
│   ├── processed/
│   │   ├── features_market.parquet
│   │   ├── features_nlp.parquet
│   │   ├── features_cv.parquet
│   │   └── features_combined.parquet
│   └── metadata/
│       └── tickers.csv        # Ticker list with sector info
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_ml_baseline.ipynb
│   ├── 03_nlp_pipeline.ipynb
│   ├── 04_cv_pipeline.ipynb
│   ├── 05_integrated_model.ipynb
│   └── 06_evaluation_ablation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Paths, constants, hyperparams
│   ├── data_collection/
│   │   ├── __init__.py
│   │   ├── market_collector.py    # Yahoo Finance data collection
│   │   ├── news_scraper.py        # RSS feeds + NewsAPI
│   │   └── chart_generator.py     # mplfinance candlestick rendering
│   ├── features/
│   │   ├── __init__.py
│   │   ├── market_features.py     # Technical indicators
│   │   ├── nlp_features.py        # FinBERT + VADER + aggregation
│   │   └── cv_features.py         # CNN inference + embedding extraction
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_ml.py            # Train all ML models + ablation
│   │   ├── evaluate.py            # Metrics, confusion matrix, error analysis
│   │   └── predict.py             # Inference function for deployment
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── finbert_sentiment.py   # FinBERT pipeline
│   │   ├── vader_sentiment.py     # VADER baseline
│   │   └── rag_chatbot.py         # RAG component with Gemini
│   ├── cv/
│   │   ├── __init__.py
│   │   ├── chart_classifier.py    # CNN for chart pattern recognition
│   │   └── chart_features.py      # Extract embeddings from CNN
│   └── app/
│       ├── __init__.py
│       ├── pages/
│       │   ├── predictor.py       # Main prediction interface
│       │   ├── model_analysis.py  # Ablation results + SHAP
│       │   ├── eda_explorer.py    # Dataset insights
│       │   ├── rag_chat.py        # News chatbot interface
│       │   └── about.py           # Project description
│       └── utils.py               # Shared Streamlit helpers
│
├── models/                    # Saved model artifacts (git-tracked)
│   ├── xgboost_final.pkl
│   ├── stacking_final.pkl
│   ├── scaler.pkl
│   ├── pca_nlp.pkl
│   ├── pca_cv.pkl
│   └── chart_cnn.pth         # PyTorch CNN weights
│
└── tests/
    ├── __init__.py
    ├── test_data_pipeline.py
    ├── test_feature_engineering.py
    ├── test_ml_pipeline.py
    └── test_streamlit.py
```

---

## 8. TESTING REQUIREMENTS

### Test Files (pytest)
tests/
├── test_data_pipeline.py   → yfinance returns correct OHLCV schema,
                              news scraper handles missing tickers gracefully,
                              chart generator produces valid PNG files
├── test_feature_engineering.py → technical indicators within expected ranges,
                                   NLP features normalized correctly,
                                   no future leakage in rolling features
├── test_ml_pipeline.py     → model loads and predicts without error,
                               ablation configs produce different results,
                               temporal split enforced (no future dates in train)
└── test_streamlit.py       → app imports cleanly, prediction function works

### Minimum Coverage
- All data collection functions: test with mock/cached responses
- All feature extractors: test with 5 hardcoded example tickers
- All model training functions: test with 100-row sample dataset
- Run tests in CI via GitHub Actions on every push to dev

### GitHub Actions (.github/workflows/tests.yml)
Trigger: push to dev or any feature/* branch
Steps: pip install, run pytest, report coverage

---

## 9. STREAMLIT APP (DEPLOYMENT)

### Target Platform: Hugging Face Spaces (Streamlit SDK)
URL format: https://huggingface.co/spaces/Scampolonii/financial-market-predictor

### App Flow

#### MODE 1 — Live Ticker Analysis
1. User inputs a ticker symbol (e.g. AAPL, NVDA, AIG)
2. App fetches live market data from yfinance (last 60 days)
3. App fetches recent news headlines via RSS/NewsAPI
4. Feature pipelines run:
   - Market features calculated from OHLCV
   - NLP: FinBERT sentiment on recent headlines
   - CV: Generate current 30-day chart, run CNN
5. Saved ML model (Stacking Ensemble) makes prediction
6. App displays:
   - Predicted direction (UP / DOWN / SIDEWAYS) with confidence %
   - SHAP waterfall chart: "RSI_14 contributes +12% toward UP"
   - News sentiment gauge: "Recent news sentiment: 0.67 (bullish)"
   - Current candlestick chart with CNN pattern overlay
   - Feature importance breakdown (market vs NLP vs CV contributions)
   - Ablation info: "NLP features improved model accuracy by +X%"

#### MODE 2 — Historical Backtest View
1. User selects a ticker + date range
2. App shows historical predictions vs actual outcomes
3. Displays rolling accuracy over time
4. Highlights periods where NLP/CV signals were particularly helpful
5. Useful for validating model behavior

### App Pages / Tabs
Tab 1: 📈 Market Predictor — main prediction with live ticker lookup
Tab 2: 📰 News Sentiment — real-time sentiment dashboard across tickers
Tab 3: 💬 Market Chat — RAG chatbot for news Q&A
       ("What happened to NVDA stock last week?")
Tab 4: 📊 Model Analysis — ablation study results, feature importance, SHAP
Tab 5: 🔍 EDA Explorer — dataset insights, sector distributions, correlations
Tab 6: ℹ️ About — project description, data sources, methodology, ethics

### Separation of Training and Inference (required by grading criteria)
- Training code: src/models/train_ml.py (runs offline, saves models)
- Inference code: src/app/ (loads saved models, never retrains)
- Saved artifacts in: models/ directory

---

## 10. DOCUMENTATION REQUIREMENTS

### README.md (root of repo)
Must contain:
- Project title and one-line description
- Scientific hypothesis and approach
- Block combination explanation (ML + NLP + CV)
- Data sources with links
- How to run locally (step by step)
- How to reproduce training
- Deployment URL
- Results table (ablation study summary)
- Ethical considerations (see section 12)
- Authors

### Notebooks (complete, clean, with markdown explanations)
01_eda.ipynb
- Dataset overview: shape, dtypes, missing values
- Distribution plots for all market features
- Return distribution by sector, by year
- Correlation heatmap (features vs. target)
- News volume analysis: do more-covered stocks behave differently?
- Key findings summarized in markdown

02_ml_baseline.ipynb
- Feature matrix construction (market features only)
- Temporal train/val/test split
- Logistic Regression, Random Forest, XGBoost training
- TimeSeriesSplit cross-validation results
- Feature importance plot
- Error analysis: confusion matrix, per-sector performance

03_nlp_pipeline.ipynb
- News data exploration: volume, source distribution
- FinBERT vs VADER comparison on 50 example headlines
- Sentiment distribution over time
- Correlation: sentiment vs. next-day returns
- NLP feature extraction pipeline demonstration

04_cv_pipeline.ipynb (bonus)
- Example candlestick charts displayed
- Chart labeling strategy explained
- CNN training: architecture, epochs, loss curve
- Pattern classification accuracy
- Example predictions: correctly and incorrectly classified charts

05_integrated_model.ipynb
- Combine all feature sets
- Full ablation study across all configurations
- Stacking ensemble training
- SHAP analysis on best model

06_evaluation_ablation.ipynb
- Final results table with all configs
- Statistical significance of improvements (if possible)
- Error analysis across sectors, volatility regimes, time periods
- Limitations and conclusions

---

## 11. CODING STANDARDS

### Python Style
- Python 3.10+
- PEP8 compliant (use black formatter)
- Type hints on all function signatures
- Docstrings on all functions (Google style)

### Project Rules
- No hardcoded API keys — use .env file + python-dotenv
- .env is in .gitignore (never commit credentials)
- Config values in src/config.py (paths, constants, hyperparams)
- Separate data collection from feature engineering from modeling
- All timestamps in UTC

### Requirements
requirements.txt:
yfinance>=0.2.31
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=1.7.0
transformers>=4.35.0
torch>=2.0.0
torchvision>=0.15.0
mplfinance>=0.12.10b0
nltk>=3.8.0
shap>=0.42.0
plotly>=5.17.0
streamlit>=1.28.0
pytest>=7.4.0
python-dotenv>=1.0.0
feedparser>=6.0.0
rapidfuzz>=3.3.0
google-generativeai>=0.3.0
Pillow>=10.0.0

---

## 12. ETHICAL CONSIDERATIONS (include in docs — bonus points)

This section should appear in the documentation and README:

- **Not financial advice:** This model is an academic exercise.
  Predictions should NOT be used for actual trading decisions.
  Past performance does not predict future results.
- **Data bias:** Yahoo Finance data has survivorship bias (delisted
  companies are excluded). News coverage is biased toward large-cap stocks.
- **Model limitations:** Financial markets are influenced by countless
  factors not captured in this model (geopolitics, central bank policy,
  insider activity, black swan events).
- **Responsible AI:** Sentiment analysis can misinterpret sarcasm,
  context-dependent language, and non-English content. FinBERT was
  trained on English financial text and may not generalize.
- **Market impact:** If widely deployed, sentiment-based trading systems
  could create feedback loops and contribute to market instability.

---

## 13. WHAT TO BUILD FIRST (PRIORITY ORDER)

Phase 1 — Foundation (Week 1-2)
1. Create GitHub repo with exact structure from Section 7
2. Create .env.example, .gitignore, requirements.txt
3. Implement market_collector.py — collect 5yr OHLCV for 80 tickers
4. Implement news_scraper.py — collect headlines from RSS feeds
5. Save raw data to data/raw/
6. Commit: data(market): collect 5yr OHLCV for 80 S&P 500 tickers

Phase 2 — EDA + ML Baseline (Week 2-3)
1. Complete 01_eda.ipynb with full visualizations
2. Implement market_features.py (technical indicators)
3. Complete 02_ml_baseline.ipynb (market features only)
4. Establish baseline F1 — this is Config A in ablation study
5. Branch: feature/ml-baseline

Phase 3 — NLP Pipeline (Week 3-5)
1. Implement finbert_sentiment.py
2. Implement vader_sentiment.py
3. Compare FinBERT vs VADER in 03_nlp_pipeline.ipynb
4. Implement nlp_features.py — aggregate per ticker-day
5. Merge NLP features into main dataframe
6. Retrain ML models — report Config B delta
7. Branch: feature/nlp-pipeline

Phase 4 — Integration + Streamlit (Week 5-7)
1. Complete 05_integrated_model.ipynb
2. Run full ablation study (Config A vs B)
3. Build Streamlit app (Mode 1 + tabs)
4. Implement RAG chatbot for news Q&A
5. Deploy to Hugging Face Spaces

Phase 5 — CV Bonus (Week 7-9)
1. Implement chart_generator.py (mplfinance candlestick rendering)
2. Label charts based on subsequent price movement
3. Train CNN chart classifier
4. Extract embeddings, add as features
5. Report Config C delta in ablation
6. Branch: feature/cv-pipeline

Phase 6 — Polish (Week 9-11, before June 7)
1. Complete 06_evaluation_ablation.ipynb
2. Clean all notebooks (restart kernel, run all)
3. Write ethical considerations section
4. Complete README with results table
5. Final commit history review
6. Add both graders as GitHub collaborators:
   - jasminh (Jasmin Heierli)
   - bkuehnis (Benjamin Kühnis)
7. Verify deployed app works end-to-end

---

## 14. KNOWN CONSTRAINTS AND EDGE CASES

- yfinance has no rate limit but can be slow for bulk downloads
  → download in batches, save incrementally, use caching
- News availability varies wildly by ticker: AAPL has thousands of articles,
  small-caps might have 2 per week → handle sparse news gracefully
  → Use sector-level news as fallback when ticker-specific news is sparse
- FinBERT inference is slow → process in batches of 32,
  save embeddings to disk after first run (don't recompute)
- Chart generation: mplfinance can be slow for thousands of charts
  → generate charts in parallel (multiprocessing) or pre-generate and cache
- Time zones: Yahoo Finance uses market-local time (EST for US stocks),
  news sources use various TZs → normalize everything to UTC
- Weekend/holiday gap: no market data on non-trading days
  → forward-fill features, but mark with is_after_gap flag
- Temporal leakage is the #1 risk: ALWAYS verify that no feature
  uses data from after the prediction date. Add assertion tests.

---

## 15. GRADING SELF-CHECK (before final submission)

Run through this list — every item must be YES:

### Requirements
- [ ] At least 2 blocks implemented (ML + NLP minimum)
- [ ] Blocks are integrated (NLP output → ML input), not parallel
- [ ] Multiple data sources used (Yahoo Finance + News RSS minimum)
- [ ] EDA completed with visualizations
- [ ] At least 2 ML models compared
- [ ] At least 1 NLP model comparison (FinBERT vs VADER)
- [ ] Quantitative evaluation with metrics
- [ ] Error analysis included
- [ ] Working deployment with public URL
- [ ] Training and inference separated
- [ ] README complete
- [ ] GitHub repo clean with meaningful commit history
- [ ] Both graders added as collaborators
- [ ] Tests written and passing
- [ ] Ablation study with results table
- [ ] Temporal train/test split (no data leakage)

### Quality Signals for 6.0
- [ ] Commit messages follow convention (feat/fix/data/model/docs)
- [ ] Feature branches used (not everything on main)
- [ ] SHAP explanations in the app
- [ ] CV block implemented as third block (bonus)
- [ ] RAG chatbot functional
- [ ] Ethical considerations section in documentation
- [ ] No hardcoded credentials anywhere
- [ ] All notebooks run cleanly top-to-bottom

---

## 16. RECYCLED COMPONENTS FROM SPOTIFY PROJECT

The following components from the Spotify Hit Predictor can be
adapted for this project (copy code, not git history):

| Spotify Component | Finance Equivalent | What Changes |
|---|---|---|
| Streamlit app skeleton | Same architecture | New pages, new data |
| ML pipeline (train/eval/compare) | Same logic | New features, temporal split |
| NLP preprocessing pipeline | Adapted for headlines | FinBERT replaces lyrics sentiment |
| SHAP visualization code | Reusable as-is | Different feature names |
| Model comparison framework | Reusable as-is | Different metrics (F1 vs RMSE) |
| CI/CD GitHub Actions | Reusable as-is | Same structure |
| .gitignore, .env setup | Reusable as-is | Different API keys |
| Ablation study structure | Same methodology | Different configs |

Do NOT copy: Spotify-specific data collection, lyrics scraping,
album cover CV pipeline, Spotify API auth code.

---

END OF INSTRUCTIONS.md
Any AI assistant reading this file should implement features in the priority
order defined in Section 13, follow the git workflow in Section 6,
and validate every change against the grading checklist in Section 15.
