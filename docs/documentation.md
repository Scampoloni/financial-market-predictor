# AI Applications Project Documentation Template

### Example: Reference to a notebook section
Reference to the header `## Data Preprocessing` in the notebook `analysis.ipynb`:

> See *Data Preprocessing* in
> [`analysis.ipynb`](analysis.ipynb#data-preprocessing)

### Example: Reference to Python code

Reference to a single line in `model.py`, line 42:
> [`model.py`, line 42](model.py#L42)

Reference to multiple lines in `train.py`, lines 15-38:
> [`train.py`, lines 15-38](train.py#L15-L38)

---

## Project Metadata

- **Project title:** Financial Market Predictor
- **Student:** Luis Scampoloni
- **GitHub repository URL:** https://github.com/Scampoloni/financial-market-predictor
- **Deployment URL:** https://financial-market-predictorr.streamlit.app/
- **Submission date:** 2026-06-07

### Mandatory Setup Checks

- [x] At least 2 blocks selected
- [x] Multiple and different data sources used
- [x] Deployment URL provided
- [x] Required GitHub users added to repository (`jasminh`, `bkuehnis`)

---

## Selected AI Blocks

- [x] ML Numeric Data
- [x] NLP
- [x] Computer Vision

**Primary blocks used for core solution (choose 2):**
- Primary block 1: ML Numeric Data
- Primary block 2: NLP

**Computer Vision is documented as the third block (extra work).**

---

## 1. Project Foundation (Short)

### 1.1 Problem Definition

- **Problem statement:** Predicting whether a stock will move UP or DOWN over the next 5 trading days is difficult because markets incorporate public information quickly, yet practitioners still rely on technical indicators, news sentiment, and chart patterns as decision support.
- **Goal:** Evaluate whether combining three complementary signal sources (structured market data, financial news sentiment, and candlestick chart embeddings) yields more robust out-of-sample directional predictions for 67 large-cap S&P 500 stocks than any single source alone.
- **Success criteria:** A structured ablation study (Config A → B → C) that measures the incremental F1-macro contribution of each block on a held-out 2025 test set, along with a live Streamlit application that delivers real-time predictions with interpretable evidence.

### 1.2 Integration Logic

- **How the selected blocks interact:** Each block independently produces a fixed-width feature vector for every (ticker, date) observation. The three vectors are horizontally concatenated into one feature matrix that feeds a shared LightGBM classifier. An A/B/C ablation with identical temporal splits isolates each block's marginal contribution.
- **Data and output flow between blocks:**
  ```
  Yahoo Finance OHLCV  →  [ML Block]  →  28 market features ──────────────────┐
  RSS / NewsAPI text   →  [NLP Block] →  28 sentiment/embedding features ──────┼──► combined matrix ──► LightGBM ──► UP/DOWN probability
  Candlestick PNGs     →  [CV Block]  →  10 PCA chart-embedding features ──────┘
  ```

See *Block Integration* in [`notebooks/05_integrated_model.ipynb`](notebooks/05_integrated_model.ipynb).

---

## 2. Block Documentation

### 2A. ML Numeric Data

#### 2A.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | [Yahoo Finance via yfinance](https://finance.yahoo.com) | OHLCV time series (CSV) | 69 files · ~2020–2026 · 67 tickers + ^VIX + ^GSPC | Primary feature source: price returns, volume, VIX |
| 2 | Sector classification (GICS, embedded in config) | Categorical metadata | 67 rows × 7 sectors | One-hot sector dummies added to feature matrix |

Data collection: [`src/data_collection/market_collector.py`](src/data_collection/market_collector.py)

#### 2A.2 Preprocessing and Features

- **Cleaning steps:** Date alignment across all tickers; forward-fill of missing OHLCV days (holidays); removal of tickers with fewer than 500 trading days.
- **Preprocessing steps:** Log-return calculation; volume normalisation (z-score vs. 20-day rolling mean); VIX level appended as a market-regime proxy.
- **Feature engineering and selection:** 28 features total — see [`src/features/market_features.py`](src/features/market_features.py):
  - Returns: 1-day, 5-day, 20-day
  - Momentum: RSI-14
  - Trend: MACD line, signal, histogram; SMA-20/50 ratios; EMA-12 ratio
  - Volatility: Bollinger Bands (upper/lower/width), ATR-14, 20-day realised volatility
  - Volume: ratio vs. rolling mean, VIX level
  - Cyclical: day-of-week and month encoded as sin/cos pairs
  - Sector: 7 one-hot dummies

See *Feature Engineering* in [`notebooks/02_ml_baseline.ipynb`](notebooks/02_ml_baseline.ipynb).

**Target variable and binary scope:** The v1 pipeline used a 3-class target (UP / DOWN / SIDEWAYS, where SIDEWAYS = ±1 % 5-day return). In v2, SIDEWAYS observations are excluded and the task is reduced to binary UP / DOWN classification. This halves noise and raises CV F1 from ~0.33 to ~0.49. See Iteration 1→2 in Section 2A.4. See [`src/features/market_features.py`](src/features/market_features.py) for the target construction logic.

#### EDA Key Findings

Full exploratory analysis in [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb). Summary of key findings that drove modelling decisions:

| Finding | Value | Implication for modelling |
|---------|-------|--------------------------|
| Dataset scale | 97,351 ticker-day rows; 67 tickers, 7 sectors, 2020–2025 | Sufficient for robust ML across multiple market regimes |
| Dataset start | 2020-03-13 (not 2020-01-01) | SMA-50 warm-up consumes first ~50 trading days — rows dropped intentionally |
| Missing values | 0 % across all feature columns | No imputation required before training |
| Return kurtosis | 13.05 (Gaussian = 3) | Fat tails → classification preferred over regression; tree models preferred |
| Target distribution | UP 43.1 %, DOWN 33.9 %, SIDEWAYS 23.1 % | S&P 500 upward drift; macro-F1 + class weights required |
| Feature–target correlations | All \|r\| < 0.2 | No dominant linear signal — non-linear model (LightGBM) required |
| SMA ratio inter-correlation | ~0.85–0.95 (sma\_20, sma\_50, ema\_12) | Redundant for trees; kept — LightGBM handles collinearity internally |
| High-VIX periods → UP rate | 50.1 % (vs 43.1 % base rate) | VIX captures snap-back rallies; kept as continuous feature |
| RSI < 30 → UP rate | 52.1 % (vs 43.1 % base rate) | Strong mean-reversion signal: 8–11 pp above base rate |
| Extreme moves (>\|10 %\|) | 455 rows (0.47 %) | Real market events — kept; tree models are robust to outliers via rank splits |
| Panel balance | Exactly 1,453 rows per ticker (Std = 0.0) | Perfect panel — no ticker-specific data gaps |

#### 2A.3 Model Selection

- **Models tested:** RandomForest (GridSearchCV), LightGBM (Optuna, 40 trials), StackingClassifier (RF + XGB + LGB meta-ensemble).
- **Why these models were chosen:** All three are strong on tabular data with mixed feature types. LightGBM handles large datasets efficiently and is well-suited to financial time series. Stacking tests whether complementary learner biases can be exploited.

See [`src/models/train_ml.py`](src/models/train_ml.py) for training logic and [`src/config.py`](src/config.py) for hyperparameter grids.

#### 2A.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Establish baseline with raw OHLCV | 3-class target (UP / DOWN / SIDEWAYS where \|5-day return\| ≤ 1 %), next-day horizon | RandomForest | CV F1-macro ≈ 0.33 | — |
| 2 | Improve signal-to-noise | **SIDEWAYS rows dropped** (±1 % threshold produced ~23 % neutral rows with little predictive structure); switch to binary 5-day UP/DOWN; add technical indicators (RSI, MACD, Bollinger) | RandomForest, XGBoost | CV F1-macro ≈ 0.49 | +0.16 |
| 3 | Systematic hyperparameter optimisation | Optuna tuning for LightGBM; TimeSeriesSplit CV; Stacking ensemble | RF, LightGBM, Stacking | Test F1-macro = 0.4970 (LightGBM best) | +0.007 vs iteration 2 |

See *Model Comparison* in [`notebooks/06_evaluation_ablation.ipynb`](notebooks/06_evaluation_ablation.ipynb).

#### 2A.5 Evaluation and Error Analysis

- **Metrics used:** Macro F1, accuracy, per-class precision/recall/F1 (UP vs DOWN).
- **Final results (held-out 2025 test set, Config A):**

| Metric | Value |
|--------|-------|
| Test F1-macro | 0.4970 |
| Test accuracy | 0.4971 |
| DOWN F1 | 0.4876 |
| UP F1 | 0.5063 |

- **Error patterns and likely causes:** Near-random performance (~0.50) is consistent with the semi-strong form of the Efficient Market Hypothesis — public technical signals are quickly arbitraged. The model slightly favours UP predictions, reflecting a long-term upward drift in the S&P 500 universe.

Ablation results stored in [`data/processed/ablation_results.json`](data/processed/ablation_results.json).

#### 2A.6 Integration with Other Block(s)

- **Inputs received from other block(s):** None — this block operates solely on market data.
- **Outputs provided to other block(s):** 28-feature vector per (ticker, date) row, persisted in [`data/processed/features_market.parquet`](data/processed/features_market.parquet), joined with NLP and CV features for Configs B and C.

---

### 2B. NLP (If selected)

#### 2B.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | RSS financial news feeds (Yahoo Finance, Reuters, CNBC, Seeking Alpha) | Unstructured text (headlines) | ~6,200 headlines across 67 tickers | Primary sentiment signal |
| 2 | [NewsAPI](https://newsapi.org) | Unstructured text (headlines + snippets) | ~2,350 additional headlines | Supplementary coverage for low-news tickers |
| 3 | [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) (HuggingFace) | Pre-trained transformer model | ~440 MB | Sentiment scoring model |
| 4 | [Yahoo Finance via yfinance](https://finance.yahoo.com) — `ticker.upgrades_downgrades` + `ticker.recommendations` | Analyst rating time series (structured) | 67 tickers × ~1,453 dates; historical firm-level upgrades/downgrades + monthly consensus counts | 5 analyst features: `analyst_consensus`, `analyst_upgrade_score`, `analyst_coverage_count`, `price_target_upside`, `analyst_sentiment_momentum` |

News collection: [`src/data_collection/news_scraper.py`](src/data_collection/news_scraper.py).  
Analyst feature builder: [`src/data_collection/build_analyst_features.py`](src/data_collection/build_analyst_features.py).  
Total corpus: 8,552 headline-rows stored in `data/raw/news/`.

#### 2B.2 Preprocessing and Prompt Design

- **Text preprocessing:** Lower-case normalisation; removal of boilerplate ticker mentions; deduplication by headline hash; 512-token truncation for FinBERT. See [`src/nlp/finbert_sentiment.py`](src/nlp/finbert_sentiment.py).
- **Prompt design or retrieval setup:** No generative prompting for the sentiment pipeline. For the RAG chatbot ([`src/nlp/rag_chatbot.py`](src/nlp/rag_chatbot.py)): headlines are chunked and embedded with `sentence-transformers/all-MiniLM-L6-v2`; top-5 retrieved chunks are prepended to a Gemini API call. Coverage fallback hierarchy: ticker-level → sector-average → market-average → forward-fill.

  **NLP sentiment coverage by fallback tier** (out of 97,351 total ticker-day rows):

  | Tier | Source | Approx. ticker-day coverage |
  |------|--------|:---------------------------:|
  | 1 — Direct ticker news | Ticker-specific RSS/NewsAPI headlines | ~1.7 % |
  | 2 — Sector fallback | Mean FinBERT/VADER score across same-sector tickers on that day | ~35 % |
  | 3 — Market fallback | Mean score across all tickers on that day (used when no sector news) | ~22 % |
  | 4 — Forward-fill | Last non-null sentiment value carried forward | ~41 % |
  | **Total with signal** | Tiers 1–4 combined | **~100 %** |

  Rows that rely on tier 2–4 are flagged by `is_sentiment_imputed = 1` in the NLP feature matrix. The net effect: Config B achieves complete row coverage but with a weak, aggregated signal for the majority of observations — the primary reason NLP adds noise rather than predictive lift (−0.0143 F1 vs Config A).

- **Analyst data (5 additional features):** `analyst_consensus`, `analyst_coverage_count`, `analyst_sentiment_momentum`, `analyst_upgrade_score`, `price_target_upside` — structured signals derived from analyst rating data, joined to the NLP feature matrix. Together with the 23 text-derived features this yields 28 NLP-block features total.
- **PCA note:** FinBERT embedding PCA (10 components) is fitted on training-period rows only (date ≤ 2024-06-30); val/test rows are transformed using the saved scaler/PCA without re-fitting ([`src/features/nlp_features.py`](src/features/nlp_features.py)). This eliminates any temporal leakage from test-period embedding distributions.

#### 2B.3 Approach Selection

- **Approach used:** Dual-model sentiment scoring (FinBERT transformer + VADER lexicon) combined with PCA-compressed FinBERT embeddings; RAG chatbot as supplementary NLP feature.
- **Alternatives considered:** Classical TF-IDF + logistic regression (rejected: no contextual understanding); GPT-4 scoring (rejected: cost and rate limits at 8,552 headlines); single-model VADER-only (rejected: misses domain-specific financial language).

See *Approach Selection* in [`notebooks/03_nlp_pipeline.ipynb`](notebooks/03_nlp_pipeline.ipynb#approach-selection).

#### 2B.4 Comparison and Iterations

| Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Validate sentiment models on financial text | Curated 50-headline benchmark with human labels | VADER only | Direction accuracy = 0.800, Macro F1 = 0.796 | — |
| 2 | Compare transformer vs lexicon | Add FinBERT | FinBERT vs VADER | FinBERT acc = 0.792, F1 = 0.791; 0 abstentions vs 9 for VADER | FinBERT more decisive; VADER marginally higher raw accuracy |
| 3 | Build full feature set | Add rolling windows, momentum, dispersion, surprise z-score, sentiment×volume interaction, 10-dim PCA embeddings | Both models combined | Config B test F1 = 0.4826 vs Config A = 0.4970 (−0.0143) | NLP adds coverage but overlaps with priced-in information |

Benchmark results in [`notebooks/03_nlp_pipeline.ipynb`](notebooks/03_nlp_pipeline.ipynb); integration impact in [`notebooks/06_evaluation_ablation.ipynb`](notebooks/06_evaluation_ablation.ipynb).

#### 2B.5 Evaluation and Error Analysis

- **Evaluation strategy:** (a) Intrinsic: 50-headline curated benchmark with human sentiment labels. (b) Extrinsic: ablation comparison Config A vs Config B on the 2025 held-out test set.
- **Results:**

| Metric | FinBERT | VADER |
|--------|:-------:|:-----:|
| Direction accuracy | 0.792 | 0.800 |
| Macro F1 | 0.791 | 0.796 |
| Score–direction correlation | 0.694 | 0.656 |
| Abstentions (NEUTRAL) | 0 / 50 | 9 / 50 |
| Inter-model score correlation | 0.47 | — |

Config B (Market + NLP) test F1-macro = 0.4826 (−0.0143 vs Config A baseline).

- **Error patterns and likely causes:** The negative delta indicates that news sentiment is largely already priced into the 5-day return window. Raw news coverage is sparse (1.7 % of ticker-days have direct news), so most rows use sector/market-level fallback sentiment — a weak signal. Headlines also reflect events, not future price moves, limiting predictive utility at the 5-day horizon.

#### 2B.6 Integration with Other Block(s)

- **Inputs received from other block(s):** Ticker symbol and date index from the ML block (used to align sentiment features temporally).
- **Outputs provided to other block(s):** 28-feature NLP vector per (ticker, date) — 23 sentiment/embedding features plus 5 analyst-data features (`analyst_consensus`, `analyst_coverage_count`, `analyst_sentiment_momentum`, `analyst_upgrade_score`, `price_target_upside`) — persisted in [`data/processed/features_nlp.parquet`](data/processed/features_nlp.parquet), concatenated with market features to produce the Config B/C feature matrix.

---

### 2C. Computer Vision

#### 2C.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | Generated candlestick charts (mplfinance, from Yahoo Finance OHLCV) | PNG images (30-day rolling windows, bi-daily step) | 61,640+ images @ 224×224 px | Input to EfficientNet-B0 feature extractor |
| 2 | [EfficientNet-B0](https://pytorch.org/vision/stable/models/efficientnet.html) (torchvision, domain-fine-tuned) | Pre-trained CNN backbone | ~16.3 MB fine-tuned weights | Visual feature extraction model |

Chart generation: [`src/data_collection/chart_generator.py`](src/data_collection/chart_generator.py).  
Fine-tuning script: [`scripts/finetune_cnn.py`](scripts/finetune_cnn.py).

#### 2C.2 Preprocessing and Augmentation

- **Image preprocessing:** 30-day OHLCV window rendered as a dark-background candlestick PNG (224×224 px) using mplfinance. Images are normalised with ImageNet mean/std before EfficientNet inference. See [`src/cv/chart_classifier.py`](src/cv/chart_classifier.py).
- **Augmentation strategy:** No data augmentation during inference. During CNN fine-tuning (`scripts/finetune_cnn.py`): random horizontal flip, colour jitter (brightness/contrast ±0.2), random rotation ±5°. Augmentation is conservative to preserve chart semantics.
- **PCA note:** EfficientNet embedding PCA (10 components) is fitted on training-period rows only (date ≤ 2024-06-30); val/test rows are transformed using the saved scaler/PCA without re-fitting ([`src/features/cv_features.py`](src/features/cv_features.py)). This eliminates any temporal leakage from test-period embedding distributions.

#### 2C.3 Model Selection

- **Vision model(s) used:** EfficientNet-B0 (ImageNet pre-trained, then domain-fine-tuned on chart→UP/DOWN labels). The 1,280-dim penultimate-layer embedding is PCA-compressed to 10 components.
- **Why these model(s) were chosen:** EfficientNet-B0 offers a strong accuracy/parameter trade-off (~5.3 M parameters), fits in Streamlit Cloud RAM limits, and generalises well from ImageNet to chart images without full retraining. A heavier model (ResNet-50) was considered but exceeded deployment memory constraints.

#### 2C.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Model(s) used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Baseline chart embeddings | Monthly chart step (2,788 images), frozen ImageNet weights, 50-dim PCA | EfficientNet-B0 (frozen) | CV coverage ~1.7 % | — |
| 2 | Increase coverage | Switch to bi-daily step (61,640 images); reduce PCA to 10 dims | EfficientNet-B0 (frozen) | CV coverage ~59 %; Config C test F1 = 0.4861 (+0.0035 vs Config B) | +0.0035 F1 vs iteration 1 |
| 3 | Domain adaptation | Fine-tune final two EfficientNet blocks on chart→direction labels (10 epochs) | EfficientNet-B0 (fine-tuned) | Improved embedding separability (qualitative); weights in `models/cnn_finetuned.pth` | Embedding clusters better aligned with UP/DOWN |

#### 2C.5 Evaluation and Error Analysis

**Intrinsic (fine-tuning validation):**

`scripts/finetune_cnn.py` uses a stratified 85 / 15 train–val split and reports per-epoch validation accuracy and macro F1. The best checkpoint (by val F1) is written to `models/cnn_finetuned.pth` with its metric persisted inside the file.

| Split | Samples | Best val F1-macro | Epochs |
|-------|---------|:-----------------:|:------:|
| Train | 36,960 | — | 10 |
| Val | ~6,522 | **0.538** | best at checkpoint |

Fine-tuning training strategy: head-only for epochs 1–3 (lr = 1e-3), then top-2 EfficientNet blocks unfrozen for epochs 4–10 (backbone lr = 3e-5, head lr = 1e-3). Class weights applied to CrossEntropyLoss to handle UP/DOWN imbalance.

Qualitative PCA scatter plots of embeddings (frozen vs fine-tuned) are available in [`notebooks/04_cv_pipeline.ipynb`](notebooks/04_cv_pipeline.ipynb) — clusters show improved UP/DOWN separability after domain adaptation.

**Extrinsic (ablation on held-out test set):**

| Config | Features | Test F1-macro | Δ vs Config B |
|--------|----------|---------------|---------------|
| B | Market + NLP (56 features) | 0.4826 | — |
| C | Market + NLP + CV (66 features) | 0.4861 | +0.0035 |

Bootstrap 95 % CI for Config C: [0.487, 0.502] (N = 2,000 resamples). Overlapping CIs across configs indicate the marginal improvement is not statistically significant — CV features provide complementary signal but do not dominate.

- **Metrics and/or visual checks:** Extrinsic ablation above; qualitative: PCA embedding scatter plots in [`notebooks/04_cv_pipeline.ipynb`](notebooks/04_cv_pipeline.ipynb); additional LightGBM-on-CV-only baseline (F1 ≈ 0.35, below random baseline) confirms CV embeddings are not sufficient alone.
- **Final results:** Config C test F1-macro = 0.4861 (+0.0035 vs Config B without CV).
- **Error patterns and limitations:** CV embeddings overlap strongly with existing technical indicators (RSI, MACD, Bollinger Bands already capture most visual candlestick information algebraically). Survivorship bias in the ticker universe (all currently-listed S&P 500 stocks) inflates historical win rates for UP predictions.

#### 2C.6 Integration with Other Block(s)

- **Inputs received from other block(s):** Ticker and date index from the ML block (used to select the matching chart image from `data/raw/charts/`).
- **Outputs provided to other block(s):** 10-dim PCA embedding vector per (ticker, date), persisted in [`data/processed/features_cv.parquet`](data/processed/features_cv.parquet), concatenated with market and NLP features to produce the Config C feature matrix.

---

## 3. Deployment

- **Deployment URL:** https://financial-market-predictorr.streamlit.app/
- **Main user flow:**
  1. User selects a ticker and date range on the **Prediction** page.
  2. App loads pre-computed artifacts — `models/stacking_final.pkl` (contains the best-performing model, LightGBM in all configs, plus its feature column list; the filename is kept for backwards compatibility), `data/processed/features_market.parquet`, `data/processed/features_nlp.parquet`, and `data/processed/features_cv.parquet` — and returns a directional UP/DOWN probability with a Plotly candlestick chart. Live inference assembles features on-the-fly using `src/models/predict.py`; no `features_combined.parquet` is required.
  3. User can explore per-block evidence on the **Analysis** page (SHAP feature importance, ablation bar chart).
  4. User can query the **News Chat** tab (RAG chatbot) for contextual news evidence behind any prediction.
- **Screenshot or short demo:** See [`docs/screenshots/01_prediction_flow.png`](docs/screenshots/01_prediction_flow.png), [`docs/screenshots/02_model_analysis.png`](docs/screenshots/02_model_analysis.png), [`docs/screenshots/03_nlp_cv_integration.png`](docs/screenshots/03_nlp_cv_integration.png).

App entry point: [`app.py`](app.py). Page modules: [`src/app/pages/`](src/app/pages/).

---

## 4. Execution Instructions

- **Environment setup:**
  ```bash
  python -m venv .venv && source .venv/bin/activate   # Linux/Mac
  # or: .venv\Scripts\activate                         # Windows
  pip install -r requirements.txt
  cp .env.example .env
  # Edit .env: add NEWS_API_KEY and (optional) GEMINI_API_KEY
  ```

- **Data setup (full pipeline, several hours):**
  ```bash
  python -m src.data_collection.market_collector   # Download OHLCV from Yahoo Finance
  python -m src.data_collection.news_scraper       # Scrape RSS + NewsAPI headlines
  python -m src.features.market_features           # Build 28 market features
  python -m src.features.nlp_features              # Build 28 NLP features (FinBERT + VADER + analyst data)
  python -m src.data_collection.chart_generator --step 2   # Generate 61k candlestick PNGs
  python -m src.features.cv_features               # Extract EfficientNet embeddings + PCA
  ```

- **Training command(s):**
  ```bash
  python -m src.models.train_ml          # Config A/B/C ablation (LightGBM + RF + Stacking)
  python scripts/finetune_cnn.py --epochs 10   # Optional: fine-tune EfficientNet-B0
  ```

- **Inference/run command(s):**
  ```bash
  streamlit run app.py
  ```

- **Smoke test (5–10 minutes, no API keys required):**
  ```bash
  python scripts/build_smoke_dataset.py
  python -m src.features.market_features --test
  python -m src.features.nlp_features --test
  python -m src.data_collection.chart_generator --test --step 2
  python -m src.features.cv_features --test
  python -m src.models.train_ml --config C
  pytest tests/ -q   # 8 tests
  ```

- **Reproducibility notes:** Python 3.11+. All random seeds fixed via `src/config.py`. Pre-computed artifacts are committed to the repository (`models/`, `data/processed/`) so the app can be launched without re-running the full pipeline.

---

## 5. Optional Bonus Evidence

- [x] Third selected block implemented with strong quality — Computer Vision (EfficientNet-B0, domain fine-tuning, bi-daily chart generation, PCA compression, full ablation measurement).
- [x] More than two data sources used with clear added value — Yahoo Finance OHLCV, RSS feeds, NewsAPI, FinBERT (HuggingFace), EfficientNet-B0 (torchvision) with fine-tuning on domain data.
- [x] Extended evaluation — Bootstrap 95 % CI (N = 2,000), 5-fold TimeSeriesSplit, per-class precision/recall/F1, multi-horizon comparison (5-day vs 21-day), sector-level and regime-level sensitivity analysis. See [`notebooks/06_evaluation_ablation.ipynb`](notebooks/06_evaluation_ablation.ipynb).
- [x] Ethics, bias, or fairness analysis — Documented in `README.md`: survivorship bias (currently-listed S&P 500 only), English-language news concentration, market access inequality, EMH interpretation of ~0.50 F1. System carries an explicit "not financial advice" disclaimer in the app.

Evidence for selected bonus items: full ablation results in [`data/processed/ablation_results.json`](data/processed/ablation_results.json); evaluation visualisations in [`data/processed/ablation_f1_bar.png`](data/processed/ablation_f1_bar.png), [`data/processed/per_class_performance.png`](data/processed/per_class_performance.png), [`data/processed/feature_importance.png`](data/processed/feature_importance.png).
