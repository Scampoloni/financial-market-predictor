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
- **Success criteria:** A structured ablation study (Config A → B → C → D) that measures the incremental F1-macro contribution of each feature block on a held-out 2025 test set, along with a live Streamlit application that delivers real-time predictions with interpretable evidence.

### 1.2 Integration Logic

- **How the selected blocks interact:** Each block independently produces a fixed-width feature vector for every (ticker, date) observation. The vectors are horizontally concatenated into one feature matrix that feeds a shared LightGBM classifier. An A/B/C/D ablation with identical temporal splits isolates each block's marginal contribution.
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

Data collection: [`src/data_collection/market_collector.py`](https://github.com/Scampoloni/financial-market-predictor/blob/main/src/data_collection/market_collector.py)

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

**Target variable and binary scope:** The v1 pipeline used a 3-class target (UP / DOWN / SIDEWAYS, where SIDEWAYS = ±1 % 5-day return). In v2, the SIDEWAYS class is eliminated: all observations are reclassified as binary UP (5-day return > 0 %) / DOWN (return ≤ 0 %). No rows are dropped — the ±1 % zone (~23 % of data) is redistributed into UP/DOWN by sign rather than filtered out. This simplification raises CV F1 from ~0.33 to ~0.49. See Iteration 1→2 in Section 2A.4. See [`src/features/market_features.py` L273–294](https://github.com/Scampoloni/financial-market-predictor/blob/main/src/features/market_features.py#L273-L294) for the target construction logic.

#### EDA Key Findings

Full exploratory analysis in [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb). Summary of key findings that drove modelling decisions:

| Finding | Value | Implication for modelling |
|---------|-------|--------------------------|
| Dataset scale | 97,351 ticker-day rows; 67 tickers, 7 sectors, 2020–2025 | Sufficient for robust ML across multiple market regimes |
| Dataset start | 2020-03-13 (not 2020-01-01) | SMA-50 warm-up consumes first ~50 trading days — rows dropped intentionally |
| Missing values | 0 % across all feature columns | No imputation required before training |
| Return kurtosis | 13.05 (Gaussian = 3) | Fat tails → classification preferred over regression; tree models preferred |
| Target distribution (v1 3-class) | UP 43.1 %, DOWN 33.9 %, SIDEWAYS 23.0 % | S&P 500 upward drift; v2 binary: UP ≈ 56 %, DOWN ≈ 44 % after SIDEWAYS redistribution; macro-F1 + class weights used |
| Feature–target correlations | All \|r\| < 0.2 | No dominant linear signal — non-linear model (LightGBM) required |
| SMA ratio inter-correlation | ~0.85–0.95 (sma\_20, sma\_50, ema\_12) | Redundant for trees; kept — LightGBM handles collinearity internally |
| High-VIX periods → UP rate | 50.1 % (vs 43.1 % v1 base rate) | VIX captures snap-back rallies in v1 data; v2 binary base rate ≈ 56 %, so VIX effect is smaller but direction preserved — kept as continuous feature |
| RSI < 30 → UP rate | 52.1 % (vs 43.1 % v1 base rate) | Mean-reversion signal in v1 data; in v2 binary (base rate ≈ 56 %), RSI < 30 → 52.1 % is modestly below base rate — mean-reversion remains useful as a relative feature but oversold stocks trend DOWN less often than the market average |
| Extreme moves (>\|10 %\|) | 455 rows (0.47 %) | Real market events — kept; tree models are robust to outliers via rank splits |
| Panel balance | Exactly 1,453 rows per ticker (Std = 0.0) | Perfect panel — no ticker-specific data gaps |

#### 2A.3 Model Selection

- **Models tested:** RandomForest (fixed hyperparameters, cross-validated via 5-fold TimeSeriesSplit), LightGBM (Optuna, 40 trials), StackingClassifier (RF + XGB + LGB meta-ensemble).
- **Why these models were chosen:** All three are strong on tabular data with mixed feature types. LightGBM handles large datasets efficiently and is well-suited to financial time series. Stacking tests whether complementary learner biases can be exploited.

See [`src/models/train_ml.py`](https://github.com/Scampoloni/financial-market-predictor/blob/main/src/models/train_ml.py) (RF: [`train_random_forest` L164](https://github.com/Scampoloni/financial-market-predictor/blob/main/src/models/train_ml.py#L164); LightGBM + Optuna: [`_optuna_lgb` L194](https://github.com/Scampoloni/financial-market-predictor/blob/main/src/models/train_ml.py#L194); Stacking: [`train_stacking` L256](https://github.com/Scampoloni/financial-market-predictor/blob/main/src/models/train_ml.py#L256); ablation runner: [`run_ablation` L377](https://github.com/Scampoloni/financial-market-predictor/blob/main/src/models/train_ml.py#L377)) for training logic and [`src/config.py`](https://github.com/Scampoloni/financial-market-predictor/blob/main/src/config.py) for hyperparameter grids.

#### 2A.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Establish baseline with raw OHLCV | 3-class target (UP / DOWN / SIDEWAYS where \|5-day return\| ≤ 1 %), next-day horizon | RandomForest | CV F1-macro ≈ 0.33 | — |
| 2 | Improve signal-to-noise | SIDEWAYS class eliminated: all rows reclassified as binary UP (return > 0 %) / DOWN (return ≤ 0 %); ±1 % zone (~23 % of data) redistributed by sign rather than dropped; switch to 5-day horizon; add technical indicators (RSI, MACD, Bollinger) | RandomForest, XGBoost | CV F1-macro ≈ 0.49 | +0.16 |
| 3 | Systematic hyperparameter optimisation | Optuna tuning for LightGBM; TimeSeriesSplit CV; Stacking ensemble | RF, LightGBM, Stacking | Test F1-macro = 0.4970 (LightGBM best) | +0.007 vs iteration 2 |

See *Model Comparison* in [`notebooks/06_evaluation_ablation.ipynb`](notebooks/06_evaluation_ablation.ipynb).

#### 2A.5 Evaluation and Error Analysis

- **Metrics used:** Macro F1, accuracy, per-class precision/recall/F1 (UP vs DOWN). Bootstrap 95 % CI (N = 2,000 resamples) for Config C/D to assess statistical significance of ablation deltas.
- **Final results (held-out 2025 test set — all four ablation configs):**

| Config | Features | Test F1-macro | Test Accuracy | DOWN F1 | UP F1 |
|--------|----------|:-------------:|:-------------:|:-------:|:-----:|
| **A** — Market only | 28 | **0.4970** | 0.4971 | 0.4876 | 0.5063 |
| **B** — + NLP | 56 | 0.4826 | 0.4842 | 0.5113 | 0.4539 |
| **C** — + NLP + CV | 66 | 0.4861 | 0.4863 | 0.4978 | 0.4744 |
| **D** — + Analyst ✓ (corrected) | 66 | 0.4850 | 0.4850 | 0.4897 | 0.4803 |

> Config D re-trains Config C with corrected analyst data (a `NameError` in `build_analyst_features.py` caused all analyst features to be silently zero in earlier runs). The D vs C delta of −0.0011 confirms analyst ratings are effectively priced in at the 5-day horizon.

**Per-model breakdown (val F1 → test F1, Config A as representative):**

| Model | Val F1 | Test F1 | Notes |
|-------|:------:|:-------:|-------|
| LightGBM (Optuna) | 0.4771 | **0.4970** | ✓ Selected — best val F1 in all four configs |
| RandomForest (fixed params) | 0.4402 | 0.4928 | Competitive test F1 but lower val F1 |
| Stacking (RF + XGB + LGB) | 0.4207 | 0.3796 | Dramatically underperforms — KFold leakage in stacking |

The stacking ensemble uses `cv=5` (KFold) internally to generate cross-val meta-features because `TimeSeriesSplit` is incompatible with `StackingClassifier`. In a temporal financial dataset, KFold allows future data to inform base learner training, creating an optimistic val score that does not generalise — test F1 drops to 0.38–0.42 across all configs. LightGBM (Optuna-tuned, TimeSeriesSplit CV) wins all four configs and is the deployed model.

- **Per-class shift analysis:** Adding NLP features (Config A → B) increases DOWN recall from 0.520 to 0.587 but reduces UP recall from 0.478 to 0.397. The net macro-F1 falls. This reflects a systematic bearish bias introduced by the NLP block: most sentiment-imputed rows carry sector/market average scores rather than ticker-specific signals, and averaging across a cross-section that includes negative-sentiment stocks makes the model predict DOWN more often. Adding CV features (Config B → C) partially corrects the imbalance (DOWN recall 0.553, UP recall 0.429) but does not recover the Config A baseline. Adding corrected analyst features (Config C → D) produces only a marginal shift (DOWN recall 0.537, UP recall 0.441), confirming that analyst ratings contribute negligible directional signal at the 5-day horizon.

- **Interpretation of negative ablation deltas:** Three structural causes explain why NLP and CV do not improve the headline F1 beyond Config A:
  1. **Imputation dilutes signal.** Only ~1.7 % of ticker-days have direct ticker-specific news; 98.3 % use sector/market/forward-fill fallbacks. Imputed sentiment is a cross-sectional average, not an idiosyncratic signal, so it adds correlation structure without predictive content for individual tickers.
  2. **Efficient Market Hypothesis.** Public information (news sentiment, visible chart patterns) is largely priced in within minutes of release. At a 5-day horizon, the price response has already completed, leaving no exploitable residual signal.
  3. **Feature overlap with technical indicators.** Chart embeddings capture the same information encoded in RSI, MACD, and Bollinger Bands. The PCA-compressed CV features are therefore nearly redundant with the existing market block, increasing model complexity without adding independent signal.

- **Statistical significance:** Bootstrap 95 % CI for Config C F1 is [0.487, 0.502]. All configs' CIs overlap, confirming that the observed deltas (−0.0144 and −0.0109) are **not statistically significant** — they are consistent with sampling noise, not an indication that multi-modal fusion actively hurts predictive power.

- **Cross-validation stability:** 5-fold TimeSeriesSplit CV F1 standard deviations: Config A 0.016, Config B 0.027, Config C 0.018, Config D 0.020. All are well below 0.1, confirming consistent performance across market regimes (train period 2020–2024).

Ablation results stored in [`data/processed/ablation_results.json`](data/processed/ablation_results.json). Full per-fold analysis and bootstrap CI visualisations in [`notebooks/06_evaluation_ablation.ipynb`](notebooks/06_evaluation_ablation.ipynb).

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
Total corpus: ~8,550 headline-rows stored in `data/raw/news/`.

#### 2B.2 Preprocessing and Prompt Design

- **Text preprocessing:** Lower-case normalisation; removal of boilerplate ticker mentions; deduplication by headline hash; 512-token truncation for FinBERT. See [`src/nlp/finbert_sentiment.py`](src/nlp/finbert_sentiment.py).
- **Prompt design or retrieval setup:** No generative prompting for the sentiment pipeline. For the RAG chatbot ([`src/nlp/rag_chatbot.py`](src/nlp/rag_chatbot.py)): headlines are chunked and embedded with `sentence-transformers/all-MiniLM-L6-v2`; top-5 retrieved chunks are prepended to a Claude API call. Coverage fallback hierarchy: ticker-level → sector-average → market-average → forward-fill.

  **NLP sentiment coverage by fallback tier** (out of 97,351 total ticker-day rows):

  | Tier | Source | Approx. ticker-day coverage |
  |------|--------|:---------------------------:|
  | 1 — Direct ticker news | Ticker-specific RSS/NewsAPI headlines | ~1.7 % |
  | 2 — Sector fallback | Mean FinBERT/VADER score across same-sector tickers on that day | ~35 % |
  | 3 — Market fallback | Mean score across all tickers on that day (used when no sector news) | ~22 % |
  | 4 — Forward-fill | Last non-null sentiment value carried forward | ~41 % |
  | **Total with signal** | Tiers 1–4 combined | **~100 %** |

  Rows that rely on tier 2–4 are flagged by `is_sentiment_imputed = 1` in the NLP feature matrix. The net effect: Config B achieves complete row coverage but with a weak, aggregated signal for the majority of observations — the primary reason NLP adds noise rather than predictive lift (−0.0143 F1 vs Config A).

- **Analyst data (5 additional features):** `analyst_consensus`, `analyst_coverage_count`, `analyst_sentiment_momentum`, `analyst_upgrade_score`, `price_target_upside` — structured signals derived from analyst rating data, persisted separately in [`data/processed/features_analyst.parquet`](data/processed/features_analyst.parquet) and joined to the NLP feature matrix at training time. Together with the 23 text-derived features this yields 28 NLP-block features total.
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
- **Outputs provided to other block(s):** 28-feature NLP vector per (ticker, date) — 23 sentiment/embedding features persisted in [`data/processed/features_nlp.parquet`](data/processed/features_nlp.parquet), plus 5 analyst-data features (`analyst_consensus`, `analyst_coverage_count`, `analyst_sentiment_momentum`, `analyst_upgrade_score`, `price_target_upside`) persisted separately in [`data/processed/features_analyst.parquet`](data/processed/features_analyst.parquet). Both files are joined during training to produce the complete 28-feature NLP input for Configs B, C, and D.

---

### 2C. Computer Vision

#### 2C.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | Generated candlestick charts (mplfinance, from Yahoo Finance OHLCV) | PNG images (30-day rolling windows, bi-daily step) | 61,640+ images @ 224×224 px | Input to EfficientNet-B0 feature extractor |
| 2 | [EfficientNet-B0](https://pytorch.org/vision/stable/models/efficientnet.html) (torchvision, domain-fine-tuned) | Pre-trained CNN backbone | ~16.3 MB fine-tuned weights | Visual feature extraction model |

Chart generation: [`src/data_collection/chart_generator.py`](https://github.com/Scampoloni/financial-market-predictor/blob/main/src/data_collection/chart_generator.py).
Fine-tuning script: [`scripts/finetune_cnn.py`](https://github.com/Scampoloni/financial-market-predictor/blob/main/scripts/finetune_cnn.py).

#### 2C.2 Preprocessing and Augmentation

- **Image preprocessing:** 30-day OHLCV window rendered as a dark-background candlestick PNG (224×224 px) using mplfinance. Images are normalised with ImageNet mean/std before EfficientNet inference. See [`src/cv/chart_classifier.py`](https://github.com/Scampoloni/financial-market-predictor/blob/main/src/cv/chart_classifier.py).
- **Augmentation strategy:** No data augmentation during inference. During CNN fine-tuning (`scripts/finetune_cnn.py`): random horizontal flip, colour jitter (brightness/contrast ±0.2), random rotation ±5°. Augmentation is conservative to preserve chart semantics.
- **PCA note:** EfficientNet embedding PCA (10 components) is fitted on training-period rows only (date ≤ 2024-06-30); val/test rows are transformed using the saved scaler/PCA without re-fitting ([`src/features/cv_features.py`](https://github.com/Scampoloni/financial-market-predictor/blob/main/src/features/cv_features.py)). This eliminates any temporal leakage from test-period embedding distributions.

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

[`scripts/finetune_cnn.py`](https://github.com/Scampoloni/financial-market-predictor/blob/main/scripts/finetune_cnn.py) uses a **temporal train/val split** that mirrors the ablation protocol: charts with date ≤ 2023-12-31 go to train; charts from 2024-01-01 to 2024-06-30 go to validation. This prevents any future-data leakage into CNN training. Per-epoch validation accuracy and macro F1 are reported; the best checkpoint (by val F1) is written to `models/cnn_finetuned.pth` with its metric persisted inside the file.

| Split | Period | Best val F1-macro | Epochs |
|-------|--------|:-----------------:|:------:|
| Train | ≤ 2023-12-31 | — | 10 |
| Val | 2024-01-01–2024-06-30 | **0.538** | best at checkpoint |

**Held-out evaluation (2025 charts):**

Because `val F1 = 0.538` was the criterion used to *select* the best checkpoint, it cannot also serve as an independent estimate of generalisation — a classic selection-on-validation problem. To obtain a clean, unbiased performance number, the fine-tuned classifier was re-evaluated on all chart images dated **2025-01-01 or later** — a window entirely outside both the training period (≤ 2023-12-31) and the validation window (2024-H1) used during fine-tuning.

[`scripts/eval_cv_held_out.py`](https://github.com/Scampoloni/financial-market-predictor/blob/main/scripts/eval_cv_held_out.py) loads the classifier head intact (not stripped), scans `data/raw/charts/` for matching 2025 PNGs, and runs batch inference. Results on **n = 11 993** charts:

| Metric | Value |
|--------|------:|
| Accuracy | 0.5147 |
| F1-macro | **0.5057** |
| F1 DOWN | 0.4390 |
| F1 UP | 0.5724 |

Confusion matrix (rows = actual, cols = predicted):

|  | Pred DOWN | Pred UP |
|--|----------:|--------:|
| **Actual DOWN** | 2 277 | 3 143 |
| **Actual UP** | 2 677 | 3 896 |

The model shows a systematic **UP-prediction bias** (recall UP = 0.59 vs DOWN = 0.42), consistent with the class imbalance in the training data (UP ≈ 55 %). F1-macro = 0.506 lies only marginally above the random baseline (0.50), confirming that chart pattern alone has limited predictive power — in line with the extrinsic ablation results above. Full metrics are persisted in [`data/processed/cv_held_out_eval.json`](data/processed/cv_held_out_eval.json).

Fine-tuning training strategy: head-only for epochs 1–3 (lr = 1e-3), then top-2 EfficientNet blocks unfrozen for epochs 4–10 (backbone lr = 3e-5, head lr = 1e-3). Class weights applied to CrossEntropyLoss to handle UP/DOWN imbalance.

Qualitative PCA scatter plots of embeddings (frozen vs fine-tuned) are available in [`notebooks/04_cv_pipeline.ipynb`](notebooks/04_cv_pipeline.ipynb) — clusters show improved UP/DOWN separability after domain adaptation.

**Extrinsic (ablation on held-out test set):**

| Config | Features | Test F1-macro | Δ vs Config B |
|--------|----------|---------------|---------------|
| B | Market + NLP (56 features) | 0.4826 | — |
| C | Market + NLP + CV (66 features) | 0.4861 | +0.0035 |
| D | Market + NLP + CV + Analyst ✓ (66 features) | 0.4850 | +0.0024 |

Bootstrap 95 % CI for Config C: [0.487, 0.502] (N = 2,000 resamples). Overlapping CIs across configs indicate the marginal improvement is not statistically significant — CV features provide complementary signal but do not dominate. Config D (corrected analyst data) is within the CI of Config C, confirming the analyst correction has no measurable impact on the CV block's contribution.

- **Metrics and/or visual checks:** Extrinsic ablation above; qualitative: PCA embedding scatter plots in [`notebooks/04_cv_pipeline.ipynb`](notebooks/04_cv_pipeline.ipynb); additional LightGBM-on-CV-only baseline (F1 ≈ 0.35, below random baseline) confirms CV embeddings are not sufficient alone.
- **Final results:** Config C test F1-macro = 0.4861 (+0.0035 vs Config B without CV). Config D = 0.4850 (corrected analyst pipeline, Δ = −0.0011 vs C).
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
  # For byte-exact reproducibility use pinned versions instead:
  # pip install -r requirements-pinned.txt
  cp .env.example .env
  # Windows PowerShell: Copy-Item .env.example .env
  # Edit .env: add NEWS_API_KEY and (optional) CLAUDE_API_KEY
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
  python -m src.models.train_ml          # Config A/B/C/D ablation (LightGBM + RF + Stacking)
  python scripts/finetune_cnn.py --epochs 10   # Optional: fine-tune EfficientNet-B0
  ```

- **Inference/run command(s):**
  ```bash
  streamlit run app.py
  ```

- **Smoke test (5–10 minutes, no API keys required):**
  ```bash
  # Only needed if data/smoke is missing locally:
  # python scripts/build_smoke_dataset.py
  python -m src.features.market_features --test
  python -m src.features.nlp_features --test
  python -m src.data_collection.chart_generator --test --step 2
  python -m src.features.cv_features --test
  python -m src.models.train_ml --config C
  pytest tests/ -q   # 76 tests (no downloads required)
  ```

- **Verify committed results (no data download needed):**
  ```bash
  python scripts/verify_results.py   # checks all headline F1 metrics against documentation
  ```
  Expected output: `ALL CHECKS PASSED — results match documentation.`

- **Reproducibility notes:** Python 3.10+. All random seeds fixed via [`src/config.py`](https://github.com/Scampoloni/financial-market-predictor/blob/main/src/config.py). Pre-computed artifacts are committed to the repository (`models/`, `data/processed/`) so the app can be launched without re-running the full pipeline. For byte-exact reproducibility use `requirements-pinned.txt`.

---

## 5. Optional Bonus Evidence

- [x] Third selected block implemented with strong quality — Computer Vision (EfficientNet-B0, domain fine-tuning, bi-daily chart generation, PCA compression, full ablation measurement).
- [x] More than two data sources used with clear added value — Yahoo Finance OHLCV, RSS feeds, NewsAPI, FinBERT (HuggingFace), EfficientNet-B0 (torchvision) with fine-tuning on domain data.
- [x] Extended evaluation — Bootstrap 95 % CI (N = 2,000), 5-fold TimeSeriesSplit, per-class precision/recall/F1, multi-horizon comparison (5-day vs 21-day), per-class shift analysis (DOWN/UP recall trade-off across configs), statistical significance of ablation deltas. See [`notebooks/06_evaluation_ablation.ipynb`](notebooks/06_evaluation_ablation.ipynb).
- [x] Ethics, bias, or fairness analysis — See dedicated Section 6 below.
- [x] Comprehensive test suite — 76 pytest tests covering feature parquet contracts (column names, value ranges, binary flags), ablation result validity (including Config D), 21-day model bundle contracts, VADER and FinBERT pipeline output format, CV PCA dimension and variance checks, temporal split non-overlap, and model artifact integrity. All tests run without network access or GPU. See [`tests/`](tests/).

Evidence for selected bonus items: full ablation results in [`data/processed/ablation_results.json`](data/processed/ablation_results.json); evaluation visualisations in [`data/processed/ablation_f1_bar.png`](data/processed/ablation_f1_bar.png), [`data/processed/per_class_performance.png`](data/processed/per_class_performance.png), [`data/processed/feature_importance.png`](data/processed/feature_importance.png), [`data/processed/bootstrap_ci.png`](data/processed/bootstrap_ci.png).

---

## 6. Ethics, Bias, and Fairness

> **This is a research prototype. It is not a financial adviser, not a trading signal, and must never be used for real capital allocation.** The app carries an explicit disclaimer on every page.

### 6.1 Data Bias

- **Survivorship bias:** The ticker universe consists of 67 currently-listed S&P 500 large-caps. Companies that were delisted, went bankrupt, or were removed from the index between 2020 and 2026 are absent. This biases the training distribution toward historically successful firms and may overstate UP prediction reliability — fallen stocks, which would produce DOWN labels, are invisible to the model.
- **English-language concentration:** All news inputs come from English-language RSS feeds and NewsAPI, dominated by US financial media (Reuters, MarketWatch, Yahoo Finance). Non-English coverage of the same companies — particularly relevant for companies with significant European or Asian operations — is invisible to the NLP block, which may distort sentiment for multinational firms.
- **Source concentration:** A small number of high-volume news feeds account for most headlines. Sentiment driven by niche, regional, or specialised outlets is under-represented. A story breaking first in a domain-specific publication may not reach the model's pipeline in time to predict the associated price move.
- **Temporal distribution shift:** The model is trained on 2020–2024 data that includes the COVID-19 crash, post-pandemic recovery, and aggressive Fed rate cycles — exceptional macro regimes. Performance on future data under different macro conditions (e.g. prolonged low-volatility, deflationary environments) is not guaranteed.

### 6.2 Model Fairness and Market Access

- **Retail vs institutional inequality:** Institutional investors operate with sub-millisecond latency feeds, proprietary alternative data (satellite imagery, credit-card transaction data, earn-call audio NLP), and larger compute budgets than the public sources used here. Any signal captured by this model is likely already arbitraged away at the institutional level. Retail investors who act on model predictions without understanding the ~0.50 F1 ceiling and the EMH context could incur losses.
- **Amplification risk (self-fulfilling prophecy):** If this model — or a similar public model — were deployed at scale and its predictions widely followed, coordinated buy/sell signals from many users acting simultaneously could move prices in the predicted direction, not because the model is accurate, but because of coordinated action. This feedback loop would corrupt future evaluation and could amplify market volatility.

### 6.3 Limitations and Honest Scope

- **Accuracy ceiling:** ~0.50 F1 on 5-day directional prediction is consistent with the semi-strong Efficient Market Hypothesis (EMH): public information is rapidly priced in, leaving little exploitable signal in the publicly-available inputs used here. The Config D analyst features result (Δ = −0.0011 vs Config C) is a direct empirical confirmation: even professional analyst ratings — which represent significant analytical effort — add no measurable predictive value at the 5-day horizon.
- **No causal claims:** All model outputs are correlational. A high UP probability does not mean the model has identified a causal driver of the price move; it means the feature pattern resembles historical UP episodes.
- **Not a trading system:** The model outputs directional probabilities, not position sizes, entry/exit rules, or risk management logic. Translating a probability into a trading decision requires additional infrastructure (risk limits, transaction costs, portfolio constraints) that is outside the scope of this project.
