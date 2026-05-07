# Project Documentation Template (Filled)

Project: Financial Market Predictor  
Repository: https://github.com/Scampoloni/financial-market-predictor  
Deployment: https://financial-market-predictorr.streamlit.app/  
Date: 2026-05-06

## 1. Project Idea & Methodology

### Problem definition and objectives
Goal is to predict 5-day stock direction (`UP`/`DOWN`) for 67 S&P 500 stocks using an integrated multimodal system.  
Primary objective is not trading automation, but evaluation of whether combining ML numeric + NLP + CV yields more robust out-of-sample signal than a single block.

### Use case motivation
Use case is realistic for decision support: analysts receive a directional probability signal plus contextual evidence (sentiment and chart-derived patterns).  
System is explicitly a research prototype, not financial advice.

### Block integration (conceptual + technical)
- ML Numeric Data: technical indicators + regime features from OHLCV.
- NLP: FinBERT/VADER sentiment features and derived text-based signals.
- CV: EfficientNet-B0 chart embeddings reduced via PCA.
- Integration mechanism: all block outputs are joined into one ticker-date feature matrix and evaluated via A/B/C ablation with identical temporal splits.

### Scope and assumptions
- Scope: daily prediction for selected liquid large-cap US stocks.
- Assumptions: public data only, temporal causality preserved by split.
- Non-goals: HFT, portfolio optimization, live order execution.

## 2. Data & Preprocessing

### Data sources
- Yahoo Finance (OHLCV, indices including VIX): structured numeric time series.
- RSS + NewsAPI headlines: unstructured text.
- Generated candlestick chart images: vision input.
- Analyst-derived signals (where available): auxiliary structured block.

### Cleaning and preparation
- Strict date indexing and ticker alignment.
- Missing NLP/CV handled through deterministic fallback/imputation strategy.
- Feature matrices persisted in `data/processed/*.parquet`.

### Block-specific preprocessing
- ML: returns, volatility, RSI/MACD/Bollinger/ATR, cyclical calendar features, sector encoding.
- NLP: FinBERT/VADER scoring, aggregation windows, momentum/dispersion/surprise features, embedding PCA.
- CV: chart image generation, EfficientNet embedding extraction, PCA to compact components.

### Feature engineering / augmentation
- Config A/B/C ablation setup makes each block contribution measurable.
- Additional interaction features include sentiment-volume interactions and regime-aware signals.

### EDA key findings
- Non-stationarity across market regimes confirms need for temporal split.
- Sparse raw text coverage requires fallback strategy.
- Visual signals are weak standalone but complementary in fused setup.

## 3. Modeling & Implementation

### Model choice justification
- RandomForest: robust baseline on tabular data.
- LightGBM: strong gradient-boosting baseline with Optuna tuning.
- Stacking (RF + XGB + LGB): ensemble candidate for complementary bias/variance behavior.

### Training / fine-tuning strategy
- TimeSeriesSplit CV on training era.
- Model selection by validation (`val_f1_macro`) only.
- Final test evaluation executed after selection.
- CV model fine-tuned via `scripts/finetune_cnn.py`.

### Comparison and iterations
- At least two models compared per config (actually three).
- Cross-config comparison: A (ML), B (ML+NLP), C (ML+NLP+CV).
- Iterative upgrades documented in README development journey and notebooks.

#### Model comparison summary

| Config | Features | # Features | Best Model | CV F1 ± std | Test F1 | Test Acc | Δ vs Baseline |
|--------|----------|-----------|------------|:-----------:|:-------:|:--------:|:-------------:|
| **A** | Market only | 28 | LightGBM | 0.509 ± 0.016 | **0.4970** | **0.4971** | — |
| **B** | Market + NLP | 56 | LightGBM | 0.510 ± 0.027 | 0.4826 | 0.4842 | −0.0143 |
| **C** | Market + NLP + CV | 66 | LightGBM | 0.511 ± 0.018 | 0.4861 | 0.4863 | −0.0109 |

### Technical implementation details
- Core stack: pandas, scikit-learn, lightgbm, xgboost, torch/torchvision, streamlit, plotly.
- Separation of concerns:
  - Training: `src/models/train_ml.py`
  - Inference: `src/models/predict.py`
  - UI: `app.py` and `src/app/pages/*`

## 4. Evaluation & Analysis

### Evaluation strategy
- Temporal split:
  - Train: <= 2024-06
  - Validation: 2024H2
  - Test: 2025
- Metrics: macro F1, accuracy, per-class precision/recall/F1.

### Performance analysis
- Ablation metrics stored in `data/processed/ablation_results.json`.
- Model comparison page visualizes per-config and per-model results.

#### Ablation results (held-out 2025 test set)

| Config | Test F1 (macro) | Test Acc | Per-class DOWN F1 | Per-class UP F1 |
|--------|:--------------:|:--------:|:-----------------:|:---------------:|
| A (Market) | 0.4970 | 0.4971 | 0.4876 | 0.5063 |
| B (Market + NLP) | 0.4826 | 0.4842 | 0.5113 | 0.4539 |
| C (Market + NLP + CV) | 0.4861 | 0.4863 | 0.4978 | 0.4744 |

Bootstrap 95% CI for Config C (N=2,000): [0.487, 0.502]. Overlapping CIs across configs
are consistent with the semi-strong Efficient Market Hypothesis.

#### NLP block evaluation (curated 50-headline benchmark)

| Metric | FinBERT | VADER |
|--------|:-------:|:-----:|
| Direction accuracy | 0.792 | 0.800 |
| Macro F1 | 0.791 | 0.796 |
| Score–direction correlation | 0.694 | 0.656 |
| Abstentions (NEUTRAL) | 0 / 50 | 9 / 50 |
| Inter-model score correlation | 0.47 | — |

FinBERT is more decisive (no abstentions); VADER abstains on ambiguous headlines.
Both are used as complementary features in Config B/C.

#### Multi-horizon comparison

| Horizon | Test F1 | Test Acc | Test Rows |
|---------|:-------:|:--------:|:---------:|
| 5-day (primary) | 0.4861 | 0.4863 | 20,033 |
| 21-day | 0.4961 | 0.4989 | 18,961 |

### Error analysis
- Sector and regime sensitivity analyzed in notebooks and summarized in README.
- Explicit limitations documented (coverage sparsity, EMH ceiling, survivorship risk).

### Block-specific evaluation
- NLP: FinBERT vs VADER comparison (quantitative + qualitative).
- CV: confusion/inspection in CV notebook plus integration impact in ablation.
- ML: cross-model and cross-config quantitative comparison.

## 5. Deployment

### Working deployment URL
https://financial-market-predictorr.streamlit.app/

### Training vs inference separation
- Training jobs and artifact production are offline.
- Streamlit loads persisted artifacts only for inference/analysis.

### Screenshots (key functionality)
- `docs/screenshots/01_prediction_flow.png`
- `docs/screenshots/02_model_analysis.png`
- `docs/screenshots/03_nlp_cv_integration.png`

## 6. Execution Instructions

### Reproduce locally
1. `pip install -r requirements.txt`
2. Build/prepare data via modules in `src/data_collection/` and `src/features/`.
3. Train ablation models: `python -m src.models.train_ml`
4. Run app: `streamlit run app.py`

### Smoke path (faster)
Use smoke dataset scripts documented in README to verify end-to-end pipeline quickly.

## Submission Notes

- Deadline: 07 June 2026, 18:00.
- Add GitHub collaborators before submission:
  - `jasminh` (Jasmin Heierli)
  - `bkuehnis` (Benjamin Kühnis)
