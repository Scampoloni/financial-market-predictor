# Financial Market Direction Research

> An AI-assisted research prototype evaluating whether technical market data, financial-news sentiment, and candlestick-image features improve short-horizon equity-direction classification.

> **Key takeaway:** the retained legacy artefacts did not show that the multimodal configurations outperform the market-only configuration. This repository demonstrates an end-to-end applied research workflow and the difficulty of finding robust short-horizon signals in public data; it is not a profitable prediction system.

**Research prototype only.** This project is not a trading system and does not provide investment, financial, or trading advice. Historical outputs do not imply future performance.

## At a glance

| Question | What was investigated |
| --- | --- |
| Research question | Do public market, headline-sentiment, and chart-image features improve 5- or 21-trading-day direction classification? |
| Universe | 67 large-cap US equities across seven sectors, plus market context data |
| Core workflow | Collect -> engineer features -> temporally validate -> compare feature blocks -> serve an educational demo |
| Primary metric | Macro F1, supported by accuracy and per-class precision/recall |
| Central finding | Legacy results were near chance-level and did not demonstrate robust out-of-sample directional predictability. |

## Demo status

The Streamlit application is an educational interface for inspecting legacy experimental artefacts and generating an **experimental direction estimate** from live market inputs. Its model confidence is not calibrated and must not be used for decisions. The separate news Q&A feature searches the bundled headline index; public demo mode does not fetch live news or call a paid LLM API.

The historic deployment link is [financial-market-predictorr.streamlit.app](https://financial-market-predictorr.streamlit.app/). Verify it manually before linking it from a CV: deployment state and artefact freshness can change independently of this repository.

| Estimate interface | Analysis interface | NLP/CV interface |
| :---: | :---: | :---: |
| ![Estimate interface](docs/screenshots/01_prediction_flow.png) | ![Analysis interface](docs/screenshots/02_model_analysis.png) | ![NLP/CV interface](docs/screenshots/03_nlp_cv_integration.png) |

## System overview

```mermaid
flowchart LR
    A[Yahoo Finance OHLCV] --> B[Market features and forward-return labels]
    C[RSS / optional NewsAPI headlines] --> D[FinBERT + VADER features]
    A --> E[30-session candlestick images]
    E --> F[EfficientNet embeddings + PCA]
    B --> G[Temporal ablation training]
    D --> G
    F --> G
    G --> H[Saved research artefacts]
    H --> I[Streamlit educational demo]
    C --> J[Headline retrieval index]
    J --> I
```

## Data and modalities

- **Market:** OHLCV data collected through `yfinance`, technical indicators, calendar features, sector dummies, and VIX context.
- **NLP:** FinBERT financial-sentiment scores, VADER scores, headline volume, and PCA-reduced FinBERT embeddings. Direct ticker-news coverage is sparse; sector/market fallbacks and forward filling increase row completeness, but do **not** create direct news information.
- **Computer vision:** 30-trading-day candlestick images ending at the prediction timestamp; EfficientNet-B0 embeddings are compressed to ten PCA components.
- **Analyst data:** the repository can construct analyst features from `yfinance`, but current aggregate recommendation and price-target values are not a verified point-in-time historical source. They are excluded from any validity claim.

See [data-card.md](docs/data-card.md) for source, missingness, and licensing considerations.

## Experimental design

The target is the sign of the close-to-close forward return after 5 or 21 trading days (`UP` if positive; otherwise `DOWN`). Features are intended to be available at the daily close. Chart windows end on that date and do not include target-period candles.

The revised training code uses chronological train, validation, and reporting partitions, purges the forward-label horizon at boundaries, applies `TEST_END`, and uses date-grouped expanding folds with an embargo. This prevents rows for different tickers on one session from being split across a fold.

The saved artefacts pre-date this cleanup. Their results must be treated as **legacy exploratory results**, not a clean final benchmark, because the former split logic did not purge forward-label overlap or apply the configured test end. The analyst configuration is additionally invalid for historical evaluation. Regenerate results with the revised code before publishing a final table.

Details: [methodology.md](docs/methodology.md) and [leakage-audit.md](docs/leakage-audit.md).

## Legacy results: diagnostic only

The existing `data/processed/ablation_results.json` contains the following 5-day diagnostic outputs. They are shown for transparency, not as validated performance claims.

| Configuration | Feature blocks | Stored macro F1 | Interpretation |
| --- | --- | ---: | --- |
| A | Market only | 0.4970 | Legacy diagnostic; rerun required with purged splits. |
| B | Market + NLP | 0.4826 | Below A in this legacy run; sparse direct-news coverage limits interpretation. |
| C | Market + NLP + CV | 0.4861 | Below A in this legacy run; rerun required. |
| D | C + analyst | 0.4850 | Methodologically invalid for historical reporting because aggregate analyst data is current. |

The observed pattern is consistent with the difficulty of extracting short-horizon signals from public data. It does not establish profitability, alpha, a universal performance ceiling, market efficiency, or the usefulness of a modality.

## Architecture and technologies

| Area | Used technologies |
| --- | --- |
| Data and analysis | Python, pandas, NumPy, yfinance, pyarrow |
| ML and evaluation | scikit-learn, LightGBM, Optuna, pytest |
| NLP | Hugging Face Transformers, ProsusAI/FinBERT, NLTK/VADER, sentence-transformers |
| Computer vision | PyTorch, torchvision, EfficientNet-B0, mplfinance |
| Application | Streamlit, Plotly |
| News Q&A | Local vector retrieval, optional Anthropic Claude or OpenAI generation; not Gemini |

FinBERT is implemented as a feature pipeline. Candlestick images and EfficientNet embeddings are implemented. The news Q&A component is separate from the classification pipeline, uses Claude/OpenAI only when an environment key is configured, and displays retrieved headlines. It is not a Gemini RAG feature and should not be described as part of prediction.

## Reproducibility

### Lightweight inspection

The tracked artefacts and tests support local inspection without downloading data or calling external APIs.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest tests -q
ruff check src tests scripts app.py
streamlit run app.py
```

### Full research rerun

Full reproduction needs data collection, FinBERT processing, chart generation, CNN embedding extraction, and model training. It can take hours and requires meaningful local storage and compute. Do not expect it to be a quick or byte-identical command sequence: external data, model versions, and data-provider availability can change.

```powershell
python -m src.data_collection.market_collector
python -m src.data_collection.news_scraper  # optional API key only if deliberately used
python -m src.features.market_features
python -m src.features.nlp_features
python -m src.data_collection.chart_generator --step 2
python -m src.features.cv_features --finetuned
python -m src.models.train_ml
```

Before a final rerun, exclude analyst features until a licensed point-in-time source is available. See [reproducibility.md](docs/reproducibility.md).

## Quality checks

- Unit tests cover feature contracts, target construction, split boundaries, model bundles, NLP/CV schemas, and app bootstrap.
- The revised temporal tests verify date-grouped fold separation and embargoed split boundaries.
- GitHub Actions runs ruff and pytest without secrets, paid APIs, raw datasets, or model training.

## Repository structure

```text
src/                    Production-style data, feature, model, NLP, CV, and app modules
scripts/                Training, evaluation, and smoke-data utilities
tests/                  Lightweight unit and artefact-contract checks
docs/                   Methodology, model/data cards, leakage audit, architecture, reproduction
data/processed/         Small processed artefacts and legacy diagnostic results
models/                 Tracked demo artefacts; review before long-term Git storage
notebooks/              Exploratory development record (may differ from the revised pipeline)
```

## Important limitations

- Stored metrics require regeneration with the revised purge and reporting-period controls.
- Current analyst aggregate data is not point-in-time and is not valid for historical model results.
- News timestamps are normalised to UTC calendar dates; session-close, after-hours, holiday, and source-timezone alignment require a market-calendar-aware redesign before strong timing claims.
- Fallback sentiment primarily improves completeness, not direct ticker-news coverage.
- Overlapping chart windows require careful temporal training and held-out evaluation; the CNN has temporal train/validation code but no final end-to-end rerun after this audit.
- Market data and headlines have provider terms and possible redistribution restrictions. Raw inputs should not be committed without review.

## AI-assisted development approach

This project was developed as an AI-assisted learning and research project. Modern AI tools supported parts of implementation, debugging, documentation, and iteration. The author defined the research question, structured and connected the components, reviewed methodological choices, tested the application, and interpreted the results. The repository demonstrates applied problem structuring and critical evaluation; it is not evidence of expert-level proficiency in every model, library, or quantitative technique used.

## Further documentation

- [Methodology](docs/methodology.md)
- [Model card](docs/model-card.md)
- [Data card](docs/data-card.md)
- [Leakage audit](docs/leakage-audit.md)
- [Architecture](docs/architecture.md)
- [Reproducibility](docs/reproducibility.md)

## Publication checklist

Before pinning or linking this project publicly: regenerate valid metrics, verify the live deployment, review model-artefact storage and data rights, choose a licence with the owner, and update the repository description/topics in GitHub. Do not claim price prediction, trading performance, Gemini integration, or clean out-of-sample validation until the documented manual actions are complete.
