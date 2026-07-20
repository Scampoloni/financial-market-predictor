# Reproducibility

## Supported environment

Use Python 3.13. The exact versions in `requirements-pinned.txt` were verified together on Python 3.13 / Windows 11; `requirements.txt` is more flexible but does not guarantee identical serialised-model behaviour.

## Lightweight verification

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest tests -q
ruff check src tests scripts app.py
python -m py_compile src/models/splits.py src/models/train_ml.py scripts/train_21d.py
```

These checks do not download raw market data, invoke FinBERT, train the CNN, or call paid APIs.

## Tracked smoke pipeline

The tracked smoke dataset verifies data and feature-pipeline plumbing in a clean checkout:

```powershell
python scripts/use_smoke_data.py --activate
python -m src.features.market_features --test
python -m src.features.nlp_features --test
python -m src.data_collection.chart_generator --test
python -m src.features.cv_features --test
python scripts/use_smoke_data.py --restore
```

On the audited Windows/Python 3.13 environment this sequence completed in about 92 seconds. The bundled smoke-news files are empty and the three generated charts are insufficient to fit ten PCA components, so the run deliberately produces neutral NLP features and zero CV-PCA features. It verifies paths, schemas, imports, feature generation, and safe raw-data restoration; it is not a model-quality test and cannot run the temporal A/B evaluation.

## Audited A/B evaluation rerun

With the tracked processed inputs present locally, run the bounded A/B evaluation without downloading data or retraining the CNN:

```powershell
python -m scripts.rerun_purged_ab --trials 40
```

It excludes analyst features and stacking, applies date-grouped purged folds, and writes separate result and prediction artefacts. It took roughly 25 minutes on the audited Windows environment. It does not rebuild raw inputs and therefore records input SHA-256 hashes for traceability.

## Environment variables

Copy `.env.example` to `.env`. `NEWS_API_KEY`, `CLAUDE_API_KEY`, and `OPENAI_API_KEY` are optional. Never commit the `.env` file. Leave paid-provider keys unset for public demos.

## Full rerun

1. Obtain and snapshot source data subject to provider terms.
2. Generate market and NLP features; document source dates and headline session assignment.
3. Generate charts and train the CNN using temporal partitions only.
4. Fit PCA only on training observations; fail rather than use an all-data fallback.
5. Exclude analyst data until a point-in-time dataset is available.
6. Run `python -m src.models.train_ml` and save predictions, metrics, class balance, data hash, and model metadata.

The full workflow requires substantial disk space, data downloads, a local model download, and potentially hours of CPU/GPU compute. It is not expected to run in CI.

The audited checkout used approximately 1.85 GiB for the pinned virtual environment, 243 MiB for current raw/processed/model data, and 114 MiB for `.git`. Allow at least 3 GiB plus any external Hugging Face cache and additional chart snapshots.

## Streamlit deployment

Streamlit Community Cloud's Python version is selected in the deployment's Advanced settings rather than `.streamlit/config.toml`. Configure the public app to deploy `main` with Python 3.13, then reboot or redeploy and verify the rendered tabs and disclaimer.
