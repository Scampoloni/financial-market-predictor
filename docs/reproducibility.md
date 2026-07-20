# Reproducibility

## Supported environment

Use Python 3.10 or 3.11 and install `requirements-pinned.txt` when compatible with the local platform. `requirements.txt` is more flexible but does not guarantee identical serialised-model behaviour. The project has primarily been exercised on Windows.

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
