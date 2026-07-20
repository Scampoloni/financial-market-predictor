# Methodology

## Research question

This prototype asks whether public market features, financial-news features, and candlestick-image embeddings improve short-horizon equity direction classification relative to a market-feature configuration.

## Target and timestamp

For each ticker and market date `t`, the target is `UP` when `close[t + h] / close[t] - 1 > 0`, else `DOWN`, where `h` is 5 or 21 trading days. Features are intended to be calculated from data available by the market close on `t`. The target itself is future information and is never a model feature.

## Partitions and validation

The configured periods are training through 2024-06-30, validation in 2024H2, and reporting from 2025-01-01 through the configured `TEST_END` (currently 2026-03-21). Revised code applies three safeguards:

1. It removes the final `h` business days of train, validation, and reporting ranges so a label cannot reach into the next partition.
2. It bounds reporting data by `TEST_END`, instead of accepting every row after `TEST_START`.
3. It uses `PurgedDateTimeSeriesSplit`, which keeps all tickers from a date together and puts a forward-label embargo before each fold's test dates.

An A/B rerun applied these safeguards and stored source hashes, 20,033 reporting predictions per configuration, confusion matrices, class balance, and benchmarks in `data/processed/rerun_purged_ab_*`. The final usable reporting dates are 2025-01-02 through 2026-03-13 after target availability and purge. C has not yet been rerun; legacy artefacts remain for transparency only.

## Feature availability and preprocessing

Technical indicators use current and prior OHLCV observations. Chart generation uses a trailing 30-trading-day window ending at `t`. NLP PCA and CV PCA code fits on pre-training-period rows when enough relevant rows exist, then transforms later rows. Any fallback to all rows due to insufficient training samples is a leakage risk and must block final reporting.

Rows with missing NLP/CV content are filled with neutral values and availability indicators. This is a modelling choice, not evidence that information was observed. Tree models do not require a scaler or imputer outside these explicit feature transforms.

## Ablation design

`A` is market-only, `B` adds NLP, and `C` adds CV. Analyst data is isolated to `D`, rather than being silently joined to B/C. D is exploratory and invalid for historic performance reporting because it can use current aggregate analyst data. Revised ablations exclude the legacy stacking path because its internal K-fold split is not appropriate for temporal panel data.

## Evaluation

Macro F1 is the primary metric because it weights `UP` and `DOWN` equally. Report accuracy, class balance, per-class precision/recall/F1, confusion matrices, and simple benchmarks alongside it. Select a model by validation performance only; inspect the reporting period only after a preregistered-like rerun. Store predictions to permit independent metric reproduction.
