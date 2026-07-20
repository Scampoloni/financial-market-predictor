# Model Card

## Intended use

Educational inspection of a research workflow for short-horizon binary equity-direction classification. It may support discussion of data alignment, feature engineering, temporal validation, and model limitations.

## Not intended use

Not for investment, trading, credit, portfolio allocation, automated execution, or any decision with financial consequences. A displayed probability is an uncalibrated classifier score, not a probability of return or a recommendation.

## Configurations

Market features are used in A; B adds FinBERT/VADER and derived headline features; C adds EfficientNet chart embeddings. D adds analyst features, but cannot be treated as historically valid without point-in-time data. The app loads retained legacy artefacts for demonstration; it is not a validated live inference service.

## Data and evaluation status

The code defines 2020-2024 training, 2024H2 validation, and 2025 reporting partitions. Current stored metrics pre-date the revised purged split and endpoint enforcement, so they are diagnostic only. A future rerun must record the exact dataset version, artefact hash, sample count, class balance, predictions, and metrics.

## Known failure modes

- sparse direct news coverage masked by fallback values;
- after-hours and session assignment uncertainty;
- overlapping chart windows and feature correlation;
- current analyst aggregate values applied historically;
- changing data-provider responses and model serialisation dependencies;
- model confidence without calibration.

## Risk and ethics

Public financial data can encode survivorship, coverage, and media-attention bias. The model may fail unpredictably during macro events, earnings, corporate actions, or data outages. Do not infer causality or financial advantage from feature importance or historical classifier scores.
