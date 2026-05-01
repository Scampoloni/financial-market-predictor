# A-E Requirements Matrix

This matrix is the operational checklist for reaching full compliance with the ZHAW combined-project requirements.

## A. General Project Requirements

| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Combine at least two blocks (ML/NLP/CV) | Done | `src/models/train_ml.py`, `src/features/nlp_features.py`, `src/features/cv_features.py` | None |
| Meaningful conceptual and technical integration | Done | Shared feature matrix and Config A/B/C ablation in `src/models/train_ml.py`, integration narrative in `README.md` | None |
| Multiple and different data sources | Done | `src/data_collection/market_collector.py`, `src/data_collection/news_scraper.py`, `src/data_collection/chart_generator.py`, source-size table in `README.md` | None |
| Well-motivated and realistic use case | Done | Motivation + assumptions/non-goals in `README.md` | None |
| Independently completed and documented | Done | `README.md`, notebooks, `docs/FINAL_SUBMISSION_RUNBOOK.md`, live deployment at https://financial-market-predictorr.streamlit.app/ | None |

## B. Documentation Requirements

### B1. Project Idea and Methodology
| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Clear problem definition and objectives | Done | `README.md` opening section | None |
| Realistic use case motivation | Done | `README.md` Motivation section | None |
| Explain block combination | Done | Architecture + ablation description + interaction paragraph in `README.md` | None |
| Scope and assumptions | Done | Explicit scope/assumptions/non-goals in `README.md` | None |

### B2. Data and Preprocessing
| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Data sources (type, origin, size) | Done | `README.md` Data Sources table with file and row counts | None |
| Data cleaning/preparation | Done | Preprocessing bullets in `README.md` + feature modules under `src/features/` | None |
| Block-specific preprocessing | Done | `src/features/market_features.py`, `src/features/nlp_features.py`, `src/features/cv_features.py` | None |
| Feature engineering/augmentation | Done | Feature modules + ablation setup | None |
| EDA with key findings | Done | `notebooks/01_eda.ipynb` + EDA findings bullets in `README.md` | None |

### B3. Modeling and Implementation
| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Model/prompt selection justified | Done | README **Model Selection Protocol** subsection + `data/processed/ablation_results.json` (`selection_metric: val_f1_macro` for A/B/C) + `src/models/train_ml.py` | None |
| Training/finetuning strategy | Done | `src/models/train_ml.py`, `scripts/finetune_cnn.py` | None |
| Model/prompt approach comparison | Done | Ablation and model comparisons | Ensure one concise summary table in README |
| Iterations and improvements | Done | `UPGRADE_PLAN.md`, README phases | None |
| Technical implementation details | Done | README structure + `requirements.txt` | None |

### B4. Evaluation and Analysis
| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Clear strategy (split/metrics/qualitative) | Done | README + evaluation notebook/code | None |
| Performance analysis | Done | Ablation results in README/notebook + **bootstrap 95% CI in `notebooks/06_evaluation_ablation.ipynb` (Section 9)**: Config C CI [0.487, 0.502], N=2,000 iterations | None |
| Error analysis | Done | Explicit error analysis section in `README.md` + sector-F1 and VIX-regime analysis in `notebooks/06_evaluation_ablation.ipynb` (Section 8) | None |
| Interpretation of results | Done | README interpretation subsection + bootstrap overlap interpretation (EMH, Section 9 notebook 06) | None |
| Block-specific evaluation | Done | CV isolated eval in `notebooks/04_cv_pipeline.ipynb` (Section 7) + NLP curated benchmark (FinBERT acc=0.792, VADER acc=0.800) in `notebooks/03_nlp_pipeline.ipynb` (Section 9) | None |

### B5. Deployment
| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Working deployment URL | Done | https://financial-market-predictorr.streamlit.app/ | None |
| Separation training vs inference | Done | `src/models/train_ml.py` vs `src/models/predict.py` | None |
| Screenshots for key functionality | Done | `docs/screenshots/` — 3 real PNG captures in `README.md` | None |

### B6. Execution Instructions
| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Reproducible local run instructions | Done | `README.md` Setup + smoke-test dataset instructions | Keep commands synced with modules |

## C. Assessment Criteria Readiness

| Criterion | Readiness | Evidence | Remaining Action |
|---|---|---|---|
| Clarity | High | Structured README sections + explicit compliance checklist | None |
| Technical correctness | High | Pipelines and module separation + smoke tests in `tests/` | None |
| Depth of analysis | High | Ablation + interpretation + explicit error analysis | None |
| Quality of integration | High | Shared A/B/C pipeline + integration narrative | None |
| Reproducibility | High | Setup instructions + smoke tests + real screenshots + live deployment URL | None |

## D. Submission

| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Submit GitHub link by deadline | In progress | Repository at https://github.com/Scampoloni/financial-market-predictor | Submit link by 07 June 2026, 18:00 |
| Add `jasminh` and `bkuehnis` | Pending | — | Add both as collaborators on GitHub (Settings → Collaborators) |

## E. Specific Block Requirements

### E (General rules for combined blocks)
| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Blocks clearly identifiable | Done | `src/features/*`, `src/nlp/*`, `src/cv/*` | None |
| Blocks integrated in one coherent app | Done | `src/models/train_ml.py`, `app.py` | None |
| Interaction via shared data/features/outputs | Done | A/B/C feature integration + explicit interaction paragraph in `README.md` | None |

### E.1 ML Numeric Data
| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Structured dataset used | Done | Market data collectors and processed dataset | None |
| EDA performed | Done | `notebooks/01_eda.ipynb` + concise summary in `README.md` | None |
| Feature engineering/selection/transformation | Done | `src/features/market_features.py` | None |
| At least two models trained/compared | Done | `src/models/train_ml.py` | None |
| Quantitative metrics | Done | Evaluation outputs and README tables | None |
| Interpretation and error analysis | Done | README interpretation + explicit error analysis block + sector-F1 breakdown and VIX-regime error analysis in `notebooks/06_evaluation_ablation.ipynb` (Section 8) | None |
| Integration with another block | Done | NLP/CV features in final model | None |

### E.2 NLP
| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Text data clearly defined | Done | News source + corpus sections | None |
| NLP preprocessing/prompt design | Done | NLP pipeline modules | None |
| At least one NLP approach | Done | FinBERT/VADER/RAG | None |
| At least one comparison | Done | Explicit NLP comparison table in `README.md` + **curated 24-headline benchmark** in `notebooks/03_nlp_pipeline.ipynb` (Section 9): FinBERT acc=0.792 vs VADER acc=0.800, score-direction corr 0.694 vs 0.656, confusion matrices | None |
| Qualitative/quantitative evaluation | Done | Block evaluation bullets + ablation metrics in `README.md` + FinBERT vs VADER confusion matrices, direction accuracy, macro-F1, score-direction correlation in `notebooks/03_nlp_pipeline.ipynb` (Section 9) | None |
| Integration benefit with other blocks | Done | NLP features fed into ML model | None |

### E.3 CV
| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Image data used | Done | `data/raw/charts/` + generator | None |
| Image preprocessing/augmentation | Done | chart generation + CV feature extraction | None |
| Vision model training/finetuning/application | Done | `src/cv/chart_classifier.py`, `scripts/finetune_cnn.py` | None |
| Evaluation by metrics/visual inspection | Done | `notebooks/04_cv_pipeline.ipynb` (Section 7 confusion matrix + visual inspection) + `README.md` |
| Behavior limitations interpreted | Done | Dedicated limitations bullets in `README.md` (Error Analysis) |
| Integration contribution to another block | Done | CV features in Config C | None |

## Remaining Actions

1. Add GitHub collaborators `jasminh` and `bkuehnis` (Settings → Collaborators).
2. Submit GitHub link by 07 June 2026, 18:00.
