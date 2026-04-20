# A-E Requirements Matrix

This matrix is the operational checklist for reaching full compliance with the ZHAW combined-project requirements.

## A. General Project Requirements

| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Combine at least two blocks (ML/NLP/CV) | Done | `src/models/train_ml.py`, `src/features/nlp_features.py`, `src/features/cv_features.py` | None |
| Meaningful conceptual and technical integration | Done | Shared feature matrix and Config A/B/C ablation in `src/models/train_ml.py`, integration narrative in `README.md` | None |
| Multiple and different data sources | Done | `src/data_collection/market_collector.py`, `src/data_collection/news_scraper.py`, `src/data_collection/chart_generator.py`, source-size table in `README.md` | None |
| Well-motivated and realistic use case | Done | Motivation + assumptions/non-goals in `README.md` | None |
| Independently completed and documented | Done (repo-side) | `README.md`, notebooks, `UPGRADE_PLAN.md`, `docs/FINAL_SUBMISSION_RUNBOOK.md` | External deployment publication still required |

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
| Model/prompt selection justified | Done | README + model training code | None |
| Training/finetuning strategy | Done | `src/models/train_ml.py`, `scripts/finetune_cnn.py` | None |
| Model/prompt approach comparison | Done | Ablation and model comparisons | Ensure one concise summary table in README |
| Iterations and improvements | Done | `UPGRADE_PLAN.md`, README phases | None |
| Technical implementation details | Done | README structure + `requirements.txt` | None |

### B4. Evaluation and Analysis
| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Clear strategy (split/metrics/qualitative) | Done | README + evaluation notebook/code | None |
| Performance analysis | Done | Ablation results in README/notebook | None |
| Error analysis | Done | Explicit error analysis section in `README.md` | None |
| Interpretation of results | Done | README interpretation subsection | None |
| Block-specific evaluation | Done | Block-specific evaluation bullets in `README.md` + notebooks/modules | None |

### B5. Deployment
| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Working deployment URL | External | Deployment section in `README.md` and runbook contain publish steps | Publish app and insert URL |
| Separation training vs inference | Done | `src/models/train_ml.py` vs `src/models/predict.py` | None |
| Screenshots for key functionality | In progress | `docs/screenshots/` placeholders + links in `README.md` | Replace placeholders with real captures from deployed app |

### B6. Execution Instructions
| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Reproducible local run instructions | Done | `README.md` Setup and pipeline commands | Keep commands synced with modules |

## C. Assessment Criteria Readiness

| Criterion | Readiness | Evidence | Remaining Action |
|---|---|---|---|
| Clarity | High | Structured README sections + explicit compliance checklist | None |
| Technical correctness | High | Pipelines and module separation + smoke tests in `tests/` | None |
| Depth of analysis | High | Ablation + interpretation + explicit error analysis | None |
| Quality of integration | High | Shared A/B/C pipeline + integration narrative | None |
| Reproducibility | Medium-High | Setup instructions + smoke tests + screenshot scaffold | Add public URL and real screenshot captures |

## D. Submission

| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Submit GitHub link by deadline | In progress (external) | Repository exists | Final submission upload/URL confirmation |
| Add `jasminh` and `bkuehnis` | External | Not verifiable in code | Add collaborators on GitHub and confirm |

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
| Interpretation and error analysis | Done | README interpretation + explicit error analysis block | None |
| Integration with another block | Done | NLP/CV features in final model | None |

### E.2 NLP
| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Text data clearly defined | Done | News source + corpus sections | None |
| NLP preprocessing/prompt design | Done | NLP pipeline modules | None |
| At least one NLP approach | Done | FinBERT/VADER/RAG | None |
| At least one comparison | Done | Explicit NLP comparison table in `README.md` |
| Qualitative/quantitative evaluation | Done | Block evaluation bullets + ablation metrics in `README.md` and notebooks |
| Integration benefit with other blocks | Done | NLP features fed into ML model | None |

### E.3 CV
| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Image data used | Done | `data/raw/charts/` + generator | None |
| Image preprocessing/augmentation | Done | chart generation + CV feature extraction | None |
| Vision model training/finetuning/application | Done | `src/cv/chart_classifier.py`, `scripts/finetune_cnn.py` | None |
| Evaluation by metrics/visual inspection | Done | CV pipeline + ablation impact + dedicated CV notes in `README.md` |
| Behavior limitations interpreted | Done | Dedicated limitations bullets in `README.md` (Error Analysis) |
| Integration contribution to another block | Done | CV features in Config C | None |

## Execution Order for Remaining Work

1. Publish app publicly and insert final deployment URL in `README.md`.
2. Replace screenshot placeholder SVGs with real app captures in `docs/screenshots/`.
3. Confirm GitHub collaborators (`jasminh`, `bkuehnis`) and submission URL.
4. Run final consistency check (commands, names, links) and tag release candidate.
