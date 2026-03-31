# A-E Requirements Matrix

This matrix is the operational checklist for reaching full compliance with the ZHAW combined-project requirements.

## A. General Project Requirements

| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Combine at least two blocks (ML/NLP/CV) | Done | `src/models/train_ml.py`, `src/features/nlp_features.py`, `src/features/cv_features.py` | Keep final README section aligned with code |
| Meaningful conceptual and technical integration | Done | Shared feature matrix and Config A/B/C ablation in `src/models/train_ml.py` | Add one explicit integration diagram in README (final polish) |
| Multiple and different data sources | Done | `src/data_collection/market_collector.py`, `src/data_collection/news_scraper.py`, `src/data_collection/chart_generator.py` | Add final source-size table in README |
| Well-motivated and realistic use case | Done | Motivation section in `README.md` | Add 2-3 assumptions/non-goals bullets |
| Independently completed and documented | In progress | `README.md`, notebooks, `UPGRADE_PLAN.md` | Add deployment evidence and screenshots |

## B. Documentation Requirements

### B1. Project Idea and Methodology
| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Clear problem definition and objectives | Done | `README.md` opening section | None |
| Realistic use case motivation | Done | `README.md` Motivation section | None |
| Explain block combination | Done | Architecture + ablation description | Add one short interaction paragraph |
| Scope and assumptions | In progress | Partial in README | Add explicit scope/assumptions subsection |

### B2. Data and Preprocessing
| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Data sources (type, origin, size) | In progress | `README.md` Data Sources | Add precise counts per source/date |
| Data cleaning/preparation | In progress | Feature modules under `src/features/` | Add explicit bullet list in README |
| Block-specific preprocessing | Done | `src/features/market_features.py`, `src/features/nlp_features.py`, `src/features/cv_features.py` | None |
| Feature engineering/augmentation | Done | Feature modules + ablation setup | None |
| EDA with key findings | In progress | `notebooks/01_eda.ipynb` | Add key findings bullets in README |

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
| Error analysis | In progress | Partial interpretation in README | Add explicit error analysis section |
| Interpretation of results | Done | README interpretation subsection | None |
| Block-specific evaluation | In progress | Notebooks and modules | Add concise ML/NLP/CV evaluation bullets |

### B5. Deployment
| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Working deployment URL | Open | Placeholder in README | Add public URL |
| Separation training vs inference | Done | `src/models/train_ml.py` vs `src/models/predict.py` | None |
| Screenshots for key functionality | Open | None yet | Add screenshot files and link them in README |

### B6. Execution Instructions
| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Reproducible local run instructions | Done | `README.md` Setup and pipeline commands | Keep commands synced with modules |

## C. Assessment Criteria Readiness

| Criterion | Readiness | Evidence | Remaining Action |
|---|---|---|---|
| Clarity | High | Structured README sections | Final consistency sweep |
| Technical correctness | High | Pipelines and module separation | Add smoke tests |
| Depth of analysis | Medium-High | Ablation + interpretation | Expand error analysis |
| Quality of integration | High | Shared A/B/C pipeline | Add one integration summary figure/table |
| Reproducibility | Medium | Setup instructions present | Add screenshots, URL, and tests |

## D. Submission

| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Submit GitHub link by deadline | In progress | Repository exists | Final submission checklist before deadline |
| Add `jasminh` and `bkuehnis` | Open (external) | Not verifiable in code | Add collaborators on GitHub and confirm |

## E. Specific Block Requirements

### E (General rules for combined blocks)
| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Blocks clearly identifiable | Done | `src/features/*`, `src/nlp/*`, `src/cv/*` | None |
| Blocks integrated in one coherent app | Done | `src/models/train_ml.py`, `app.py` | None |
| Interaction via shared data/features/outputs | Done | A/B/C feature integration | Add explicit evidence bullets in README |

### E.1 ML Numeric Data
| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Structured dataset used | Done | Market data collectors and processed dataset | None |
| EDA performed | Done | `notebooks/01_eda.ipynb` | Add concise EDA findings summary |
| Feature engineering/selection/transformation | Done | `src/features/market_features.py` | None |
| At least two models trained/compared | Done | `src/models/train_ml.py` | None |
| Quantitative metrics | Done | Evaluation outputs and README tables | None |
| Interpretation and error analysis | In progress | README interpretation available | Add explicit error analysis block |
| Integration with another block | Done | NLP/CV features in final model | None |

### E.2 NLP
| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Text data clearly defined | Done | News source + corpus sections | None |
| NLP preprocessing/prompt design | Done | NLP pipeline modules | None |
| At least one NLP approach | Done | FinBERT/VADER/RAG | None |
| At least one comparison | In progress | FinBERT vs VADER exists in code context | Add explicit comparison table in README |
| Qualitative/quantitative evaluation | In progress | Partial evidence in notebooks | Add explicit evaluation bullets/table |
| Integration benefit with other blocks | Done | NLP features fed into ML model | None |

### E.3 CV
| Requirement | Status | Evidence | Remaining Action |
|---|---|---|---|
| Image data used | Done | `data/raw/charts/` + generator | None |
| Image preprocessing/augmentation | Done | chart generation + CV feature extraction | None |
| Vision model training/finetuning/application | Done | `src/cv/chart_classifier.py`, `scripts/finetune_cnn.py` | None |
| Evaluation by metrics/visual inspection | Done | CV pipeline + ablation impact | Add one concise visual-evaluation note |
| Behavior limitations interpreted | In progress | Partial in README | Add dedicated limitations bullets |
| Integration contribution to another block | Done | CV features in Config C | None |

## Execution Order for Remaining Work

1. Fill deployment URL and add screenshots in README.
2. Add explicit scope/assumptions subsection.
3. Add concise EDA findings subsection.
4. Add explicit error analysis subsection.
5. Add explicit NLP comparison and block-specific evaluation table.
6. Add minimal smoke tests for data/features/train/app imports.
7. Run final consistency check (commands, names, links) and tag release candidate.
