# Note-6 Upgrade Plan

Goal: make the project clearly note-6 worthy by closing methodical gaps (CV isolated evaluation, proper model selection) and strengthening analysis, reproducibility, and documentation with explicit evidence.

## 1) Critical Gaps (must fix)

### 1.1 CV block isolated evaluation
Add a dedicated CV evaluation in notebooks/04_cv_pipeline.ipynb:
- Confusion matrix: chart -> UP/DOWN on validation set
- Accuracy/F1 of the CV model alone (not only ablation delta)
- Visual inspection: 2-3 example charts with predicted label vs. actual

Why: CV is currently only evaluated indirectly by its contribution to the full model.

### 1.2 Model selection protocol (test-set isolation)
Fix src/models/train_ml.py:
- Choose models exclusively via validation set (2024H2) or CV
- Use test set (2025) exactly once at the end for final reporting
- Update README and docs to explicitly state the protocol
Notes:
- This is the most time-intensive step. It requires re-training all models.
- After the fix, regenerate ablation_results.json and all saved models so README numbers and artifacts match.

Why: current model selection based on test-F1 risks optimism bias.

### 1.3 Error analysis depth by sector and regime
Extend notebooks/06_evaluation_ablation.ipynb:
- F1 per sector (Tech vs Energy vs Healthcare, etc.)
- Error rates in volatile vs calm regimes (e.g., VIX > 25 vs VIX < 15)
- Most frequent error types (False UP / False DOWN) with interpretation

Why: analysis is currently high-level; deeper failure modes are required for top marks.

## 2) Important Improvements (raise score by ~0.1-0.15 each)

### 2.1 Reproducibility without manual steps
Provide a fully reproducible path:
- Add scripts/download_data.* or a Makefile to fetch data
- Or add DVC integration with remote storage
- Or include a minimal smoke-test dataset (e.g., 5 tickers, 3 months) in repo

Recommendation:
- Use a smoke-test dataset (5 tickers, 3 months). This is lowest effort and does not depend on external infra.
- A download script alone is not enough because historical NewsAPI data is not always available retroactively.

Update README with exact steps and expected outputs.

### 2.2 Explicit FinBERT vs VADER comparison
Extend notebooks/03_nlp_pipeline.ipynb with:
- Correlation between FinBERT and VADER scores
- Which correlates better with returns
- Confusion matrix: sentiment label vs next-day return

Why: the comparison exists implicitly but not as a formal analysis.

### 2.3 Ethics and bias section
Add a dedicated Ethics section in README:
- Data bias (English-only, limited sources)
- Self-fulfilling prophecy risk
- Market access inequality

Why: current disclaimer is present but too short for full bonus credit.

## 3) Nice-to-have (bonus and clarity)

### 3.1 Ticker selection rationale
Moved to Important: this is a low-effort, high-signal improvement.
Add a short rationale in README:
- Criteria: liquidity, sector diversity, market cap, data availability
- Why these 67 tickers, and why some are excluded

### 3.2 Notebook version clarity
Reduce confusion between v1 and v2:
- Add a clear header in each notebook (v1 exploratory vs v2 production)
- Or move v1 notebooks to notebooks/archive and add a README note

## 4) Verification Checklist

1. Run CV evaluation section in notebooks/04_cv_pipeline.ipynb and save outputs
2. Re-run training with validation-based model selection, then final test once
3. Re-run error analysis in notebooks/06_evaluation_ablation.ipynb
4. Re-run NLP comparison in notebooks/03_nlp_pipeline.ipynb
5. Validate reproducibility via new script or DVC path
6. Update README and docs/AE_REQUIREMENTS_MATRIX.md with evidence links

## 5) File Targets

- README.md (methodology, ethics, ticker rationale, reproducibility, evidence links)
- src/models/train_ml.py (model selection protocol)
- src/models/evaluate.py (optional helper for sector/regime analysis)
- src/cv/chart_classifier.py (optional CV evaluation helper)
- notebooks/03_nlp_pipeline.ipynb (FinBERT vs VADER analysis)
- notebooks/04_cv_pipeline.ipynb (CV isolated evaluation)
- notebooks/06_evaluation_ablation.ipynb (error analysis depth)
- docs/AE_REQUIREMENTS_MATRIX.md (evidence updates)
- docs/FINAL_SUBMISSION_RUNBOOK.md (confirm updated checks)
- scripts/ (reproducibility helper if chosen)

## 6) Priority Summary

- Highest impact: CV isolated evaluation + model selection protocol fix
- Medium impact: error analysis depth + reproducibility
- Lower impact: ethics expansion + notebook version clarity

## 7) Decisions Needed

- Reproducibility approach: DVC vs download script vs smoke-test dataset (recommended: smoke-test dataset)
- Notebook versioning: archive v1 vs add explicit headers
- CV evaluation: frozen vs fine-tuned comparison (recommended)

## 8) Additional Notes

- notebooks/04_cv_pipeline.ipynb is currently thin; expect a larger refactor to add a proper CV evaluation section.
- Estimated time: 2-3 days total. Most points (1.1, 1.3, 2.2, 2.3, 3.1, 3.2) fit in one day. 1.2 and 2.1 each need a dedicated block.

## 9) Branch and Commit Strategy

Goal: keep grading evidence traceable and avoid mixing unrelated changes.

Branches:
- main (stable)
- feat/note-6-cv-eval
- feat/note-6-model-selection
- feat/note-6-error-analysis
- feat/note-6-nlp-comparison
- feat/note-6-reproducibility
- feat/note-6-docs-and-ethics

Commit cadence:
- One work package per branch, 1-3 focused commits each.
- Merge in order of dependencies: model selection -> ablation regen -> analysis notebooks -> docs updates.

Suggested commit sequence:
1. cv: isolated CNN evaluation + visual inspection
2. ml: validation-based model selection + retrain + regen artifacts
3. analysis: sector/regime error analysis additions
4. nlp: FinBERT vs VADER comparison section
5. repro: smoke-test dataset + reproducibility docs
6. docs: ethics + ticker rationale + notebook version clarity
7. docs: update AE_REQUIREMENTS_MATRIX evidence
