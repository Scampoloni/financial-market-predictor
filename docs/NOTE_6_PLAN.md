# Note-6 Upgrade Plan (updated 2026-04-30)

Current grade: **5.6 / 6** (second evaluator assessment)
Goal: close the remaining 0.4 points via two concrete technical additions.

---

## Status — completed items

All items below are done and verified. Do not redo them.

| Item | Evidence |
|---|---|
| CV isolated evaluation (04_cv_pipeline.ipynb Sektion 7) | Confusion matrix, Accuracy/F1, visual inspection — executed and saved |
| Model selection protocol (val_f1_macro, not test) | Documented in README + ablation_results.json (`selection_metric: val_f1_macro`) + code line 398 in train_ml.py |
| Error analysis by sector and VIX regime (06_evaluation_ablation.ipynb Sektion 8) | Sector-F1 bar chart, VIX regime F1, False UP / False DOWN breakdown |
| Reproducibility — smoke-test dataset | scripts/build_smoke_dataset.py + README section with steps and expected outputs |
| FinBERT vs VADER section (03_nlp_pipeline.ipynb Sektion 9) | Executed — produced "Days with news: 0" due to smoke data sparsity (see open item 1 below) |
| Ethics & Limitations section | README section with Data Bias, Self-fulfilling Prophecy, Market Access Inequality |
| Ticker selection rationale | README section with criteria table and explicit exclusions |
| Notebook version headers | All notebooks (01–06) have v1/v2 version note at top |

---

## Open items (what separates 5.6 from 6.0)

The evaluator named exactly two gaps that cost the remaining points:

### 1. NLP comparison has no usable numbers (highest impact, ~0.2–0.3 points)

**Problem:** Notebook 03 Sektion 9 (FinBERT vs VADER Return-Alignment) outputs
"Days with news: 0" because the smoke-dataset has all headlines on a single scraping
date with no overlap to daily market data. The comparison cell runs but produces
no meaningful correlation or confusion matrix.

**Fix:** Add a small curated evaluation dataset directly in the notebook — no external
data needed. Approach:

1. In `notebooks/03_nlp_pipeline.ipynb` Sektion 9, before the return-alignment cell,
   add a new cell that creates a minimal hardcoded evaluation set:
   - 15–20 real financial headlines (can be taken from existing `data/raw/news/`)
   - Each headline manually labeled as positive/negative/neutral sentiment based on
     the known market reaction (e.g. "Apple beats earnings estimates" → positive,
     "Fed raises rates by 75bps" → negative)
   - Run both FinBERT and VADER on these headlines
   - Compute: agreement rate, Cohen's Kappa, which model is closer to the manual label
   - Show a small confusion matrix for each model vs. manual labels

2. Add a markdown result cell interpreting which model is more reliable for
   financial headlines and why (FinBERT expected to win on domain-specific text).

**File:** `notebooks/03_nlp_pipeline.ipynb`
**Effort:** ~2–3 hours

---

### 2. Block contribution deltas are not statistically validated (~0.1–0.15 points)

**Problem:** NLP adds −0.004 F1, CV adds +0.0016 F1 — these deltas are so small they
could be noise. The evaluator explicitly asked for a significance discussion.

**Fix:** Add a Bootstrap confidence interval analysis to `notebooks/06_evaluation_ablation.ipynb`
as a new Sektion 9:

1. For each Config (A, B, C), bootstrap resample the test set predictions (e.g. 1000
   iterations, sample with replacement from the test rows).
2. Compute F1 macro for each bootstrap sample → distribution of F1 per config.
3. Compute bootstrap CI for the delta B−A and C−B.
4. Plot: overlapping distributions or CI bar chart for all three configs.
5. Interpret: if CI for delta B−A excludes zero → NLP signal is statistically real.
   If not → honest statement that the improvement is within noise at this sample size.

This does not require retraining — only the saved test-set predictions are needed.
If `ablation_results.json` does not store per-row predictions, load the model and
run `model.predict(X_test)` directly in the notebook.

**File:** `notebooks/06_evaluation_ablation.ipynb`
**Effort:** ~2 hours

---

## Priority order

1. **NLP curated evaluation** (item 1) — higher impact, clearly named by evaluator
2. **Bootstrap CI** (item 2) — medium impact, ~1 paragraph of analysis

---

## Verification checklist

- [ ] 03_nlp_pipeline.ipynb Sektion 9 shows actual numbers (agreement rate, kappa, confusion matrices)
- [ ] 06_evaluation_ablation.ipynb Sektion 9 shows bootstrap CI plots with interpretation
- [ ] Both notebooks re-executed with kernel `financial-market-predictor` and outputs saved
- [ ] AE_REQUIREMENTS_MATRIX.md updated with evidence links for both new sections

---

## What NOT to change

- Do not retrain models — all training artifacts are final.
- Do not change the smoke-dataset setup — it works.
- Do not touch README sections that are already done (ethics, ticker rationale, etc.).
- Do not add new features to the Streamlit app.
