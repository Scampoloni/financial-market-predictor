# Final Submission Runbook (A-E)

Purpose: Close every remaining requirement with concrete evidence before final hand-in.

## 1) Working Mode

- Use one branch for compliance hardening: `feat/a-e-compliance-implementation`.
- Keep one commit per completed work package.
- After each package: run relevant tests and update evidence tables.

## 2) Completion Criteria

A package is complete only if all of the following are true:
- Changes are implemented in code/docs.
- README is consistent with actual file/module names.
- Evidence location is linked in `docs/AE_REQUIREMENTS_MATRIX.md`.
- Validation command has been run and result recorded.

## 3) Work Packages

### WP-01: Documentation Backbone (B1-B3)

Deliverables:
- Finalize problem/objectives/scope/assumptions/non-goals in README.
- Finalize data source table with type, origin, and size.
- Finalize preprocessing bullets per block (ML/NLP/CV).
- Finalize modeling rationale and iteration summary.

Validation:
- Manual consistency pass: all names/paths/commands exist.

Evidence update:
- `docs/AE_REQUIREMENTS_MATRIX.md`: B1, B2, B3 set to Done with exact references.

### WP-02: Evaluation Hardening (B4, E.1-E.3)

Deliverables:
- Add explicit block-specific evaluation subsection in README.
- Add explicit error analysis subsection in README.
- Add interpretation/limitations for each block.
- Add or reference at least one compact comparison table per selected block.

Validation:
- Run evaluation notebook/script used for final reported numbers.

Evidence update:
- Matrix rows B4, E.1, E.2, E.3 closed.

### WP-03: Deployment Evidence (B5)

Deliverables:
- Public app URL inserted in README.
- Public model/repo URL inserted in README.
- At least 3 screenshots stored in `docs/screenshots/` and linked in README:
  - main prediction flow
  - model analysis/ablation
  - NLP/CV integration evidence
- Train vs inference separation explicitly referenced.

Validation:
- Open URL in browser and verify app loads.

Evidence update:
- Matrix rows for B5 set to Done.

### WP-04: Reproducibility and QA (B6, C)

Deliverables:
- Keep setup and run instructions executable end-to-end.
- Keep smoke tests stable and passing.
- Add short "known limitations" section for grading transparency.

Validation:
- `python -m pytest tests/test_data_pipeline.py tests/test_feature_engineering.py tests/test_ml_pipeline.py tests/test_streamlit.py -q`

Evidence update:
- Matrix row B6 set to Done.
- C-criteria readiness set to High where applicable.

### WP-05: Submission Gate (D)

Deliverables:
- Confirm collaborators added on GitHub: `jasminh`, `bkuehnis`.
- Confirm repository is public/accessible for grading.
- Confirm final README includes all required links.
- Tag final hand-in state (optional but recommended).

Validation:
- Manual GitHub check with screenshots or checklist confirmation.

Evidence update:
- Matrix section D set to Done.

## 4) Required Evidence Inventory

Must exist before submission:
- `README.md` fully aligned with A-E.
- `docs/AE_REQUIREMENTS_MATRIX.md` with final statuses.
- `docs/screenshots/` with at least 3 screenshot files.
- Public deployment URL.
- Public model/repo URL.

## 5) Recommended Commit Sequence

1. `docs: finalize A-E matrix and README compliance structure`
2. `docs: add deployment URLs and screenshot evidence`
3. `test: stabilize smoke tests and reproducibility checks`
4. `chore: submission gate checklist and final consistency pass`

## 6) Final Pre-Submission Checklist

- [ ] A: all general requirements explicitly evidenced
- [ ] B1-B6: all mandatory documentation items present
- [ ] C: clarity, correctness, depth, integration, reproducibility addressed
- [ ] D: collaborators confirmed and GitHub link ready
- [ ] E: combined blocks integrated and block-specific requirements evidenced
- [ ] README commands tested and valid
- [ ] Deployment URL works publicly
- [ ] Screenshots linked and visible in README
