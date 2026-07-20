# Leakage and Evaluation Audit

| Risk reviewed | Status | Evidence and effect |
| --- | --- | --- |
| Forward-return target definition | Passed in feature code | `close.shift(-horizon)` is used and final undefined targets are dropped. |
| Split-boundary label overlap | Fixed and rerun for A/B | Revised splits purge the horizon. A/B predictions were regenerated; C remains legacy. |
| Reporting-period endpoint | Fixed and rerun for A/B | Revised code applies `TEST_END`; A/B reporting rows are saved separately. |
| Same-date panel rows in CV | Fixed and rerun for A/B | `PurgedDateTimeSeriesSplit` groups dates and embargoes labels; A/B used it. |
| Stacking internal K-fold | Limitation removed from future ablations | The legacy helper remains for reference but is no longer called by `run_ablation`. |
| Market indicators | Passed by code inspection | Rolling/EMA functions use current and prior data. |
| Chart future candles | Passed by code inspection | Chart generator slices through the chart date only. |
| CNN final held-out evaluation | Unresolved | Temporal CNN train/validation exists, but no audited full rerun ties it to final downstream metrics. |
| PCA preprocessing | Conditional limitation | Code fits on train-period observations when sufficient; an all-row fallback must not be used for final reporting. |
| NLP session alignment | Unresolved | UTC calendar-date normalisation does not resolve after-hours, weekends, or holidays. |
| Analyst features | Invalid for historical reporting | Current aggregate consensus/targets can be assigned to old dates. Exclude configuration D until point-in-time data is used. |
| Test reuse | Limitation | Commit history shows iterative artefact/UI/result development. Do not claim the reporting set was evaluated exactly once. |

The current legacy metrics should not be described as clean held-out performance. The A/B rerun saves prediction rows and input hashes, but immutable raw-data snapshots, session-aware news alignment, and a C rerun remain requirements for a complete multimodal claim.
