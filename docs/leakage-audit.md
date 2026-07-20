# Leakage and Evaluation Audit

| Risk reviewed | Status | Evidence and effect |
| --- | --- | --- |
| Forward-return target definition | Passed in feature code | `close.shift(-horizon)` is used and final undefined targets are dropped. |
| Split-boundary label overlap | Fixed in code; metrics stale | Previous splits used dates through boundaries. Revised splits purge the horizon; rerun required. |
| Reporting-period endpoint | Fixed in code; metrics stale | Previous split selected every date after `TEST_START`. Revised code applies `TEST_END`. |
| Same-date panel rows in CV | Fixed in code; metrics stale | Previous `TimeSeriesSplit` operated on rows. `PurgedDateTimeSeriesSplit` groups dates and embargoes labels. |
| Stacking internal K-fold | Limitation removed from future ablations | The legacy helper remains for reference but is no longer called by `run_ablation`. |
| Market indicators | Passed by code inspection | Rolling/EMA functions use current and prior data. |
| Chart future candles | Passed by code inspection | Chart generator slices through the chart date only. |
| CNN final held-out evaluation | Unresolved | Temporal CNN train/validation exists, but no audited full rerun ties it to final downstream metrics. |
| PCA preprocessing | Conditional limitation | Code fits on train-period observations when sufficient; an all-row fallback must not be used for final reporting. |
| NLP session alignment | Unresolved | UTC calendar-date normalisation does not resolve after-hours, weekends, or holidays. |
| Analyst features | Invalid for historical reporting | Current aggregate consensus/targets can be assigned to old dates. Exclude configuration D until point-in-time data is used. |
| Test reuse | Limitation | Commit history shows iterative artefact/UI/result development. Do not claim the reporting set was evaluated exactly once. |

The current legacy metrics should not be described as clean held-out performance. A reproducible rerun with immutable input snapshots and saved prediction rows is the remaining requirement.
