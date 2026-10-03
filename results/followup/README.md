# followup/

Diagnostic and revision CSVs used for manuscript tables / supplements.

**Canonical Protocol-1 numbers** come from
`../summary_protocol1_random_wells_seed{42,43,44,45}.csv` with
`pipeline_version=r3_fullcurve_stats_20260926` (config hash
`29fe1b0663bc66457de0da00ee84d5ec4df6483fe78f5fd476b650795a4a3138` /
`configs/paper_r3_full.yaml`).

**Matched leakage Table 4** is from `matched_leakage_study.csv` /
`matched_leakage_study.json` (r3). The ten equal-n `D_eq` subsample
sensitivity is `matched_deq_subsample_sensitivity.csv` (must be r3-tagged).
Dirty-run code snapshot: `matched_leakage_r3_code_snapshot/`.

**Canonical while / LOWO companions (r3):**
- `while_drilling_protocol1_r3_seed42.csv`
- `while_drilling_protocol1_r3_seeds42_45.csv`
- `lowo_per_well_r3_seed42_primary.csv`

Rebuild helpers: `scripts/build_r3_canonical_followup.py`,
`scripts/check_active_results_r3.py` (fails if active CSVs carry non-r3
`pipeline_version`).

Historical r2 dumps (parent/drop-sibling, old while/LOWO names, spaced-alt
LOWO, naive depth summary, etc.) live under `archive_pre_r3/`.
Sim/Plus / FORCE-penalty quarantines: `archive_historical_sim_plus/`.

Bootstrap CIs in `protocol1_hybrid_wellnorm_bootstrap.json` are **seed-level**
(resampling the four seed deltas only), not hierarchical well/block bootstrap.
