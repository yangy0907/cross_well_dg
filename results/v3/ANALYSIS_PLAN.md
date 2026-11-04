# Relative-depth validation plan — locked before new model scores

The user authorized the next step after the v2 pilot. This phase tests the
observed dependence of relative-depth benefit on available log channels.

## Fixed training budget

- Parent-grouped seeds: 20261103, 20261104, 20261105. Use every seed; no score-based replacement.
- Two curve sets: GR/RHOB/NPHI/DTC and the original sixteen curves.
- Two methods: Hybrid-norel and Hybrid, using raw missingness masks.
- Total: twelve new-split fits, plus one original-v2 reproduction gate fit.
- Keep full labelled training/validation/test cohorts and the original model settings. No hyperparameter tuning.
- Classify closed versus open sets from source/test label coverage before fitting. Keep any open case and report it separately.
- Save split definitions in this phase's result tree. Never overwrite legacy data/splits or v2 results.
- Save trained boosters, fitted transformation state and row-level predictions so inference stresses can be repeated without retraining.

Implementation amendment before any new-partition model score: the initial gate
fit reached model persistence, where LightGBM's native Windows file API rejected
the Unicode workspace path. Preserve that failed execution manifest and source
under `relative_depth_validation/`. Use Python UTF-8 model-text writing and
readback, with a new frozen execution under `relative_depth_validation_run2/`.
The gate is repeated; total planned fit attempts are now fourteen (thirteen
successful fits plus one failed-persistence gate). Seed/method/boundary choices
are unchanged. No failed new-split results are excluded.

## Main contrast

For each new closed split compute

`(Hybrid - Hybrid-norel)_core4 - (Hybrid - Hybrid-norel)_full16`.

Report each paired effect, the mean and split SD, along with fixed-taxonomy
well tails. New partitions still reuse the FORCE corpus and overlap in source
and test wells; they are not an independent external dataset or population
replication. Existing v2 results are exploratory discovery evidence; these new
partitions are a follow-up consistency check on the same corpus.

## Boundary sensitivity, without additional training

For each fitted Hybrid, crop 10% or 20% of the observed depth span at the top,
bottom, or both. Score only rows inside the retained interval. Recompute only
the relative-depth coordinate from the cropped boundaries; keep all other
features, source fitting and whole-well normalization moments frozen.

For every crop compare three predictions on exactly the same retained rows:
baseline Hybrid coordinates, cropped-boundary Hybrid coordinates, and
Hybrid-norel. The primary stress change is cropped-boundary Hybrid minus
baseline Hybrid on this shared support. Record retained row/well/class counts.
Do not compare a cropped score directly against a full-cohort score and call
that a boundary effect.

This isolates coordinate sensitivity. It is deliberately not a physical
incomplete-logging simulation, since well-normalization statistics remain
full-curve values. It cannot prove geological transfer or instrument causality.
No stress scenario is selected or tuned using target labels.

## Interpretation rules

- Positive, consistent new paired effects support an input-set-dependent benefit; inconsistent effects must be reported.
- Coordinate sensitivity restricts reliance on relative depth; absence of a drop in these fixed stresses does not prove universal robustness.
- No population p values, no learned triage rule, no external-corpus validation, and no automatic claim of journal acceptance.
- Update the manuscript only after complete results and provenance checks, keeping the delivered v2 candidate intact.
