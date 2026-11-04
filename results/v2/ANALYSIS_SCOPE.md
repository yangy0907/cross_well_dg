# V2 revision scope and interpretation

The user selected existing-result analysis plus a small four-curve experiment.
This is a bounded exploratory extension, not a preregistered confirmatory study.

- Keep the candidate-v1 snapshot, ZIP, raw data, split files and historical results intact.
- Reproduce the original Global seed-42 reference before fitting new models.
- Use the locked historical splits 42/43/45 as three closed pilot splits; report open seed 44 separately.
- Fit four methods on GR/RHOB/NPHI/DTC (16 fits), keeping the complete labelled cohort and missing observations.
- Fit corrected full16 WellNorm on the same four splits (4 fits).
- Reuse 12 full16 Global/Hybrid-norel/Hybrid references with original provenance and feature-equivalence tests. Never label these as newly fitted.
- One additional legacy-mask WellNorm seed-42 diagnostic at eight threads distinguishes a missingness-mask change from the thread-count difference. The numerical extension has 20 pilot fits plus 1 diagnostic fit. Including the earlier original-source Global reproduction gate, this phase executes 22 fits in total.
- Preserve the legacy pipeline by default. Opt into original-observation masks for new results.
- Average repeated test appearances per well, then average siblings per parent, for exploratory well-descriptor associations. Shared models and training wells still induce dependence.
- Use present-class per-well F1 for effect heterogeneity, alongside the separate fixed-taxonomy coverage-aware screens. Do not interpret a low fixed score as a calibrated failure probability.
- Treat observed class counts, class proportions and the class-conditioned mean-shift identity as target-label-dependent post-hoc analyses. They cannot be deployed as predictors.
- Report overlapping-split standard deviations and paired changes descriptively. Do not infer a population effect from three pilot seeds or manufacture significant mechanisms from weak descriptor correlations.
- Do not add an external corpus, a learned triage rule, another algorithm search, or a full twenty-split new training schedule in this phase.

The mask audit changes the interpretation of the historical WellNorm comparisons:
its original missingness indicators were computed after missing values had become
zero. Thus historical Hybrid-versus-WellNorm gaps cannot isolate the value of
preserving global log levels. Global/Hybrid reference features are algebraically
unchanged by the correction, which is checked at fitting and inference.

Sources, configurations, split definitions and input parquet hashes are recorded
in the pilot execution manifest. `COMPLETE.json` is written only after all twenty
pilot fits succeed. New manuscript sources and PDFs are built in `paper/gse_v2`.
