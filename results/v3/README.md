# Completed V3 follow-up

The executed numerical results are under `relative_depth_validation_run2/`.
The initial `relative_depth_validation/` folder preserves a failed gate's source
and manifest; native Windows model saving rejected the Unicode path. Python
model-text serialization and probability readback resolved that problem.

All twelve new-partition models were saved successfully. The first boundary
postprocessor used exported float32 depths; exact baseline-coordinate checks
rejected that route. `v3_recover_boundary_inference.py` uses original float64
dataframe depths, restores baseline predictions on every test row and completes
all stresses without fitting models again. Recovery provenance is explicit.

Key findings:

- New relative-depth gains average +0.019140 for four curves and +0.008181 for sixteen. The difference of differences averages +0.010959 and is positive in two of three new parent-grouped partitions.
- Across six fixed boundary crops, mean four-curve Macro-F1 changes are all negative, ranging from -0.052207 to -0.011078. Individual effects vary; all results are retained.
- On the same retained support, the four-curve mean Hybrid-minus-Hybrid-norel gain changes from +0.034752 to -0.017151 under the bottom-20% coordinate stress, and from +0.034455 to -0.017752 under the two-sided 20% stress.
- These coordinate sensitivities support an interval-definition limitation. They do not prove geological causality, universal model superiority or physical incomplete-log performance.

`ANALYSIS_PLAN.md` records the fixed scientific design and the technical
amendment. `EXTERNAL_DATA_FEASIBILITY.md` records the second-corpus source audit;
no independent external-corpus result is claimed. `V2_PRESERVATION_MANIFEST.json`
and `relative_depth_validation_run2/revision_verification.json` verify that V2
and original candidate boundaries were preserved.

The separate manuscript package is `paper/gse_v3/`: main.pdf (31 pages),
supplementary.pdf (13 pages), sources, figures, REVISION_V3_CN.md and QA.md.
