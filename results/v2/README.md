# v2 enhancement work

Candidate v1 is preserved under the workspace candidate_versions directory. Original r3 summaries and manuscript files are not replaced by these experiments.

## Agreed scope

The user selected existing-result analysis plus a small four-curve pilot, not a complete new twenty-split schedule. The pilot uses fixed seeds 42, 43, 44, 45, three closed and one open-set stress. It includes 16 core4 fits and four full16 corrected-WellNorm fits. Twelve full16 Global/Hybrid-family reference rows are reused with explicit provenance, not presented as new fits. A single additional legacy-WellNorm fit at the new thread count checks that a discovered missing-mask effect is not a thread-setting effect.

## Completed checks

- Original seed-42 full-row Global was retrained using the original locked Python 3.11.5 dependencies. Five headline metrics matched to 1e-10, see reproduction/gate.json.
- Found and reproduced an r3 preprocessing asymmetry. WellNorm replaces missing curves with zero during well-z engineering, then computes its mask from engineered columns; the original missing positions therefore disappear. The opt-in v2 route uses raw observed missingness. Defaults preserve r3 reproduction.
- Twenty legacy pipeline tests plus seven new missingness, feature-equivalence and matched-row tests passed. The new tests show raw-mask correction leaves Global/Hybrid features unchanged but corrects WellNorm masks.
- Existing closed results were averaged across repeated test appearances within well, then sibling wells within parent. No repeated depth or split is treated as an independent population replicate.

## Outputs and interpretation

existing_well_analysis/ contains long paired scores, parent averages, observable descriptors, post-hoc label descriptors, rank associations, and an exact observed-mean-shift decomposition. These are exploratory descriptions with no population p-values or deployment thresholds. Simple descriptors show limited association and must not be advertised as a validated risk rule.

pilot_common_core/ contains the locked config/source/data fingerprint and atomic per-fit JSON files. Completion requires COMPLETE.json. Three closed splits provide sensitivity evidence, not a new eighteen-split estimate or a replication of Carvalho's complete-case preprocessing.

No external-corpus validation or interpreter-triage benefit is claimed. Those conditional extensions depend on the evidence from the current bounded phase.

## Completed revision

All 20 pilot fits completed, together with the one same-thread diagnostic. Including the original-source reproduction fit, this phase executed 22 fits; 12 additional full16 reference rows were reused, not retrained. The diagnostic reproduces legacy seed-42 WellNorm at eight threads exactly and isolates a +0.062901 Macro change from restoring raw masks. Across the three closed pilot splits, the new-minus-archived change averages +0.014416 and is positive in only one split; the correction is not a universal accuracy improvement.

The matched closed pilot gives core4/full16 Macro means of 0.296/0.371 (Global), 0.277/0.320 (corrected WellNorm), 0.321/0.401 (Hybrid-norel), and 0.364/0.409 (Hybrid). Relative-depth gains are +0.043 versus +0.008. These results have only three overlapping closed splits and do not replace the eighteen-split historical headline.

The separate revision is in `paper/gse_v2`: main.pdf (29 pages), supplementary.pdf (11 pages), LaTeX sources, figures, and REVISION_V2_CN.md. Original paper/gse, historical multirep outputs, saved splits and processed inputs were checked against 427 candidate-manifest entries with no mismatches; see revision_verification.json. Candidate ZIP SHA-256 also remains unchanged. Both PDFs were compiled and all pages rendered for visual review; new figures, tables and the first page were inspected at higher resolution.
