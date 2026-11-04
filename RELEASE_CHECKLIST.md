# Release checklist (Major Revision) — **mandatory before acceptance**

Do **not** treat mixed-hash / `git_dirty=True` CSVs as a release. Recording inconsistency in `RELEASE_MANIFEST.json` does **not** replace eliminating it.

## Git dirty semantics (do not conflate)

| Field | Meaning |
|---|---|
| CSV `git_dirty` / manifest `git.source_dirty` | **source_git_dirty_at_run_start** — ignores `results/`, processed/raw data, logs, latex aux (same exclude set as `src.provenance._GIT_DIRTY_EXCLUDES`) |
| Manifest `git.artifact_dirty` (legacy alias `git.dirty`) | **artifact_tree_dirty_at_manifest_creation** — full `git status --porcelain`, including newly written result CSVs/PDFs |

A release may show `CSV git_dirty=False` while `manifest.git.dirty=True` if only outputs changed after the frozen source snapshot. Before tagging: commit everything, then re-run `build_release_manifest.py` so both `source_dirty` and `artifact_dirty` are false.

## Critical orchestration rules

1. **One canonical schedule only:** `configs/paper_r3_release.yaml`  
   It already includes WellNorm-norel and equal-row RF-full / XGB.  
   **Do not** chain `paper_equal_budget_p1.yaml` or `paper_r3_wellnorm_norel.yaml` into the same `summary_*.csv` — resume drops mismatched `config_hash` rows and would wipe other methods.
2. **Git provenance is frozen once** at runner start (source tree only; `results/` ignored). The lock file `results/.run_experiment.lock` is gitignored.
3. **Archive → commit → run** (never archive after committing “clean” if archive deletes tracked CSVs without a follow-up commit).

## Required sequence

1. **Archive stale active results** (moves `summary_*.csv` / `details_*.json` aside):
   ```bash
   py -3 scripts/archive_stale_results_for_rerun.py
   ```
2. **Commit** that archive/deletion plus all code/config/paper edits so the source tree is clean (`git status` clean aside from ignored paths).
3. **Full re-run from the single release config** (mandatory):
   ```bash
   py -3 run_experiment.py run --config configs/paper_r3_release.yaml
   ```
   Resume only keeps rows that match **current** `pipeline_version`, `config_hash`, `split_hash`, parquet/meta hashes, `n_train`/`n_test`, and non-empty `feature_param_hash`.
4. **Diagnostics / follow-ups** (same clean commit; frozen provenance; **all** must write current `PIPELINE_VERSION`):
   ```bash
   py -3 scripts/matched_leakage_study.py          # Table 4 — per-setting split_hash required
   py -3 scripts/embargo_fixed_block_study.py     # Table S5 — train/val wells disjoint
   py -3 scripts/parent_well_sensitivity.py       # parent / Table S6 if cited
   py -3 scripts/while_coldstart_study.py         # while-drilling cold-start if cited
   py -3 scripts/while_sparse_prefix_study.py     # Table S7
   ```
5. **Regenerate derived artefacts**:
   ```bash
   py -3 scripts/revision_r3_openset_bootstrap.py
   py -3 scripts/build_acceptance_record.py       # exact seeds 42/43/45; no all-seed fallback
   py -3 scripts/sync_r3_paper_tables.py
   py -3 scripts/make_paper_figures.py
   py -3 scripts/build_release_manifest.py
   ```
6. **Release gate** (must exit 0; covers P1/P2/P3 cartesian completeness, naive companion, Table 4 non-blank `split_hash`, required supp CSV pipeline lock, known W10, closed-set seeds; soft-includes multirep companion):
   ```bash
   py -3 scripts/check_active_results_r3.py
   py -3 scripts/check_multirep_r3.py --require-complete   # hard lock after clean 240-run
   ```
7. Confirm every primary summary row has:
   - `git_dirty=False` (source snapshot at run start)
   - matching `config_hash` / `split_hash` / data hashes
   - populated `worst_well_f1_fixed` / `n_fixed_labels` (`fixed_label_mode=train_learnable`)
   - open-set seed 44: `known_worst10pct_well_f1` populated
   - multirep CSVs: 120 rows × 2 modes, closed/open 18/2, non-empty `config_hash` from `configs/paper_r3_multirep.yaml`

## Protocol-1 multi-rep (co-primary; Round-4)

Separate schedule (do **not** mix into `paper_r3_release.yaml` summaries):

```bash
# After clean commit:
py -u scripts/protocol1_multirep_r3.py --split-mode both
# Crash-safe resume; use --force only to discard prior dirty rows.
py scripts/aggregate_protocol1_multirep_r3.py
py scripts/check_multirep_r3.py --require-complete
```

Config: `configs/paper_r3_multirep.yaml` (20 seeds, core 6 methods, well_id + parent_grouped).
Outputs: `results/multirep/protocol1_multirep_{well_id,parent_grouped}_core6.csv`.

Primary Protocol-1 multi-rep means use **a priori closed seeds** (18/20). Seeds 44 and 127 are open-set stress.
Historical three-seed closed means (42/43/45) remain ablation / acceptance-record contrast only.

8. **Recompile both** `paper/gse/main.tex` and `supplementary.tex` (timestamps must both refresh for the same release).
9. **Tag** immutable release and deposit on Zenodo; put the archive DOI in Data availability.

## Metric note

- **Pooled Macro-F1**: classes present in that split's test `y_true` (may include open-set unk).
- **Fixed per-well Macro-F1 / worst-well_fixed**: **train-learnable** class indices only (shared across wells within a split; excludes never-predicted unk).
- **Hybrid+Smooth**: score-support smoothing diagnostic (labelled test depths only), not full-curve deployment post-processing.
- **CORAL-MLP**: `uda_target_mode=transductive_score_support` (labelled-score positions with labels hidden).
- **C_common Macro**: mean F1 over the intersection of train-learnable class names across closed analysis splits (missing class → 0); optional sensitivity restricts to classes with positive test support in that split.
