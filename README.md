# Cross-Well Evaluation Protocols for FORCE 2020 Lithofacies

CPU-friendly pipeline for **well-held-out** (leakage-restricted) lithofacies evaluation on FORCE 2020: protocols, information-budget baselines (Global / Hybrid / while-drilling / UDA-lite), and figure scripts aligned with the reported evaluation tables.

> **Pipeline r3 (2026-09-29).** Target-well $z$-scores, relative depth, causal prefix stats and similarity means use the **full observable curve** (`data/processed/curves_all.parquet`). Supervised training and Macro-F1 scoring use labelled depths only (`labeled.parquet`). Result rows carry `pipeline_version=r3_fullcurve_stats_20260929` plus git/config/split/dataset hashes; resume refuses rows that do not match the current provenance fingerprint. Fixed per-well metrics use **train-learnable** labels. See `RELEASE_CHECKLIST.md` for the mandatory clean re-run.

> **Naming note (Protocol-2).** The dataframe column `block` and CSV key `protocol2_block_holdout` are historical. Values are **NPD quadrants** (leading token of `QUADRANT/BLOCK-WELL`, e.g. `15` in `15/9-23`), not NPD blocks. The paper calls Protocol-2 **quadrant hold-out**.

## Environment

```bash
cd cross_well_dg
py -m pip install -r requirements.txt
```

Recommended check:

```bash
py -c "import lightgbm, sklearn, pandas; print(lightgbm.__version__, sklearn.__version__, pandas.__version__)"
py -m pytest tests/ -q
```

Unit tests use `save=False` (or a temp `SPLITS_DIR`) so they cannot overwrite `data/splits/protocol1_seed*.json`.

## Reproduce main tables / figures (from saved CSVs)

Most manuscript numbers can be verified **without retraining**:

```bash
# Merge per-protocol CSVs → results/summary_all.csv
py scripts/merge_results.py

# Regenerate paper figures (incl. Protocol-2 / graphical abstract)
py scripts/make_paper_figures.py
py scripts/make_graphical_abstract.py

# Closed- vs open-set seed summaries used in the paper
py scripts/revision_closed_openset_stats.py
```

Primary artefacts:

| Artefact | Role |
|----------|------|
| `results/summary_all.csv` | All protocol × method × seed rows |
| `results/summary_protocol1_random_wells_seed{42,43,44,45}.csv` | Protocol-1 per seed |
| `results/summary_protocol2_block_holdout_blk{15,35}_seed42.csv` | Protocol-2 quadrants 15 / 35 |
| `results/followup/` | Paired gaps, per-class F1, LOWO, FORCE penalty |
| `results/paper_figures/`, `results/figures/` | Evaluation figures |

## Full retrain (optional; slow)

```bash
# 1) LAS → parquet (needs FORCE LAS directory beside this repo)
py run_experiment.py prepare

# 2) Smoke test
py run_experiment.py run --max-wells 20 --protocol 1 --seed 42 --methods lgbm_global,lgbm_hybrid,rf_global

# 3) Paper release schedule (single config_hash; includes RF-full / XGB / WellNorm-norel)
py run_experiment.py run --config configs/paper_r3_release.yaml
```

Progress is written **after each method** to `results/summary_<protocol>_*.csv`.

## Methods (current)

| Method key | Role |
|------------|------|
| `lgbm_global` | Inductive Global-LGBM |
| `lgbm_global_eq120k` | Inductive Global-LGBM with ≤120k-row schedule |
| `lgbm_wellnorm_norel` | Test-domain well-wise z only (**no** rel. depth; 2×2 cell) |
| `lgbm_wellnorm` | Test-domain well-wise z + rel. depth |
| `lgbm_hybrid` | Global + well-wise z (+ rel. depth) |
| `lgbm_hybrid_norel` | Global + well-wise z (**no** rel. depth; 2×2 cell) |
| `lgbm_hybrid_while` | Causal prefix z (while-drilling) |
| `lgbm_hybrid_smooth` | Hybrid + score-support depth majority vote |
| `lgbm_hybrid_sim` / `lgbm_hybrid_sim_perwell` | UDA-lite source reweighting (omitted from current main-text scores) |
| `lgbm_hybrid_plus` | Batch Sim + Plus diagnostic (omitted from current main-text scores) |
| `rf_global` | Inductive Global-RF (≤120k-row schedule) |
| `rf_global_full` | Equal-row inductive RF (full Protocol split; in `paper_r3_release.yaml`) |
| `xgb_global` | Equal-row inductive XGBoost (full Protocol split; in `paper_r3_release.yaml`) |
| `mlp_source_only` / `mlp_coral_uda` | MLP / CORAL references |

Also reports a **naive depth-point split** baseline and a **matched leakage study** (`scripts/matched_leakage_study.py`) that diagnoses depth-block optimism versus well/class shift (C*/D share Protocol-1 source wells; D only adds target non-test depths).

Equal-row RF / XGB are part of `paper_r3_release.yaml`. Do **not** chain `configs/paper_equal_budget_p1.yaml` afterward into the same summary CSV.

Parent-well r3 sensitivity and sparse-prefix while stresses:

```bash
py -u scripts/parent_well_sensitivity.py
py -u scripts/while_sparse_prefix_study.py
# → results/followup/protocol1_parent_grouped_global.csv
# → results/followup/while_sparse_prefix_seed42.csv
```

## Matched leakage study

```bash
py scripts/matched_leakage_study.py
# → results/followup/matched_leakage_study.csv
```

Settings: (A) random depth-point; (B) contiguous within-well depth blocks; (C) Protocol-1 well hold-out; (C★) same test rows as D without leak; (D) fixed Protocol-1 test wells with other depths from those wells in train.

## Repository layout

| Path | Tracked? | Notes |
|------|----------|--------|
| `src/`, `scripts/`, `configs/` | yes | Pipeline code |
| `data/splits/` | yes | Author-selected exploratory well splits |
| `data/processed/*.csv`, `dataset_meta.json`, `*.sha256` | yes | Small summaries + parquet digests |
| `data/processed/*.parquet` | **no** (local / Zenodo) | Rebuild with `prepare`; hashed in CSVs |
| FORCE LAS folder (outside repo) | **no** | Cite Zenodo; place beside this repo as configured in `src/constants.py` |
| `results/summary_*.csv`, `details_*.json`, `followup/` | yes | Table + worst-well verification without retrain |
| `RELEASE_MANIFEST.json`, `RELEASE_CHECKLIST.md` | yes | Immutable release packaging |
| `paper/gse/` | **no** | Manuscript kept local only (not on GitHub) |

## License

Code and evaluation artefacts: MIT (see `LICENSE`).  
FORCE 2020 logs/labels: original Zenodo / competition terms.

## Citation

Lithofacies data: FORCE 2020 Machine Learning competition  
Bormann et al. 2020 — https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition  
Zenodo: https://doi.org/10.5281/zenodo.4351156

Software archive: public GitHub repository at https://github.com/yangy0907/cross_well_dg (commit `4f0a0ac`, branch `main`); Zenodo-tagged release planned at camera-ready.  
See `RELEASE_CHECKLIST.md` for tagging a clean tree and depositing on Zenodo at camera-ready. Metadata stub: `.zenodo.json`.

Primary Protocol-1 multi-seed means use the locked **20-seed list (18 closed / 2 open)**; historical closed-set seeds 42/43/45 are SI / acceptance-record contrast only.
