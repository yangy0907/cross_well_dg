# Protocol-1 multi-rep (Round-3/4 stats revision)

Locked in `src/constants.py` + `configs/paper_r3_multirep.yaml`:
- `PROTOCOL1_MULTIREP_SEEDS` (20 seeds: 42–45 + 16 predetermined)
- `PROTOCOL1_CORE_METHODS` (6 LightGBM variants)
- Closed-set a priori: `n_test_unseen_label_kept == 0`
- `config_path` must be the YAML so CSV `config_hash` is populated

## Run / resume

```powershell
cd D:\工作空间\测井\cross_well_dg
# Prefer a clean git tree so git_dirty=False.
py -u scripts/protocol1_multirep_r3.py --split-mode both
# or resume a single mode:
py -u scripts/protocol1_multirep_r3.py --split-mode well_id
py -u scripts/protocol1_multirep_r3.py --split-mode parent_grouped
# Discard dirty prior rows under same pipeline (clean provenance re-run):
# py -u scripts/protocol1_multirep_r3.py --split-mode both --force
```

Crash-safe: completed `(seed, method)` rows in
`protocol1_multirep_{well_id|parent_grouped}_core6.csv` are skipped
when `pipeline_version` matches.

Logs: `protocol1_multirep_r3_run.log` / `.err.log`

## After completion

```powershell
py scripts/aggregate_protocol1_multirep_r3.py
py scripts/check_multirep_r3.py --require-complete
# then sync paper numbers + latexmk; restore "provenance-locked" wording
```

Do **not** re-run embargo / matched / full 16-method / P2 / P3 from this folder.
