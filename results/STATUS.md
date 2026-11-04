# Experiment status — COMPLETE (seed 42)

## Pipeline
- Data: 118 wells, 1,431,383 labeled rows
- Protocols 1 / 2 / 3 + naive leakage baseline — **done**
- Confidence weight fix: FORCE 1=high → `weight=1/conf`
- Stable LOWO splits + resume

## Main tables
| File | Content |
|------|---------|
| `summary_all.csv` | 37 rows, all protocols |
| `ablation_protocol1.csv` | P1 method ablation |
| `leakage_vs_crosswell.csv` | optimistic vs cross-well |
| `confidence_sensitivity.csv` | conf filtering study |
| `protocol3_lowo_macro_f1.csv` | LOWO pivot |
| `figures/` | paper plots |

## Protocol-1 (random wells)
| method | macro_f1 | worst_well_f1 |
|--------|----------|---------------|
| lgbm_global | **0.396** | 0.122 |
| lgbm_groupdro | 0.351 | 0.100 |
| lgbm_conf | 0.340 | 0.101 |
| lgbm_wellnorm | 0.326 | 0.098 |
| lgbm_full | 0.325 | 0.093 |
| mlp_coral_uda | 0.239 | 0.071 |
| naive_depth_split | 0.785 (leaky) | 0.337 |

## Protocol-2 (block holdout macro_f1)
| method | blk15 | blk35 |
|--------|-------|-------|
| lgbm_wellnorm | **0.327** | 0.201 |
| lgbm_conf | 0.317 | 0.202 |
| lgbm_full | 0.317 | **0.217** |
| lgbm_groupdro | 0.313 | 0.200 |
| lgbm_global | 0.301 | **0.259** |
| mlp_coral_uda | 0.302 | 0.181 |

## Protocol-3 LOWO (6 wells, mean macro_f1)
| method | mean |
|--------|------|
| lgbm_wellnorm | **0.335** |
| lgbm_groupdro | 0.328 |
| lgbm_full | 0.320 |

Wells: 15/9-23, 17/11-1, 29/3-1, 33/9-1, 34/10-16 R, 7/1-1

## Optional later
- Multi-seed (43/44)
- Paper writing / deeper case profiles
