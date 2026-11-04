# Improvement round — final status

**Updated:** 2026-09-22 (P3 resume finished: well34 + well7 hybrids)  
**Status:** Protocol-1 / 2 / 3 hybrid LOWO **complete** (6/6 wells × 4 hybrid methods); paper figures regenerated (fig5/5b/6; fig8 kept).

## What changed
1. **Hybrid features**: global-standardized raw curves + well-norm + rel_depth + masks
2. **Well similarity UDA**: weight source wells by proximity to unlabeled target
3. **Depth smoothing**: within-well majority vote (window=7)

## Protocol-1 (seed 42; vs previous best `lgbm_global` 0.396)

| method | macro_f1 | Δ | worst_well_f1 |
|--------|----------|---|---------------|
| lgbm_hybrid_sim | **0.406** | +0.010 | 0.155 |
| lgbm_hybrid_smooth | 0.404 | +0.008 | **0.174** |
| lgbm_hybrid | 0.401 | +0.005 | 0.170 |
| lgbm_hybrid_plus | 0.400 | +0.004 | 0.171 |
| lgbm_global | 0.396 | — | 0.122 |
| lgbm_wellnorm | 0.326 | — | 0.098 |

### Multi-seed (42 + 43) — see `paper_figures/fig6` / `table_protocol1_multiseed.csv`
- `lgbm_hybrid_plus`: 0.400 (s42) / **0.421** (s43)
- `lgbm_hybrid`: 0.401 / 0.417
- `lgbm_hybrid_sim`: 0.406 / 0.377 (seed-sensitive)
- `lgbm_global`: 0.396 / 0.381

## Protocol-2
- blk15: **hybrid_sim 0.391** (+0.065 vs wellnorm 0.327)
- blk35: **hybrid_plus 0.297** (+0.038 vs global 0.259)

## Protocol-3 LOWO (6 wells × 4 hybrid methods — complete)

Hybrid counts (hybrid / sim / smooth / plus): **well15,17,29,33,34,7 → 4/4 each**.

Mean Macro-F1 across 6 target wells:

| method | mean macro_f1 | vs wellnorm |
|--------|---------------|-------------|
| lgbm_wellnorm | 0.335 | — |
| lgbm_hybrid | 0.396 | +0.061 |
| lgbm_hybrid_sim | 0.397 | +0.062 |
| lgbm_hybrid_plus | 0.396 | +0.061 |
| **lgbm_hybrid_smooth** | **0.410** | **+0.075** |

Per-well hybrid_smooth Macro-F1: 15/9-23 **0.600**, 29/3-1 **0.507**, 34/10-16R **0.502**, 33/9-1 0.325, 7/1-1 0.266, 17/11-1 0.262.

## Paper figures
Directory: `results/paper_figures/` (300 dpi)

| File | Notes |
|------|-------|
| fig0 / 0b / 0c | Dataset + drift |
| fig1 | Leakage vs cross-well |
| fig2 / 2b | Protocol-1 |
| fig3 | Ablation |
| fig4 | Protocol-2 |
| fig5 / **fig5b** | Protocol-3 mean + per-well |
| **fig6** | Multi-seed robustness |
| fig7 | Worst/best wells |
| fig8 | Case strips (imshow, 300 dpi) |

Regenerate: `py scripts/make_paper_figures.py`  
Case: `py scripts/make_case_figure.py`

## Recommended main method
- Main text: **`lgbm_hybrid_sim`** or **`lgbm_hybrid_plus`** (strong on P1/P2)
- P3 LOWO mean leader: **`lgbm_hybrid_smooth`** (0.410)
