"""Aggregate closed-set (seeds 42,43,45) vs open-set seed44 Protocol-1 stats for paper revision."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "results"
OUT = ROOT / "followup"
METHODS = [
    "lgbm_global",
    "lgbm_wellnorm",
    "lgbm_hybrid",
    "lgbm_hybrid_smooth",
    "lgbm_hybrid_sim",
    "lgbm_hybrid_plus",
    "lgbm_hybrid_while",
    "lgbm_hybrid_sim_perwell",
]


def load_seed(seed: int) -> pd.DataFrame:
    p = ROOT / f"summary_protocol1_random_wells_seed{seed}.csv"
    df = pd.read_csv(p)
    df["seed"] = seed
    return df


def main() -> None:
    frames = [load_seed(s) for s in (42, 43, 44, 45)]
    all_df = pd.concat(frames, ignore_index=True)
    # keep last row per method/seed
    all_df = all_df.sort_values(["seed", "method"]).groupby(["seed", "method"], as_index=False).tail(1)

    meta = []
    for seed in (42, 43, 44, 45):
        g = all_df[(all_df.seed == seed) & (all_df.method == "lgbm_global")].iloc[0]
        meta.append(
            {
                "seed": seed,
                "n_test": int(g.n_test),
                "n_test_unseen_label_kept": int(g.n_test_unseen_label_kept),
                "unseen_frac": float(g.n_test_unseen_label_kept) / float(g.n_test),
                "n_val_dropped_unseen_label": int(g.n_val_dropped_unseen_label),
                "pipeline_version": g.pipeline_version,
                "regime": "open_set" if g.n_test_unseen_label_kept > 0 else "closed_set",
            }
        )
    meta_df = pd.DataFrame(meta)
    meta_df.to_csv(OUT / "protocol1_seed_label_regime.csv", index=False)
    print(meta_df.to_string(index=False))

    rows = []
    for regime, seeds in [("closed_42_43_45", [42, 43, 45]), ("all_42_45", [42, 43, 44, 45]), ("open_44", [44])]:
        sub = all_df[all_df.seed.isin(seeds)]
        for m in METHODS:
            ms = sub[sub.method == m]
            if ms.empty:
                continue
            g = sub[sub.method == "lgbm_global"].set_index("seed")["macro_f1"]
            mm = ms.set_index("seed")["macro_f1"]
            ww = ms.set_index("seed")["worst_well_f1"]
            gw = sub[sub.method == "lgbm_global"].set_index("seed")["worst_well_f1"]
            common = mm.index.intersection(g.index)
            dmacro = (mm.loc[common] - g.loc[common]).mean() if len(common) else np.nan
            dworst = (ww.loc[common] - gw.loc[common]).mean() if len(common) else np.nan
            worst_wins = int(((ww.loc[common] - gw.loc[common]) > 0).sum()) if m != "lgbm_global" else np.nan
            rows.append(
                {
                    "regime": regime,
                    "method": m,
                    "n_seeds": len(ms),
                    "macro_mean": mm.mean(),
                    "macro_std": mm.std(ddof=1) if len(ms) > 1 else 0.0,
                    "worst_mean": ww.mean(),
                    "worst_std": ww.std(ddof=1) if len(ms) > 1 else 0.0,
                    "delta_macro_vs_global": dmacro if m != "lgbm_global" else 0.0,
                    "delta_worst_vs_global": dworst if m != "lgbm_global" else 0.0,
                    "worst_wins_vs_global": worst_wins,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "protocol1_closed_vs_openset_summary.csv", index=False)
    print(out[out.regime == "closed_42_43_45"].to_string(index=False))

    # per-seed Global/Hybrid for closed
    detail = []
    for seed in (42, 43, 44, 45):
        for m in ("lgbm_global", "lgbm_hybrid", "lgbm_hybrid_while", "lgbm_hybrid_sim_perwell"):
            ms = all_df[(all_df.seed == seed) & (all_df.method == m)]
            if ms.empty:
                continue
            r = ms.iloc[0]
            detail.append(
                {
                    "seed": seed,
                    "method": m,
                    "macro_f1": r.macro_f1,
                    "worst_well_f1": r.worst_well_f1,
                    "worst10pct_well_f1": r.worst10pct_well_f1,
                    "n_test_unseen_label_kept": r.n_test_unseen_label_kept,
                }
            )
    pd.DataFrame(detail).to_csv(OUT / "protocol1_perseed_key_methods.csv", index=False)


if __name__ == "__main__":
    main()
