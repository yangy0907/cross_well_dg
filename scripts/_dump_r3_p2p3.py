"""Dump Protocol-2/3 r3 means for paper sync."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.train import PIPELINE_VERSION

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def load_glob(pattern: str, exclude_substr: str | None = None) -> pd.DataFrame:
    files = sorted(RES.glob(pattern))
    if exclude_substr:
        files = [f for f in files if exclude_substr not in f.name]
    dfs = []
    for f in files:
        d = pd.read_csv(f)
        if "pipeline_version" not in d.columns:
            continue
        d = d[d["pipeline_version"].astype(str) == PIPELINE_VERSION]
        if len(d):
            dfs.append(d)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


KEY = [
    "lgbm_global",
    "lgbm_global_eq120k",
    "rf_global",
    "lgbm_wellnorm",
    "lgbm_hybrid",
    "lgbm_hybrid_norel",
    "lgbm_hybrid_while",
    "lgbm_hybrid_smooth",
    "lgbm_hybrid_sim",
    "lgbm_hybrid_sim_perwell",
    "lgbm_hybrid_plus",
    "mlp_source_only",
    "mlp_coral_uda",
]


def main() -> None:
    print("PIPELINE", PIPELINE_VERSION)
    p2 = load_glob("summary_protocol2_block_holdout_blk*_seed42.csv")
    print("P2 rows", len(p2))
    if len(p2):
        bcol = "block" if "block" in p2.columns else None
        for blk in sorted(p2[bcol].unique()) if bcol else []:
            print(f"=== P2 quadrant {blk} ===")
            sub = p2[p2[bcol] == blk]
            for m in KEY:
                r = sub[sub["method"] == m]
                if len(r):
                    print(
                        f"  {m}: macro={r['macro_f1'].iloc[0]:.6f} "
                        f"worst={r['worst_well_f1'].iloc[0]:.6f}"
                    )

    p3 = load_glob("summary_protocol3_lowo_well*_seed42.csv", exclude_substr="spaced")
    print("P3 primary rows", len(p3))
    if len(p3):
        wcol = "test_well" if "test_well" in p3.columns else None
        if wcol is None:
            for c in p3.columns:
                if "well" in c.lower() and c != "n_test_wells":
                    wcol = c
                    break
        print("wcol", wcol, "n_wells", p3[wcol].nunique() if wcol else None)
        if wcol:
            print("wells", sorted(p3[wcol].astype(str).unique()))
        g = p3.groupby("method")[["macro_f1", "worst_well_f1"]].agg(["mean", "std"])
        print("=== P3 primary mean±std ===")
        for m in KEY:
            if m not in g.index:
                continue
            mm, ms = g.loc[m, ("macro_f1", "mean")], g.loc[m, ("macro_f1", "std")]
            wm, ws = g.loc[m, ("worst_well_f1", "mean")], g.loc[m, ("worst_well_f1", "std")]
            print(f"  {m}: macro={mm:.6f}±{ms:.6f} worst={wm:.6f}±{ws:.6f}")
        # per-well for hard cases
        if wcol:
            print("=== P3 per-well Hybrid/WellNorm/Global ===")
            for w in sorted(p3[wcol].astype(str).unique()):
                sub = p3[p3[wcol].astype(str) == w]
                parts = []
                for m in ["lgbm_wellnorm", "lgbm_hybrid", "lgbm_hybrid_smooth", "lgbm_global"]:
                    r = sub[sub["method"] == m]
                    if len(r):
                        parts.append(f"{m}={r['macro_f1'].iloc[0]:.3f}")
                print(f"  {w}: " + ", ".join(parts))

    p3a = load_glob("summary_protocol3_lowo_spaced_alt_well*_seed42.csv")
    print("P3 alt rows", len(p3a))
    if len(p3a):
        g = p3a.groupby("method")[["macro_f1"]].agg(["mean", "std"])
        print("=== P3 alt mean±std ===")
        for m in KEY:
            if m not in g.index:
                continue
            mm, ms = g.loc[m, ("macro_f1", "mean")], g.loc[m, ("macro_f1", "std")]
            print(f"  {m}: macro={mm:.6f}±{ms:.6f}")
    else:
        print("P3 alt: no r3 rows (paper_r3_p3_alt not run)")


if __name__ == "__main__":
    main()
