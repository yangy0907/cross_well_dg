"""Sync Protocol-1 multi-seed tables from summary CSVs into paper CSV artefacts.

Primary multi-seed means use closed-set seeds PROTOCOL1_CLOSED_SEEDS only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import PROTOCOL1_CLOSED_SEEDS, RESULTS_DIR
from src.train import PIPELINE_VERSION

OUT = RESULTS_DIR / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)
FU = RESULTS_DIR / "followup"
FU.mkdir(parents=True, exist_ok=True)

KEY = [
    "lgbm_global",
    "lgbm_global_eq120k",
    "rf_global",
    "rf_global_full",
    "xgb_global",
    "lgbm_wellnorm",
    "lgbm_wellnorm_norel",
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


def load_p1(*, require_current_pipeline: bool = False) -> pd.DataFrame:
    files = sorted(RESULTS_DIR.glob("summary_protocol1_random_wells_seed*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if require_current_pipeline:
        df = df[df["pipeline_version"].astype(str) == PIPELINE_VERSION].copy()
    else:
        # Prefer current pipeline; fall back to latest available r3_* rows for plotting.
        cur = df[df["pipeline_version"].astype(str) == PIPELINE_VERSION]
        if not cur.empty:
            df = cur.copy()
        else:
            df = df[df["pipeline_version"].astype(str).str.startswith("r3_")].copy()
    return df


def main() -> None:
    df_all = load_p1(require_current_pipeline=False)
    if df_all.empty:
        print("No Protocol-1 rows yet")
        return
    df = df_all[df_all["seed"].isin(PROTOCOL1_CLOSED_SEEDS)].copy()
    if df.empty:
        print("No closed-set Protocol-1 rows")
        return
    if (df["seed"] == 44).any():
        raise SystemExit("BUG: closed-set filter retained seed 44")

    rows = []
    for m in KEY:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        rows.append(
            {
                "method": m,
                "macro_mean": sub["macro_f1"].mean(),
                "macro_std": sub["macro_f1"].std(ddof=1) if len(sub) > 1 else 0.0,
                # Primary tail = fixed taxonomy; legacy true kept as sensitivity.
                "worst_mean": (
                    float(sub["worst_well_f1_fixed"].mean())
                    if "worst_well_f1_fixed" in sub.columns
                    and sub["worst_well_f1_fixed"].notna().any()
                    else float(sub["worst_well_f1"].mean())
                ),
                "worst_std": (
                    float(sub["worst_well_f1_fixed"].std(ddof=1))
                    if "worst_well_f1_fixed" in sub.columns
                    and sub["worst_well_f1_fixed"].notna().any()
                    and len(sub) > 1
                    else (float(sub["worst_well_f1"].std(ddof=1)) if len(sub) > 1 else 0.0)
                ),
                "worst_true_mean": float(sub["worst_well_f1"].mean()),
                "worst_true_std": (
                    float(sub["worst_well_f1"].std(ddof=1)) if len(sub) > 1 else 0.0
                ),
                "worst_fixed_mean": (
                    float(sub["worst_well_f1_fixed"].mean())
                    if "worst_well_f1_fixed" in sub.columns
                    else float("nan")
                ),
                "w10_mean": sub["worst10pct_well_f1"].mean(),
                "w10_fixed_mean": (
                    float(sub["worst10pct_well_f1_fixed"].mean())
                    if "worst10pct_well_f1_fixed" in sub.columns
                    else float("nan")
                ),
                "n_seeds": int(sub["seed"].nunique()),
                "seeds": ",".join(str(int(s)) for s in sorted(sub["seed"].unique())),
                "pipeline_version": str(sub["pipeline_version"].iloc[0]),
            }
        )
    multi = pd.DataFrame(rows)
    multi.to_csv(OUT / "table_protocol1_multiseed.csv", index=False)
    gse = ROOT / "paper" / "gse" / "figs"
    gse.mkdir(parents=True, exist_ok=True)
    multi.to_csv(gse / "table_protocol1_multiseed.csv", index=False)

    # Per-seed Hybrid / WellNorm / Global (all seeds, including open-set 44)
    pivot_rows = []
    for seed in sorted(df_all["seed"].unique()):
        s = df_all[df_all["seed"] == seed].set_index("method")
        row = {"seed": int(seed)}
        for m, key in [
            ("lgbm_global", "global"),
            ("lgbm_wellnorm", "wellnorm"),
            ("lgbm_wellnorm_norel", "wellnorm_norel"),
            ("lgbm_hybrid", "hybrid"),
            ("lgbm_hybrid_norel", "hybrid_norel"),
            ("lgbm_hybrid_while", "while"),
            ("lgbm_hybrid_sim_perwell", "sim_pw"),
            ("lgbm_global_eq120k", "global_eq"),
            ("rf_global", "rf"),
            ("rf_global_full", "rf_full"),
            ("xgb_global", "xgb"),
        ]:
            if m in s.index:
                row[f"{key}_macro"] = float(s.loc[m, "macro_f1"])
                row[f"{key}_worst"] = float(s.loc[m, "worst_well_f1"])
                if "worst_well_f1_fixed" in s.columns and pd.notna(
                    s.loc[m].get("worst_well_f1_fixed")
                ):
                    row[f"{key}_worst_fixed"] = float(s.loc[m, "worst_well_f1_fixed"])
                row[f"{key}_w10"] = float(s.loc[m, "worst10pct_well_f1"])
                if "worst10pct_well_f1_fixed" in s.columns and pd.notna(
                    s.loc[m].get("worst10pct_well_f1_fixed")
                ):
                    row[f"{key}_w10_fixed"] = float(s.loc[m, "worst10pct_well_f1_fixed"])
                if "known_macro_f1" in s.columns and pd.notna(s.loc[m].get("known_macro_f1")):
                    try:
                        row[f"{key}_known_macro"] = float(s.loc[m, "known_macro_f1"])
                    except (TypeError, ValueError):
                        pass
        if "hybrid_norel_macro" in row and "wellnorm_macro" in row:
            row["d_macro_hn_wn"] = row["hybrid_norel_macro"] - row["wellnorm_macro"]
            row["d_worst_hn_wn"] = row["hybrid_norel_worst"] - row["wellnorm_worst"]
        if "hybrid_macro" in row and "wellnorm_macro" in row:
            row["d_macro_h_wn"] = row["hybrid_macro"] - row["wellnorm_macro"]
            row["d_worst_h_wn"] = row["hybrid_worst"] - row["wellnorm_worst"]
            row["d_w10_h_wn"] = row["hybrid_w10"] - row["wellnorm_w10"]
        if "hybrid_macro" in row and "global_macro" in row:
            row["d_macro_h_g"] = row["hybrid_macro"] - row["global_macro"]
            row["d_worst_h_g"] = row["hybrid_worst"] - row["global_worst"]
        pivot_rows.append(row)
    per = pd.DataFrame(pivot_rows)
    per.to_csv(FU / "protocol1_hybrid_wellnorm_perseed.csv", index=False)
    print(multi.to_string(index=False))
    print(per.to_string(index=False))
    (FU / "protocol1_r3_sync_meta.json").write_text(
        json.dumps(
            {
                "pipeline_version_code": PIPELINE_VERSION,
                "pipeline_version_in_csv": sorted(
                    df_all["pipeline_version"].astype(str).unique().tolist()
                ),
                "closed_seeds": list(PROTOCOL1_CLOSED_SEEDS),
                "n_closed_rows": int(len(df)),
                "n_all_rows": int(len(df_all)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
