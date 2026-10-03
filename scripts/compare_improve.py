"""Compare baseline vs improved methods from summary CSVs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
FOCUS = [
    "lgbm_global",
    "lgbm_wellnorm",
    "lgbm_hybrid",
    "lgbm_hybrid_sim",
    "lgbm_hybrid_smooth",
    "lgbm_hybrid_plus",
    "lgbm_groupdro",
    "lgbm_conf",
]


def main() -> None:
    files = sorted(RESULTS.glob("summary_*.csv"))
    files = [f for f in files if f.name != "summary_all.csv" and "archive" not in f.name]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["method"].isin(FOCUS)].copy()

    rows = []
    for (prot, seed), g in df.groupby(["protocol", "seed"], dropna=False):
        if prot == "protocol2_block_holdout":
            for blk, gb in g.groupby("holdout_blocks"):
                gb = gb.drop_duplicates("method", keep="last")
                for _, r in gb.iterrows():
                    rows.append(
                        {
                            "protocol": f"{prot}|blk{blk}",
                            "method": r["method"],
                            "macro_f1": r["macro_f1"],
                            "well_macro_f1_mean": r.get("well_macro_f1_mean"),
                            "worst_well_f1": r.get("worst_well_f1"),
                            "seconds": r.get("seconds"),
                        }
                    )
        else:
            g = g.drop_duplicates("method", keep="last")
            for _, r in g.iterrows():
                rows.append(
                    {
                        "protocol": prot,
                        "method": r["method"],
                        "macro_f1": r["macro_f1"],
                        "well_macro_f1_mean": r.get("well_macro_f1_mean"),
                        "worst_well_f1": r.get("worst_well_f1"),
                        "seconds": r.get("seconds"),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "compare_improve.csv", index=False)

    print("=== Protocol-1 ===")
    p1 = out[out.protocol == "protocol1_random_wells"].sort_values("macro_f1", ascending=False)
    print(p1.to_string(index=False))
    print("\n=== Protocol-2 ===")
    p2 = out[out.protocol.str.startswith("protocol2")].sort_values(["protocol", "macro_f1"], ascending=[True, False])
    print(p2.to_string(index=False))


if __name__ == "__main__":
    main()
