"""Export worst-well case study tables from Protocol-1 details JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    path = RESULTS / "details_protocol1_random_wells_seed42.json"
    if not path.exists():
        print(f"Missing {path}")
        return
    details = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in details:
        method = item.get("method")
        for w in item.get("well_table", []):
            rows.append({"method": method, **w})
    if not rows:
        print("No well_table entries yet.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "protocol1_wellwise_all_methods.csv", index=False)

    # pick reference method for worst wells
    ref = "lgbm_full" if "lgbm_full" in set(df["method"]) else df["method"].iloc[0]
    ref_df = df[df["method"] == ref].sort_values("macro_f1")
    worst3 = ref_df.head(3)
    best3 = ref_df.tail(3)
    worst3.to_csv(RESULTS / "protocol1_worst3_wells.csv", index=False)
    best3.to_csv(RESULTS / "protocol1_best3_wells.csv", index=False)

    # compare methods on those worst wells
    targets = set(worst3["well_id"])
    cmp_ = df[df["well_id"].isin(targets)].pivot_table(
        index="well_id", columns="method", values="macro_f1", aggfunc="last"
    )
    cmp_.to_csv(RESULTS / "protocol1_worst_wells_method_compare.csv")
    print("Worst-3 wells:")
    print(worst3.to_string(index=False))
    print("\nMethod compare on worst wells:")
    print(cmp_.to_string())


if __name__ == "__main__":
    main()
