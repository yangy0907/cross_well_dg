"""Parent-well audit + Protocol-1 parent-grouped sensitivity.

1) Audit Protocol-1 seeds 42--45 for parent wellbores spanning train/test.
2) Run Global-LGBM under protocol1_parent_grouped_wells (same seeds).
3) Seed-44 ablation: drop sibling ``34/5-1 S`` from train while keeping
   ``34/5-1 A`` in test (isolates related-wellbore leakage on the published split).

Outputs under results/followup/:
  parent_well_collisions.csv
  protocol1_parent_grouped_global.csv
  protocol1_seed44_drop_sibling_global.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import RESULTS_DIR, WELL_COL
from src.prepare_data import load_labeled
from src.splits import (
    apply_split,
    parent_train_test_collisions,
    parent_well_name,
    protocol1_parent_grouped_wells,
    protocol1_random_wells,
)
from src.train import PIPELINE_VERSION, run_method_on_split

OUT = RESULTS_DIR / "followup"
SEEDS = [42, 43, 44, 45]


def _row(tag: str, seed: int, parts: dict, res: dict, **extra) -> dict:
    return {
        "setting": tag,
        "seed": seed,
        "method": "lgbm_global",
        "pipeline_version": PIPELINE_VERSION,
        "macro_f1": res["overall"]["macro_f1"],
        "accuracy": res["overall"]["accuracy"],
        "worst_well_f1": res["well_summary"]["worst_well_f1"],
        "n_train": len(parts["train"]),
        "n_val": len(parts["val"]),
        "n_test": len(parts["test"]),
        "n_train_wells": int(parts["train"][WELL_COL].nunique()),
        "n_test_wells": int(parts["test"][WELL_COL].nunique()),
        "seconds": res["seconds"],
        **extra,
    }


def main() -> None:
    df = load_labeled()
    OUT.mkdir(parents=True, exist_ok=True)

    # --- audit published wellbore-id splits ---
    audit_rows = []
    for seed in SEEDS:
        split = protocol1_random_wells(df, seed=seed)
        coll = split.get("parent_train_test_collisions") or parent_train_test_collisions(
            split["train_wells"], split["test_wells"]
        )
        if not coll:
            audit_rows.append(
                {
                    "seed": seed,
                    "n_collisions": 0,
                    "parent": "",
                    "train_wellbores": "",
                    "test_wellbores": "",
                }
            )
        for c in coll:
            audit_rows.append(
                {
                    "seed": seed,
                    "n_collisions": len(coll),
                    "parent": c["parent"],
                    "train_wellbores": ";".join(c["train_wellbores"]),
                    "test_wellbores": ";".join(c["test_wellbores"]),
                }
            )
    audit_path = OUT / "parent_well_collisions.csv"
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    print("wrote", audit_path)
    print(pd.DataFrame(audit_rows).to_string(index=False))

    # --- parent-grouped sensitivity ---
    grouped_rows = []
    for seed in SEEDS:
        print(f"[parent-grouped] seed={seed}")
        split = protocol1_parent_grouped_wells(df, seed=seed)
        assert not split["parent_train_test_collisions"]
        parts = apply_split(df, split)
        res = run_method_on_split(parts, "lgbm_global", seed=seed)
        grouped_rows.append(
            _row(
                "protocol1_parent_grouped",
                seed,
                parts,
                res,
                n_parents_train=split["n_parents_train"],
                n_parents_test=split["n_parents_test"],
                n_collisions=0,
            )
        )
    gpath = OUT / "protocol1_parent_grouped_global.csv"
    pd.DataFrame(grouped_rows).to_csv(gpath, index=False)
    print("wrote", gpath)

    # --- seed 44 drop sibling ablation ---
    print("[seed44] drop sibling 34/5-1 S from train")
    split44 = protocol1_random_wells(df, seed=44)
    assert "34/5-1 A" in split44["test_wells"]
    train_drop = [w for w in split44["train_wells"] if str(w) != "34/5-1 S"]
    split_abl = {
        **split44,
        "train_wells": train_drop,
        "protocol": "protocol1_seed44_drop_sibling",
        "dropped_train_wells": ["34/5-1 S"],
        "parent_train_test_collisions": parent_train_test_collisions(
            train_drop, split44["test_wells"]
        ),
    }
    (OUT / "protocol1_seed44_drop_sibling_split.json").write_text(
        json.dumps(split_abl, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    parts_base = apply_split(df, split44)
    parts_abl = apply_split(df, split_abl)
    res_base = run_method_on_split(parts_base, "lgbm_global", seed=44)
    res_abl = run_method_on_split(parts_abl, "lgbm_global", seed=44)
    abl_rows = [
        _row(
            "seed44_published",
            44,
            parts_base,
            res_base,
            note="train includes 34/5-1 S; test has 34/5-1 A",
        ),
        _row(
            "seed44_drop_sibling",
            44,
            parts_abl,
            res_abl,
            note="dropped 34/5-1 S from train; same test wells",
            n_dropped_train_rows=int(
                (parts_base["train"][WELL_COL].astype(str) == "34/5-1 S").sum()
            ),
        ),
    ]
    apath = OUT / "protocol1_seed44_drop_sibling_global.csv"
    pd.DataFrame(abl_rows).to_csv(apath, index=False)
    print("wrote", apath)
    print(pd.DataFrame(abl_rows)[["setting", "macro_f1", "worst_well_f1", "n_train"]].to_string(index=False))


if __name__ == "__main__":
    main()
