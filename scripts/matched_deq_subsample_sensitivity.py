"""Repeat D_eq equal-n leak contrast over subsample seeds (fixed test blocks).

Keeps Protocol-1 seed-42 source wells and contiguous test-block placement fixed
(seed=42), varies only ``subsample_seed`` when downsampling D to |C*|.

Also records C* once and full D once as anchors.

Outputs: results/followup/matched_deq_subsample_sensitivity.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import RESULTS_DIR, WELL_COL
from src.prepare_data import load_curves_all, load_labeled
from src.provenance import collect_run_provenance
from src.splits import apply_split, fixed_test_wells_depth_leak_split, protocol1_random_wells
from src.train import PIPELINE_VERSION, run_method_on_split, set_curve_context

OUT = RESULTS_DIR / "followup" / "matched_deq_subsample_sensitivity.csv"
DETAIL = RESULTS_DIR / "followup" / "matched_deq_subsample_sensitivity.json"
SUB_SEEDS = list(range(42, 52))  # 10 draws; 42 matches the published D_eq draw


def main(block_seed: int = 42) -> None:
    # Archive prior dump if present.
    if OUT.exists():
        from datetime import datetime, timezone
        import shutil

        arch = RESULTS_DIR / "followup" / "archive_pre_matched_r3"
        arch.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        shutil.copy2(OUT, arch / f"matched_deq_subsample_sensitivity_{stamp}.csv")
        if DETAIL.exists():
            shutil.copy2(DETAIL, arch / f"matched_deq_subsample_sensitivity_{stamp}.json")

    df = load_labeled()
    set_curve_context(load_curves_all())
    config_path = ROOT / "configs" / "paper_r3_full.yaml"
    stub = collect_run_provenance(config_path=config_path)
    print(
        f"[deq-subsample] pipeline={PIPELINE_VERSION} "
        f"config_hash={stub.get('config_hash','')[:12]}..."
    )
    split = protocol1_random_wells(df, seed=block_seed, save=True)
    parts_c = apply_split(df, split)

    parts_d = fixed_test_wells_depth_leak_split(
        df,
        test_wells=split["test_wells"],
        train_wells=split["train_wells"],
        val_wells=split["val_wells"],
        seed=block_seed,
        test_frac_within=0.5,
    )
    te_keys = set(
        zip(parts_d["test"][WELL_COL].astype(str), parts_d["test"]["depth"].astype(float))
    )
    c_test = parts_c["test"].copy()
    c_test["_k"] = list(zip(c_test[WELL_COL].astype(str), c_test["depth"].astype(float)))
    c_test = c_test[c_test["_k"].isin(te_keys)].drop(columns="_k")
    parts_cm = {
        "train": parts_c["train"],
        "val": parts_c["val"],
        "test": c_test.reset_index(drop=True),
    }
    n_star = len(parts_cm["train"])

    stub_hash = stub.get("config_hash", "")

    rows: list[dict] = []
    print("[C*] once")
    res_cm = run_method_on_split(parts_cm, "lgbm_global", seed=block_seed)
    rows.append(
        {
            "setting": "C_star",
            "block_seed": block_seed,
            "subsample_seed": None,
            "macro_f1": res_cm["overall"]["macro_f1"],
            "worst_well_f1": res_cm["well_summary"]["worst_well_f1"],
            "n_train": len(parts_cm["train"]),
            "n_test": len(parts_cm["test"]),
            "pipeline_version": PIPELINE_VERSION,
            "config_hash": stub_hash,
            "delta_vs_c_star": 0.0,
        }
    )
    c_star_f1 = res_cm["overall"]["macro_f1"]

    print("[D] once (full leak rows)")
    res_d = run_method_on_split(parts_d, "lgbm_global", seed=block_seed)
    rows.append(
        {
            "setting": "D_full",
            "block_seed": block_seed,
            "subsample_seed": None,
            "macro_f1": res_d["overall"]["macro_f1"],
            "worst_well_f1": res_d["well_summary"]["worst_well_f1"],
            "n_train": len(parts_d["train"]),
            "n_test": len(parts_d["test"]),
            "pipeline_version": PIPELINE_VERSION,
            "config_hash": stub_hash,
            "delta_vs_c_star": res_d["overall"]["macro_f1"] - c_star_f1,
        }
    )

    for ss in SUB_SEEDS:
        print(f"[D_eq] subsample_seed={ss}")
        parts = fixed_test_wells_depth_leak_split(
            df,
            test_wells=split["test_wells"],
            train_wells=split["train_wells"],
            val_wells=split["val_wells"],
            seed=block_seed,
            test_frac_within=0.5,
            match_n_train=n_star,
            subsample_seed=ss,
        )
        assert len(parts["test"]) == len(parts_cm["test"])
        assert len(parts["train"]) == n_star
        res = run_method_on_split(parts, "lgbm_global", seed=block_seed)
        rows.append(
            {
                "setting": "D_eq",
                "block_seed": block_seed,
                "subsample_seed": ss,
                "macro_f1": res["overall"]["macro_f1"],
                "worst_well_f1": res["well_summary"]["worst_well_f1"],
                "n_train": len(parts["train"]),
                "n_test": len(parts["test"]),
                "pipeline_version": PIPELINE_VERSION,
                "config_hash": stub_hash,
                "delta_vs_c_star": res["overall"]["macro_f1"] - c_star_f1,
            }
        )

    out_df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, index=False)
    deq = out_df[out_df["setting"] == "D_eq"]
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "config_hash": stub_hash,
        "c_star_macro_f1": float(c_star_f1),
        "d_full_macro_f1": float(res_d["overall"]["macro_f1"]),
        "d_eq_n": int(len(deq)),
        "d_eq_delta_mean": float(deq["delta_vs_c_star"].mean()),
        "d_eq_delta_std": float(deq["delta_vs_c_star"].std(ddof=1)),
        "d_eq_delta_min": float(deq["delta_vs_c_star"].min()),
        "d_eq_delta_max": float(deq["delta_vs_c_star"].max()),
        "d_eq_macro_mean": float(deq["macro_f1"].mean()),
        "subsample_seeds": SUB_SEEDS,
        "interpretation": (
            "Equal-budget effect of replacing ~10% source rows with same-target-well "
            "labelled depths under fixed test blocks; not a pure neighbour-leak estimate."
        ),
    }
    DETAIL.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out_df.to_string(index=False))
    print("summary", summary)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
