"""Diagnostic protocol contrasts for depth leakage vs well/class shift.

Runs Global-LGBM under pre-specified settings (seed 42 by default):
  A. naive random depth-point (all wells; optimistic / leaky)
  B. contiguous within-well depth-block hold-out (same wells; protocol sensitivity)
  C. Protocol-1 well hold-out (deployment; well + class shift)
  D. same source train/val wells as C; fixed test depths; other target depths in train
  D_eq. same as D but training rows downsampled to match C* size
  C*. same train/val as C and same test depths as D; no target-well leak

Writes pipeline_version / config_hash / per-setting split_hash into CSV+JSON.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import RESULTS_DIR, WELL_COL
from src.prepare_data import load_curves_all, load_labeled
from src.provenance import collect_run_provenance, freeze_git_provenance
from src.splits import (
    apply_split,
    contiguous_depth_block_split,
    fixed_test_wells_depth_leak_split,
    naive_depth_split,
    protocol1_random_wells,
)
from src.train import PIPELINE_VERSION, run_method_on_split, set_curve_context, set_run_meta

OUT = RESULTS_DIR / "followup" / "matched_leakage_study.csv"
DETAIL = RESULTS_DIR / "followup" / "matched_leakage_study.json"
CONFIG_PATH = ROOT / "configs" / "paper_r3_release.yaml"

NOTES = {
    "A_naive_depth": "random depth points; wells shared; strong adjacent leakage",
    "B_depth_blocks": "contiguous depth blocks; wells shared; protocol sensitivity vs A",
    "C_well_holdout": "Protocol-1; no test-well depths in train; well+class shift",
    "D_fixed_wells_depth_leak": (
        "same source train/val wells as C; same test depths; target non-test depths in train"
    ),
    "D_eq_n_train": (
        "same as D but training rows downsampled to |C*|; equal-n leak contrast"
    ),
    "C_matched_same_test_as_D": (
        "same source train/val as C; same test depths as D; no test-well leak"
    ),
}

_SPLIT_KEYS = (
    "protocol",
    "seed",
    "train_wells",
    "val_wells",
    "test_wells",
    "holdout_blocks",
    "target_well",
    "selection",
    "test_frac",
    "val_frac",
    "row_index_hash",
    "match_n_train",
    "subsample_seed",
)


def _identity_from_parts(protocol: str, seed: int, parts: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Build a provenance split identity; prefer fields already on ``parts``."""
    ident: dict[str, Any] = {"protocol": protocol, "seed": int(seed)}
    for k in _SPLIT_KEYS:
        if k in parts and k not in ("protocol", "seed"):
            ident[k] = parts[k]
    for k, v in extra.items():
        if v is not None:
            ident[k] = v
    if "train_wells" not in ident and "train" in parts:
        ident["train_wells"] = sorted(parts["train"][WELL_COL].astype(str).unique().tolist())
    if "val_wells" not in ident and "val" in parts:
        ident["val_wells"] = sorted(parts["val"][WELL_COL].astype(str).unique().tolist())
    if "test_wells" not in ident and "test" in parts:
        ident["test_wells"] = sorted(parts["test"][WELL_COL].astype(str).unique().tolist())
    if "row_index_hash" not in ident and all(k in parts for k in ("train", "val", "test")):
        h = hashlib.sha256()
        for split_name in ("train", "val", "test"):
            frame = parts[split_name]
            wells = frame[WELL_COL].astype(str).tolist()
            depths = frame["depth"].astype(float).tolist() if "depth" in frame.columns else []
            h.update(split_name.encode())
            for w, d in zip(wells, depths):
                h.update(f"{w}:{d};".encode())
        ident["row_index_hash"] = h.hexdigest()
    return ident


def _prov(split_ident: dict[str, Any], run_utc: str) -> dict[str, Any]:
    prov = collect_run_provenance(split=split_ident, config_path=CONFIG_PATH)
    prov["run_utc"] = run_utc
    if not str(prov.get("split_hash", "")).strip():
        raise RuntimeError(f"empty split_hash for protocol={split_ident.get('protocol')}")
    return prov


def _row(tag: str, seed: int, parts: dict, res: dict, prov: dict) -> dict:
    row = {
        "setting": tag,
        "seed": seed,
        "method": "lgbm_global",
        "macro_f1": res["overall"]["macro_f1"],
        "accuracy": res["overall"]["accuracy"],
        "worst_well_f1": res["well_summary"]["worst_well_f1"],
        "well_macro_f1_mean": res["well_summary"]["well_macro_f1_mean"],
        "n_train": len(parts["train"]),
        "n_val": len(parts["val"]),
        "n_test": len(parts["test"]),
        "n_train_wells": int(parts["train"][WELL_COL].nunique()),
        "n_test_wells": int(parts["test"][WELL_COL].nunique()),
        "seconds": res["seconds"],
        "note": NOTES.get(tag, ""),
        "pipeline_version": PIPELINE_VERSION,
        "config_path": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
        "config_hash": prov.get("config_hash", ""),
        "split_hash": prov.get("split_hash", ""),
        "git_commit": prov.get("git_commit", ""),
        "git_dirty": prov.get("git_dirty", ""),
        "run_utc": prov.get("run_utc", ""),
    }
    return row


def _well_set(parts: dict, split: str) -> set[str]:
    return set(parts[split][WELL_COL].astype(str).unique())


def main(seed: int = 42) -> None:
    # Archive prior dump if present (keep pre-rerun evidence).
    if OUT.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        arch = RESULTS_DIR / "followup" / "archive_pre_matched_r3"
        arch.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT, arch / f"matched_leakage_study_{stamp}.csv")
        if DETAIL.exists():
            shutil.copy2(DETAIL, arch / f"matched_leakage_study_{stamp}.json")

    df = load_labeled()
    set_curve_context(load_curves_all())
    freeze_git_provenance(ignore_outputs=True)
    run_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stub = collect_run_provenance(config_path=CONFIG_PATH)
    stub["pipeline_version"] = PIPELINE_VERSION
    stub["run_utc"] = run_utc

    rows: list[dict] = []
    details: dict = {"provenance": stub, "pipeline_version": PIPELINE_VERSION}

    print(f"[matched-leakage] pipeline={PIPELINE_VERSION} config_hash={stub.get('config_hash','')[:12]}...")

    print("[A] naive random depth-point")
    parts_a = naive_depth_split(df, seed=seed)
    ident_a = _identity_from_parts("naive_depth_split", seed, parts_a)
    set_run_meta(split=ident_a, config_path=CONFIG_PATH)
    res_a = run_method_on_split(parts_a, "lgbm_global", seed=seed)
    stub_a = _prov(ident_a, run_utc)
    rows.append(_row("A_naive_depth", seed, parts_a, res_a, stub_a))
    details["A"] = {
        "macro_f1": res_a["overall"]["macro_f1"],
        "seconds": res_a["seconds"],
        "split_hash": stub_a["split_hash"],
    }

    print("[B] contiguous within-well depth blocks")
    parts_b = contiguous_depth_block_split(df, seed=seed)
    ident_b = _identity_from_parts("contiguous_depth_block_split", seed, parts_b)
    set_run_meta(split=ident_b, config_path=CONFIG_PATH)
    res_b = run_method_on_split(parts_b, "lgbm_global", seed=seed)
    stub_b = _prov(ident_b, run_utc)
    rows.append(_row("B_depth_blocks", seed, parts_b, res_b, stub_b))
    details["B"] = {
        "macro_f1": res_b["overall"]["macro_f1"],
        "seconds": res_b["seconds"],
        "split_hash": stub_b["split_hash"],
    }

    print("[C] Protocol-1 well hold-out")
    split = protocol1_random_wells(df, seed=seed, save=True)
    parts_c = apply_split(df, split)
    ident_c = _identity_from_parts("protocol1_random_wells", seed, {**parts_c, **split})
    set_run_meta(split=ident_c, config_path=CONFIG_PATH)
    res_c = run_method_on_split(parts_c, "lgbm_global", seed=seed)
    stub_c = _prov(ident_c, run_utc)
    rows.append(_row("C_well_holdout", seed, parts_c, res_c, stub_c))
    details["C"] = {
        "macro_f1": res_c["overall"]["macro_f1"],
        "seconds": res_c["seconds"],
        "split_hash": stub_c["split_hash"],
        "test_wells": split["test_wells"],
        "train_wells": split["train_wells"],
        "val_wells": split["val_wells"],
    }

    print("[D] matched source wells + target depth leak")
    parts_d = fixed_test_wells_depth_leak_split(
        df,
        test_wells=split["test_wells"],
        train_wells=split["train_wells"],
        val_wells=split["val_wells"],
        seed=seed,
        test_frac_within=0.5,
    )
    ident_d = _identity_from_parts("fixed_test_wells_depth_leak_split", seed, parts_d)
    set_run_meta(split=ident_d, config_path=CONFIG_PATH)
    res_d = run_method_on_split(parts_d, "lgbm_global", seed=seed)
    stub_d = _prov(ident_d, run_utc)
    rows.append(_row("D_fixed_wells_depth_leak", seed, parts_d, res_d, stub_d))
    details["D"] = {
        "macro_f1": res_d["overall"]["macro_f1"],
        "seconds": res_d["seconds"],
        "split_hash": stub_d["split_hash"],
    }

    print("[C*] same train/val wells and same test depths as D, no leak")
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
    ident_cm = _identity_from_parts(
        "c_matched_same_test_as_d",
        seed,
        parts_cm,
        train_wells=split["train_wells"],
        val_wells=split["val_wells"],
        test_wells=split["test_wells"],
    )
    set_run_meta(split=ident_cm, config_path=CONFIG_PATH)
    res_cm = run_method_on_split(parts_cm, "lgbm_global", seed=seed)
    stub_cm = _prov(ident_cm, run_utc)
    rows.append(_row("C_matched_same_test_as_D", seed, parts_cm, res_cm, stub_cm))
    details["C_star"] = {
        "macro_f1": res_cm["overall"]["macro_f1"],
        "seconds": res_cm["seconds"],
        "split_hash": stub_cm["split_hash"],
    }

    print("[D_eq] D with n_train matched to C*")
    parts_deq = fixed_test_wells_depth_leak_split(
        df,
        test_wells=split["test_wells"],
        train_wells=split["train_wells"],
        val_wells=split["val_wells"],
        seed=seed,
        test_frac_within=0.5,
        match_n_train=len(parts_cm["train"]),
        subsample_seed=seed,
    )
    ident_deq = _identity_from_parts("fixed_test_wells_depth_leak_eq", seed, parts_deq)
    set_run_meta(split=ident_deq, config_path=CONFIG_PATH)
    res_deq = run_method_on_split(parts_deq, "lgbm_global", seed=seed)
    stub_deq = _prov(ident_deq, run_utc)
    rows.append(_row("D_eq_n_train", seed, parts_deq, res_deq, stub_deq))
    details["D_eq"] = {
        "macro_f1": res_deq["overall"]["macro_f1"],
        "seconds": res_deq["seconds"],
        "split_hash": stub_deq["split_hash"],
    }

    assert _well_set(parts_cm, "train") == _well_set(parts_c, "train")
    assert _well_set(parts_cm, "val") == _well_set(parts_c, "val")
    source_train = _well_set(parts_cm, "train")
    source_val = _well_set(parts_cm, "val")
    test_wells = set(map(str, split["test_wells"]))
    d_train_source = _well_set(parts_d, "train") - test_wells
    assert d_train_source == source_train, (len(d_train_source), len(source_train))
    assert _well_set(parts_d, "val") == source_val
    assert len(parts_d["test"]) == len(parts_cm["test"]) == len(parts_deq["test"])
    assert len(parts_deq["train"]) == len(parts_cm["train"])
    details["pairing_checks"] = {
        "source_train_wells_equal": True,
        "val_wells_equal": True,
        "n_test_equal": True,
        "n_train_C_star": len(parts_cm["train"]),
        "n_train_D": len(parts_d["train"]),
        "n_train_D_eq": len(parts_deq["train"]),
        "n_leak_rows_in_D": len(parts_d["train"]) - len(parts_cm["train"]),
        "design": (
            "C*/D share Protocol-1 source train/val wells and identical test depths; "
            "D adds non-test depths from test wells; D_eq downsamples to |C*|"
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tab = pd.DataFrame(rows)
    if tab["split_hash"].isna().any() or (tab["split_hash"].astype(str).str.strip() == "").any():
        raise RuntimeError("matched_leakage produced empty split_hash on some rows")
    tab.to_csv(OUT, index=False)
    DETAIL.write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        tab[
            [
                "setting",
                "macro_f1",
                "worst_well_f1",
                "pipeline_version",
                "config_hash",
                "split_hash",
                "run_utc",
                "seconds",
            ]
        ].to_string(index=False)
    )
    print("wrote", OUT)
    print("wrote", DETAIL)


if __name__ == "__main__":
    main()
