"""Build canonical r3 followup CSVs and archive leftover r2 dumps.

- Archives top-level followup CSVs whose pipeline_version is not r3.
- Writes clearly named while-drilling / LOWO r3 canonical tables from
  results/summary_*.csv (already r3).
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLLOWUP = ROOT / "results" / "followup"
ARCH = FOLLOWUP / "archive_pre_r3"
CANON_WHILE42 = FOLLOWUP / "while_drilling_protocol1_r3_seed42.csv"
CANON_WHILE_MULTI = FOLLOWUP / "while_drilling_protocol1_r3_seeds42_45.csv"
CANON_LOWO = FOLLOWUP / "lowo_per_well_r3_seed42_primary.csv"

import sys

sys.path.insert(0, str(ROOT))
from src.train import PIPELINE_VERSION

R3 = PIPELINE_VERSION
PRIMARY_LOWO = ["15/9-23", "17/11-1", "29/3-1", "33/9-1", "34/10-16 R", "7/1-1"]
MAIN_METHODS = {
    "lgbm_global",
    "lgbm_global_eq120k",
    "lgbm_wellnorm",
    "lgbm_hybrid",
    "lgbm_hybrid_norel",
    "lgbm_hybrid_smooth",
    "lgbm_hybrid_while",
    "rf_global",
    "mlp_source_only",
}


def _versions_in_csv(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "pipeline_version" not in rows[0]:
        return set()
    return {str(r.get("pipeline_version") or "") for r in rows}


def archive_non_r3_followup_csvs() -> list[str]:
    ARCH.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for path in sorted(FOLLOWUP.glob("*.csv")):
        vers = _versions_in_csv(path)
        if not vers:
            continue
        if any(v.startswith("r3_") for v in vers if v):
            # keep if only r3, or mixed with empty
            if all((not v) or v.startswith("r3_") for v in vers):
                continue
        # any non-r3 version present -> archive
        if any(v and not v.startswith("r3_") for v in vers):
            dest = ARCH / path.name
            if dest.exists():
                dest = ARCH / f"historical_r2_{path.name}"
            shutil.move(str(path), str(dest))
            moved.append(f"{path.name} -> {dest.relative_to(ROOT)}")
    return moved


def write_while_canonical() -> None:
    rows42: list[dict] = []
    multi: list[dict] = []
    for seed in (42, 43, 44, 45):
        src = ROOT / "results" / f"summary_protocol1_random_wells_seed{seed}.csv"
        with src.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("pipeline_version") != R3:
                    continue
                method = r.get("method", "")
                if method not in MAIN_METHODS and "while" not in method:
                    continue
                # keep reported main/appendix methods; prefer while + global/hybrid family
                if seed == 42 and method in {
                    "lgbm_global",
                    "lgbm_wellnorm",
                    "lgbm_hybrid",
                    "lgbm_hybrid_while",
                }:
                    rows42.append(
                        {
                            "method": method,
                            "macro_f1": r.get("macro_f1"),
                            "well_macro_f1_mean": r.get("well_macro_f1_mean"),
                            "worst_well_f1": r.get("worst_well_f1"),
                            "feature_mode": r.get("feature_mode"),
                            "use_relative_depth": r.get("use_relative_depth"),
                            "well_z_mode": r.get("well_z_mode"),
                            "seconds": r.get("seconds"),
                            "pipeline_version": r.get("pipeline_version"),
                        }
                    )
                if method in {
                    "lgbm_global",
                    "lgbm_hybrid",
                    "lgbm_hybrid_while",
                    "lgbm_wellnorm",
                }:
                    multi.append(
                        {
                            "protocol": r.get("protocol", "protocol1_random_wells"),
                            "seed": seed,
                            "method": method,
                            "macro_f1": r.get("macro_f1"),
                            "worst_well_f1": r.get("worst_well_f1"),
                            "well_macro_f1_mean": r.get("well_macro_f1_mean"),
                            "pipeline_version": r.get("pipeline_version"),
                        }
                    )
    with CANON_WHILE42.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows42[0].keys()))
        w.writeheader()
        w.writerows(rows42)
    with CANON_WHILE_MULTI.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(multi[0].keys()))
        w.writeheader()
        w.writerows(multi)


def write_lowo_canonical() -> None:
    rows: list[dict] = []
    for well in PRIMARY_LOWO:
        safe = (
            well.replace("/", "_")
            .replace(" ", "")
            .replace("~", "")
            .replace("R", "R")
        )
        # filenames use 34_10-16R without space
        fname = f"summary_protocol3_lowo_well{safe.replace(' ', '')}_seed42.csv"
        # special case 34/10-16 R
        if well == "34/10-16 R":
            fname = "summary_protocol3_lowo_well34_10-16R_seed42.csv"
        elif well == "15/9-23":
            fname = "summary_protocol3_lowo_well15_9-23_seed42.csv"
        elif well == "17/11-1":
            fname = "summary_protocol3_lowo_well17_11-1_seed42.csv"
        elif well == "29/3-1":
            fname = "summary_protocol3_lowo_well29_3-1_seed42.csv"
        elif well == "33/9-1":
            fname = "summary_protocol3_lowo_well33_9-1_seed42.csv"
        elif well == "7/1-1":
            fname = "summary_protocol3_lowo_well7_1-1_seed42.csv"
        src = ROOT / "results" / fname
        with src.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("pipeline_version") != R3:
                    continue
                method = r.get("method", "")
                if method not in MAIN_METHODS:
                    continue
                if method in {"lgbm_hybrid_sim", "lgbm_hybrid_plus", "lgbm_hybrid_sim_perwell"}:
                    continue
                rows.append(
                    {
                        "protocol": r.get("protocol", "protocol3_lowo"),
                        "seed": 42,
                        "target_well": well,
                        "method": method,
                        "macro_f1": r.get("macro_f1"),
                        "worst_well_f1": r.get("worst_well_f1"),
                        "accuracy": r.get("accuracy"),
                        "pipeline_version": r.get("pipeline_version"),
                    }
                )
    with CANON_LOWO.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    moved = archive_non_r3_followup_csvs()
    write_while_canonical()
    write_lowo_canonical()
    print("archived:", *moved, sep="\n  " if moved else " (none)")
    print("wrote", CANON_WHILE42)
    print("wrote", CANON_WHILE_MULTI)
    print("wrote", CANON_LOWO)


if __name__ == "__main__":
    main()
