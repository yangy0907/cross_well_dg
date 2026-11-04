"""Fixed-test-block embargo study (review Issue 2 / Table S5).

For Protocol-1 seed 42 Global-LGBM: hold out contiguous test blocks, then
allow training from non-test depths of the same wells with an embargo of
0 / 1 / 5 / 10 / 20 m from the test-block boundary.

Training uses **train_wells only** (disjoint from validation). Validation uses
**val_wells only**. Training row count is matched by subsampling within the
training pool. Multiple subsample seeds.

Output: results/followup/embargo_fixed_block_study.csv

Memory-safe defaults: LGBM n_jobs=2, BLAS threads capped, gc between fits.
"""
from __future__ import annotations

import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Cap thread thrash before numpy/lightgbm import (avoids prior OOM with n_jobs=-1).
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("LGBM_NUM_THREADS", "2")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import DEPTH_COL, LABEL_COL, RESULTS_DIR, WELL_COL
from src.models import LGBMConfig
from src.prepare_data import load_labeled
from src.provenance import collect_run_provenance, freeze_git_provenance
from src.splits import protocol1_random_wells
from src.train import PIPELINE_VERSION, run_method_on_split, set_curve_context

OUT = RESULTS_DIR / "followup"
OUT.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = ROOT / "configs" / "paper_r3_release.yaml"

EMBARGO_M = [0.0, 1.0, 5.0, 10.0, 20.0]
SUBSAMPLE_SEEDS = list(range(10))
LGBM_N_JOBS = int(os.environ.get("EMBARGO_LGBM_N_JOBS", "2"))


def _pick_test_blocks(
    df: pd.DataFrame, test_wells: list[str], seed: int = 42, block_frac: float = 0.25
) -> pd.DataFrame:
    """Within each test well, hold out a contiguous depth block (~block_frac)."""
    rng = np.random.default_rng(seed)
    parts = []
    for w in test_wells:
        g = df[df[WELL_COL] == w].sort_values(DEPTH_COL)
        n = len(g)
        if n < 20:
            continue
        bw = max(5, int(round(n * block_frac)))
        start = int(rng.integers(0, max(1, n - bw)))
        parts.append(g.iloc[start : start + bw])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _embargo_train(
    df: pd.DataFrame,
    test_block: pd.DataFrame,
    train_wells: list[str],
    embargo_m: float,
) -> pd.DataFrame:
    """Train wells fully + same-well (test) non-test depths outside embargo.

    ``train_wells`` must be disjoint from validation wells. Test-well non-block
    depths may enter training subject to the embargo distance.
    """
    src_mask = df[WELL_COL].isin(train_wells)
    te_wells = test_block[WELL_COL].unique().tolist()
    same_well_mask = np.zeros(len(df), dtype=bool)
    for w in te_wells:
        tb = test_block[test_block[WELL_COL] == w]
        if tb.empty:
            continue
        d0, d1 = float(tb[DEPTH_COL].min()), float(tb[DEPTH_COL].max())
        lo, hi = d0 - embargo_m, d1 + embargo_m
        wmask = df[WELL_COL].to_numpy() == w
        depth = df[DEPTH_COL].to_numpy()
        same_well_mask |= wmask & ((depth < lo) | (depth > hi))
    return df.loc[src_mask | same_well_mask]


def _class_prior_tv(y_a: np.ndarray, y_b: np.ndarray) -> float:
    ca = pd.Series(y_a).value_counts(normalize=True)
    cb = pd.Series(y_b).value_counts(normalize=True)
    keys = sorted(set(ca.index) | set(cb.index))
    pa = np.array([ca.get(k, 0.0) for k in keys])
    pb = np.array([cb.get(k, 0.0) for k in keys])
    return float(0.5 * np.abs(pa - pb).sum())


def main() -> None:
    # Keep LightGBM from opening one thread per logical core (prior OOM).
    # Dataclass defaults live in __dataclass_fields__, not only the class attr.
    _field = LGBMConfig.__dataclass_fields__["n_jobs"]
    _orig_default = _field.default
    object.__setattr__(_field, "default", LGBM_N_JOBS)
    LGBMConfig.n_jobs = LGBM_N_JOBS  # type: ignore[misc]
    print(f"embargo memory-safe: LGBMConfig.n_jobs={LGBM_N_JOBS}", flush=True)

    df = load_labeled()
    # Global-LGBM does not need full-curve context; skip curves_all to save ~RAM.
    set_curve_context(None)
    freeze_git_provenance(ignore_outputs=True)
    run_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    base_prov = collect_run_provenance(config_path=CONFIG_PATH)
    base_prov["run_utc"] = run_utc

    split = protocol1_random_wells(df, seed=42)
    train_wells = list(split["train_wells"])
    val_wells = list(split["val_wells"])
    test_wells = list(split["test_wells"])
    if set(train_wells) & set(val_wells):
        raise RuntimeError("train_wells overlap val_wells in Protocol-1 split")

    test_block = _pick_test_blocks(df, test_wells, seed=42, block_frac=0.25)
    if test_block.empty:
        raise RuntimeError("empty test block")

    # Reference training size: train wells only (independent validation).
    ref_train = df[df[WELL_COL].isin(train_wells)]
    n_ref = len(ref_train)
    val_df = df[df[WELL_COL].isin(val_wells)].copy()
    val_well_set = set(map(str, val_wells))

    # Resume from checkpoint if present (same pipeline + train/val disjoint design).
    study_csv = OUT / "embargo_fixed_block_study.csv"
    rows: list[dict] = []
    done: set[tuple[float, int]] = set()
    if study_csv.exists():
        prev = pd.read_csv(study_csv)
        if (
            not prev.empty
            and "pipeline_version" in prev.columns
            and (prev["pipeline_version"].astype(str) == PIPELINE_VERSION).all()
            and (
                "train_val_disjoint" not in prev.columns
                or (prev["train_val_disjoint"].astype(str).str.lower().isin({"true", "1", "yes"})).all()
            )
        ):
            rows = prev.to_dict(orient="records")
            done = {
                (float(r["embargo_m"]), int(r["subsample_seed"]))
                for r in rows
            }
            print(f"resume: loaded {len(done)} checkpoint rows from {study_csv.name}", flush=True)
        else:
            print("resume: ignoring incompatible checkpoint; starting fresh", flush=True)

    try:
        for emb in EMBARGO_M:
            train_full = _embargo_train(df, test_block, train_wells, emb)
            for ss in SUBSAMPLE_SEEDS:
                if (float(emb), int(ss)) in done:
                    print(f"skip embargo={emb} seed={ss} (checkpoint)", flush=True)
                    continue
                print(f"start embargo={emb} seed={ss} n_ref={n_ref}…", flush=True)
                rng = np.random.default_rng(ss)
                if len(train_full) > n_ref:
                    idx = rng.choice(len(train_full), size=n_ref, replace=False)
                    train_df = train_full.iloc[idx].copy()
                else:
                    train_df = train_full.copy()
                train_in_val = set(train_df[WELL_COL].astype(str).unique()) & val_well_set
                if train_in_val:
                    raise RuntimeError(
                        f"embargo train overlaps validation wells: {sorted(train_in_val)[:5]}"
                    )
                parts = {"train": train_df, "val": val_df, "test": test_block}
                res = run_method_on_split(parts, "lgbm_global", seed=42)
                tv = _class_prior_tv(
                    train_df[LABEL_COL].to_numpy(), test_block[LABEL_COL].to_numpy()
                )
                rows.append(
                    {
                        "embargo_m": emb,
                        "subsample_seed": ss,
                        "n_train": res["n_train"],
                        "n_val": len(val_df),
                        "n_test": res["n_test"],
                        "n_train_wells_ref": len(train_wells),
                        "n_val_wells": len(val_wells),
                        "macro_f1": res["overall"]["macro_f1"],
                        "worst_well_f1": res["well_summary"]["worst_well_f1"],
                        "class_prior_tv": tv,
                        "pipeline_version": PIPELINE_VERSION,
                        "config_path": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
                        "config_hash": base_prov.get("config_hash", ""),
                        "git_commit": base_prov.get("git_commit", ""),
                        "git_dirty": base_prov.get("git_dirty", ""),
                        "hash_labeled_parquet": base_prov.get("hash_labeled_parquet", ""),
                        "run_utc": run_utc,
                        "train_val_disjoint": True,
                    }
                )
                print(
                    f"embargo={emb} seed={ss} macro={res['overall']['macro_f1']:.4f} "
                    f"worst={res['well_summary']['worst_well_f1']:.4f}",
                    flush=True,
                )
                # Incremental checkpoint (resume-safe if interrupted).
                pd.DataFrame(rows).to_csv(study_csv, index=False)
                done.add((float(emb), int(ss)))
                del parts, res, train_df
                gc.collect()
            del train_full
            gc.collect()
    finally:
        object.__setattr__(_field, "default", _orig_default)
        LGBMConfig.n_jobs = _orig_default  # type: ignore[misc]

    out = pd.DataFrame(rows)
    expected = len(EMBARGO_M) * len(SUBSAMPLE_SEEDS)
    if len(out) != expected:
        raise RuntimeError(f"embargo incomplete: {len(out)} rows, expected {expected}")
    out.to_csv(study_csv, index=False)
    summary = (
        out.groupby("embargo_m")
        .agg(
            macro_mean=("macro_f1", "mean"),
            macro_std=("macro_f1", "std"),
            worst_mean=("worst_well_f1", "mean"),
            tv_mean=("class_prior_tv", "mean"),
        )
        .reset_index()
    )
    summary["pipeline_version"] = PIPELINE_VERSION
    summary.to_csv(OUT / "embargo_fixed_block_summary.csv", index=False)
    (OUT / "embargo_fixed_block_study.json").write_text(
        json.dumps(
            {
                "n_ref": n_ref,
                "n_train_wells": len(train_wells),
                "n_val_wells": len(val_wells),
                "train_val_disjoint": True,
                "pipeline_version": PIPELINE_VERSION,
                "config_hash": base_prov.get("config_hash", ""),
                "summary": summary.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
