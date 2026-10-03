"""Companion / release gate for Protocol-1 multi-rep Round-3 CSVs.

Checks presence, core-6 methods, predetermined seed list, a priori regime,
pipeline_version, row completeness (120 per split mode), closed/open 18/2,
non-empty config_hash / split_hash / data hashes, and git_dirty=False when
complete.

Usage::
    py scripts/check_multirep_r3.py
    py scripts/check_multirep_r3.py --require-complete
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import (  # noqa: E402
    PROTOCOL1_CORE_METHODS,
    PROTOCOL1_MULTIREP_SEEDS,
    RESULTS_DIR,
)
from src.train import PIPELINE_VERSION  # noqa: E402

OUT = RESULTS_DIR / "multirep"
MULTIREP_CONFIG = ROOT / "configs" / "paper_r3_multirep.yaml"
N_EXPECTED_ROWS = len(PROTOCOL1_MULTIREP_SEEDS) * len(PROTOCOL1_CORE_METHODS)
N_CLOSED_SEEDS = 18
N_OPEN_SEEDS = 2


def _blank_mask(s: pd.Series) -> pd.Series:
    return s.isna() | s.astype("string").str.strip().isin(["", "nan", "None", "<NA>"])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_nonblank(df: pd.DataFrame, col: str, label: str, errors: list[str]) -> None:
    if col not in df.columns:
        errors.append(f"{label}: missing {col}")
        return
    bad = _blank_mask(df[col])
    if bad.any():
        errors.append(f"{label}: empty/NaN {col} on {int(bad.sum())}/{len(df)} rows")


def _check_csv(
    path: Path,
    split_mode: str,
    errors: list[str],
    warnings: list[str],
    *,
    require_complete: bool,
) -> None:
    if not path.exists():
        warnings.append(f"missing {path.name}")
        return
    df = pd.read_csv(path)
    if df.empty:
        errors.append(f"{path.name}: empty")
        return
    if "pipeline_version" in df.columns:
        bad = df[df["pipeline_version"].astype(str) != PIPELINE_VERSION]
        if len(bad):
            errors.append(
                f"{path.name}: {len(bad)} rows with pipeline_version != {PIPELINE_VERSION}"
            )
    methods = set(df["method"].astype(str))
    missing_m = set(PROTOCOL1_CORE_METHODS) - methods
    if missing_m:
        msg = f"{path.name}: methods not yet present: {sorted(missing_m)}"
        (errors if require_complete else warnings).append(msg)
    seeds = set(int(s) for s in df["seed"].unique())
    expected = set(PROTOCOL1_MULTIREP_SEEDS)
    unexpected = seeds - expected
    if unexpected:
        errors.append(f"{path.name}: unexpected seeds {sorted(unexpected)}")
    regime_col = (
        "a_priori_regime"
        if "a_priori_regime" in df.columns
        else ("label_regime" if "label_regime" in df.columns else None)
    )
    if regime_col is None:
        errors.append(f"{path.name}: missing regime column")
    else:
        if "a_priori_regime" not in df.columns:
            warnings.append(f"{path.name}: missing a_priori_regime (using label_regime)")
        bad_reg = df[~df[regime_col].isin(["closed_set", "open_set"])]
        if len(bad_reg):
            errors.append(f"{path.name}: invalid {regime_col} values")

    pairs = df.drop_duplicates(["seed", "method"])
    n_pairs = len(pairs)
    print(f"  {split_mode}: {n_pairs}/{N_EXPECTED_ROWS} seed×method rows")
    if n_pairs < N_EXPECTED_ROWS:
        msg = f"{path.name}: incomplete {n_pairs}/{N_EXPECTED_ROWS}"
        (errors if require_complete else warnings).append(msg)
    elif n_pairs > N_EXPECTED_ROWS:
        errors.append(f"{path.name}: too many unique pairs {n_pairs}/{N_EXPECTED_ROWS}")

    if require_complete and n_pairs >= N_EXPECTED_ROWS and regime_col:
        # Closed/open seed counts from unique seeds (not rows).
        closed_seeds = set(
            int(s) for s in df.loc[df[regime_col] == "closed_set", "seed"].unique()
        )
        open_seeds = set(
            int(s) for s in df.loc[df[regime_col] == "open_set", "seed"].unique()
        )
        print(
            f"  {split_mode}: closed_seeds={len(closed_seeds)} open_seeds={len(open_seeds)}"
        )
        if len(closed_seeds) != N_CLOSED_SEEDS or len(open_seeds) != N_OPEN_SEEDS:
            errors.append(
                f"{path.name}: expected closed/open seeds {N_CLOSED_SEEDS}/{N_OPEN_SEEDS}, "
                f"got {len(closed_seeds)}/{len(open_seeds)}"
            )
        # Provenance on all rows once complete.
        _require_nonblank(df, "config_hash", path.name, errors)
        _require_nonblank(df, "split_hash", path.name, errors)
        _require_nonblank(df, "hash_labeled_parquet", path.name, errors)
        _require_nonblank(df, "hash_curves_all_parquet", path.name, errors)
        if "git_dirty" not in df.columns:
            errors.append(f"{path.name}: missing git_dirty")
        else:
            dirty = df["git_dirty"].astype(str).str.lower().isin({"true", "1", "yes"})
            if dirty.any():
                errors.append(
                    f"{path.name}: git_dirty=True on {int(dirty.sum())}/{len(df)} rows"
                )
        if "config_hash" in df.columns and MULTIREP_CONFIG.exists():
            expected_hash = _sha256(MULTIREP_CONFIG)
            vals = sorted(
                {
                    str(v)
                    for v in df.loc[~_blank_mask(df["config_hash"]), "config_hash"].tolist()
                }
            )
            if len(vals) > 1:
                errors.append(f"{path.name}: mixed config_hash ({len(vals)} distinct)")
            elif vals and vals[0] != expected_hash:
                errors.append(
                    f"{path.name}: config_hash {vals[0][:12]}… != "
                    f"paper_r3_multirep.yaml ({expected_hash[:12]}…)"
                )


def check_multirep(*, require_complete: bool = False) -> tuple[int, list[str], list[str]]:
    """Return (exit_code, errors, warnings). Exit 2 = soft not-started."""
    errors: list[str] = []
    warnings: list[str] = []
    print(f"[check_multirep] pipeline={PIPELINE_VERSION}")
    print(f"  seeds={list(PROTOCOL1_MULTIREP_SEEDS)}")
    print(f"  methods={list(PROTOCOL1_CORE_METHODS)}")
    if not MULTIREP_CONFIG.exists():
        errors.append(f"missing {MULTIREP_CONFIG.relative_to(ROOT)}")
    else:
        print(f"  config={MULTIREP_CONFIG.name} sha256={_sha256(MULTIREP_CONFIG)[:12]}…")

    for mode in ("well_id", "parent_grouped"):
        _check_csv(
            OUT / f"protocol1_multirep_{mode}_core6.csv",
            mode,
            errors,
            warnings,
            require_complete=require_complete,
        )

    regime = OUT / "protocol1_multirep_regime_counts.csv"
    if regime.exists():
        print(pd.read_csv(regime).to_string(index=False))
    else:
        warnings.append("missing protocol1_multirep_regime_counts.csv")

    for w in warnings:
        print("WARN:", w)
    for e in errors:
        print("ERROR:", e)

    if errors:
        return 1, errors, warnings
    if require_complete and warnings:
        return 1, errors, warnings
    if warnings and not (OUT / "protocol1_multirep_well_id_core6.csv").exists():
        return 2, errors, warnings  # soft: not started
    return 0, errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--require-complete",
        action="store_true",
        help="Treat incomplete multirep / dirty provenance as hard failure",
    )
    args = ap.parse_args()
    code, _, _ = check_multirep(require_complete=bool(args.require_complete))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
