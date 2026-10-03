"""Release gate: primary paper evidence must match the declared release fingerprint.

Checks (exit 1 on failure):
  - Protocol-1 summaries: exact PIPELINE_VERSION, single config_hash, closed-set
    methods+seeds, git_dirty=False, fixed labels, known W10 on open-set seed 44
  - Naive depth-split summary (0.737 headline companion)
  - Protocol-2 / Protocol-3: per-file cartesian completeness, hashes, no duplicates
  - matched_leakage_study.csv (Table 4) under current pipeline with non-blank split_hash
  - Required supplementary follow-up CSVs on current pipeline
  - Multi-seed paper table excludes seed 44
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import (
    PROTOCOL1_CLOSED_SEEDS,
    PROTOCOL1_MULTIREP_SEEDS,
    RESULTS_DIR,
)
from src.train import PIPELINE_VERSION

# Multi-rep primary (co-primary well-id schedule): locked 20 seeds, 18 closed / 2 open.
_MULTIREP_N_SEEDS = len(PROTOCOL1_MULTIREP_SEEDS)
_MULTIREP_N_CLOSED = 18
_MULTIREP_N_OPEN = 2

REQUIRED_P1_METHODS = {
    "lgbm_global",
    "lgbm_wellnorm",
    "lgbm_wellnorm_norel",
    "lgbm_hybrid",
    "lgbm_hybrid_norel",
    "lgbm_hybrid_while",
    "lgbm_hybrid_smooth",
    "rf_global_full",
    "xgb_global",
}

REQUIRED_P2_METHODS = {"lgbm_global", "lgbm_hybrid", "lgbm_wellnorm"}
REQUIRED_P3_METHODS = {"lgbm_global", "lgbm_hybrid", "lgbm_wellnorm"}

# Supplementary tables that must share the same pipeline as the main text.
REQUIRED_SUPP_FOLLOWUPS = (
    "matched_leakage_study.csv",
    "embargo_fixed_block_study.csv",
    "embargo_fixed_block_summary.csv",
    "while_sparse_prefix_seed42.csv",
    "acceptance_record_protocol1.csv",
)

RELEASE_CONFIG = ROOT / "configs" / "paper_r3_release.yaml"
FOLLOWUP = RESULTS_DIR / "followup"
HASH_COLS = (
    "config_hash",
    "split_hash",
    "hash_curves_all_parquet",
    "hash_labeled_parquet",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _blank_mask(s: pd.Series) -> pd.Series:
    """True where values are missing/empty (handles pandas NaN → str 'nan')."""
    return s.isna() | s.astype("string").str.strip().isin(["", "nan", "None", "<NA>"])


def _require_pipeline(df: pd.DataFrame, label: str, errors: list[str]) -> pd.DataFrame:
    if df.empty:
        errors.append(f"{label}: no rows")
        return df
    if "pipeline_version" not in df.columns:
        errors.append(f"{label}: missing pipeline_version")
        return df
    vers = sorted(df["pipeline_version"].astype(str).unique().tolist())
    if vers != [PIPELINE_VERSION]:
        errors.append(f"{label}: pipeline_version={vers}, expected {PIPELINE_VERSION!r}")
        return df[df["pipeline_version"].astype(str) == PIPELINE_VERSION].copy()
    return df


def _require_clean_git(df: pd.DataFrame, label: str, errors: list[str]) -> None:
    if df.empty:
        return
    if "git_dirty" not in df.columns:
        errors.append(f"{label}: missing git_dirty")
        return
    dirty = df["git_dirty"].astype(str).str.lower().isin({"true", "1", "yes"})
    if dirty.any():
        errors.append(f"{label}: git_dirty=True on {int(dirty.sum())}/{len(df)} rows")


def _require_nonblank_col(df: pd.DataFrame, col: str, label: str, errors: list[str]) -> None:
    if df.empty:
        return
    if col not in df.columns:
        errors.append(f"{label}: missing {col}")
        return
    bad = _blank_mask(df[col])
    if bad.any():
        errors.append(f"{label}: empty/NaN {col} on {int(bad.sum())}/{len(df)} rows")


def _require_single_hash_col(df: pd.DataFrame, col: str, label: str, errors: list[str]) -> None:
    _require_nonblank_col(df, col, label, errors)
    if df.empty or col not in df.columns:
        return
    vals = sorted({str(v) for v in df.loc[~_blank_mask(df[col]), col].tolist()})
    if len(vals) > 1:
        errors.append(f"{label}: mixed {col} ({len(vals)} distinct)")


def _require_single_config_hash(df: pd.DataFrame, label: str, errors: list[str]) -> None:
    _require_single_hash_col(df, "config_hash", label, errors)
    if df.empty or "config_hash" not in df.columns:
        return
    vals = sorted({str(v) for v in df.loc[~_blank_mask(df["config_hash"]), "config_hash"].tolist()})
    if vals and RELEASE_CONFIG.exists() and vals[0] != _sha256(RELEASE_CONFIG):
        errors.append(
            f"{label}: config_hash {vals[0][:12]}… != paper_r3_release.yaml "
            f"({_sha256(RELEASE_CONFIG)[:12]}…)"
        )


def _require_no_duplicate_keys(
    df: pd.DataFrame, keys: list[str], label: str, errors: list[str]
) -> None:
    if df.empty or any(k not in df.columns for k in keys):
        return
    dup = df.duplicated(subset=keys, keep=False)
    if dup.any():
        errors.append(
            f"{label}: duplicate rows on {keys} ({int(dup.sum())} offending rows)"
        )


def _load_release_cfg() -> dict:
    if not RELEASE_CONFIG.exists():
        return {}
    return yaml.safe_load(RELEASE_CONFIG.read_text(encoding="utf-8")) or {}


def _expected_p2_blocks(cfg: dict) -> set[int]:
    blocks: set[int] = set()
    for item in (cfg.get("protocol2") or {}).get("holdout_blocks") or []:
        if isinstance(item, (list, tuple)):
            for x in item:
                blocks.add(int(x))
        else:
            blocks.add(int(item))
    return blocks or {15, 35}


def _check_protocol_file_hashes(part: pd.DataFrame, fname: str, errors: list[str]) -> None:
    for col in HASH_COLS:
        if col not in part.columns:
            # split_hash / config_hash required; data hashes preferred
            if col in ("config_hash", "split_hash"):
                errors.append(f"{fname}: missing {col}")
            continue
        _require_single_hash_col(part, col, fname, errors)


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    cfg = _load_release_cfg()

    # ----- Protocol-1 -----
    p1_files = sorted(RESULTS_DIR.glob("summary_protocol1_random_wells_seed*.csv"))
    if not p1_files:
        errors.append("no summary_protocol1_random_wells_seed*.csv present")
        df = pd.DataFrame()
    else:
        df = pd.concat([pd.read_csv(f) for f in p1_files], ignore_index=True)
        df = _require_pipeline(df, "Protocol-1", errors)
        _require_clean_git(df, "Protocol-1", errors)
        _require_single_config_hash(df, "Protocol-1", errors)

        if "worst_well_f1_fixed" in df.columns:
            blank = _blank_mask(df["worst_well_f1_fixed"])
            if blank.any():
                errors.append(
                    f"Protocol-1: worst_well_f1_fixed empty on {int(blank.sum())}/{len(df)} rows"
                )
        else:
            errors.append("Protocol-1: missing worst_well_f1_fixed")

        closed = df[df["seed"].isin(PROTOCOL1_CLOSED_SEEDS)] if not df.empty else df
        for seed in PROTOCOL1_CLOSED_SEEDS:
            sub = closed[closed["seed"] == seed] if not closed.empty else closed
            missing = (
                REQUIRED_P1_METHODS - set(sub["method"].astype(str))
                if not sub.empty
                else REQUIRED_P1_METHODS
            )
            if missing:
                errors.append(f"Protocol-1 seed {seed} missing methods: {sorted(missing)}")

        s44 = df[df["seed"] == 44] if not df.empty else df
        if not s44.empty:
            if "known_worst10pct_well_f1" not in s44.columns:
                errors.append("Protocol-1 seed 44: missing known_worst10pct_well_f1 column")
            else:
                for m in (
                    "lgbm_global",
                    "lgbm_wellnorm",
                    "lgbm_hybrid",
                    "lgbm_hybrid_norel",
                    "lgbm_hybrid_while",
                ):
                    row = s44[s44["method"] == m]
                    if row.empty:
                        errors.append(f"Protocol-1 seed 44 missing method {m}")
                        continue
                    if _blank_mask(row["known_worst10pct_well_f1"]).any():
                        errors.append(f"Protocol-1 seed 44 {m}: empty known_worst10pct_well_f1")

        for f in p1_files:
            part = pd.read_csv(f)
            _check_protocol_file_hashes(part, f.name, errors)

    # ----- Naive depth-split (0.737 companion) -----
    naive_csv = RESULTS_DIR / "summary_naive_depth_split.csv"
    naive_json = RESULTS_DIR / "details_naive_depth_split.json"
    if not naive_csv.exists():
        errors.append("missing summary_naive_depth_split.csv")
    else:
        nd = pd.read_csv(naive_csv)
        nd = _require_pipeline(nd, "naive", errors)
        _require_clean_git(nd, "naive", errors)
        if not nd.empty:
            _require_nonblank_col(nd, "split_hash", "naive", errors)
            if "row_index_hash" in nd.columns:
                _require_nonblank_col(nd, "row_index_hash", "naive", errors)
            macro = float(nd.iloc[0].get("macro_f1", float("nan")))
            if not (0.7 <= macro <= 0.8):
                warnings.append(f"naive macro_f1={macro:.3f} outside expected ~0.737 band")
    if naive_json.exists():
        try:
            detail = json.loads(naive_json.read_text(encoding="utf-8"))
            pv = str(detail.get("pipeline_version", ""))
            if pv and pv != PIPELINE_VERSION:
                errors.append(
                    f"details_naive_depth_split.json pipeline_version={pv!r} "
                    f"!= {PIPELINE_VERSION!r}"
                )
            if not detail.get("config_hash"):
                errors.append("details_naive_depth_split.json: empty config_hash")
            if not detail.get("split_hash"):
                errors.append("details_naive_depth_split.json: empty split_hash")
        except json.JSONDecodeError:
            errors.append("details_naive_depth_split.json: invalid JSON")
    else:
        errors.append("missing details_naive_depth_split.json")

    # ----- Protocol-2 (per holdout × method cartesian) -----
    p2_files = sorted(RESULTS_DIR.glob("summary_protocol2_block_holdout_*.csv"))
    expected_blocks = _expected_p2_blocks(cfg)
    if not p2_files:
        errors.append("no Protocol-2 summary_protocol2_block_holdout_*.csv")
    else:
        p2_parts: list[pd.DataFrame] = []
        seen_blocks: set[int] = set()
        for f in p2_files:
            part = pd.read_csv(f)
            part = _require_pipeline(part, f"Protocol-2/{f.name}", errors)
            _require_clean_git(part, f"Protocol-2/{f.name}", errors)
            _check_protocol_file_hashes(part, f.name, errors)
            if part.empty:
                continue
            if "holdout_blocks" not in part.columns or "method" not in part.columns:
                errors.append(f"{f.name}: missing holdout_blocks/method")
                continue
            blocks = {int(x) for x in part["holdout_blocks"].dropna().unique().tolist()}
            if len(blocks) != 1:
                errors.append(f"{f.name}: expected one holdout_blocks, got {sorted(blocks)}")
            else:
                seen_blocks |= blocks
            missing = REQUIRED_P2_METHODS - set(part["method"].astype(str))
            if missing:
                errors.append(f"{f.name}: missing methods {sorted(missing)}")
            _require_no_duplicate_keys(part, ["method", "seed", "holdout_blocks"], f.name, errors)
            # Full cartesian: every required method present exactly once per seed/block
            for blk in blocks:
                for seed in sorted(part["seed"].dropna().unique().tolist()):
                    sub = part[(part["holdout_blocks"] == blk) & (part["seed"] == seed)]
                    have = set(sub["method"].astype(str))
                    miss = REQUIRED_P2_METHODS - have
                    if miss:
                        errors.append(
                            f"{f.name}: holdout={blk} seed={seed} missing methods {sorted(miss)}"
                        )
            p2_parts.append(part)
        missing_blocks = expected_blocks - seen_blocks
        if missing_blocks:
            errors.append(f"Protocol-2 missing holdout blocks: {sorted(missing_blocks)}")
        if p2_parts:
            p2 = pd.concat(p2_parts, ignore_index=True)
            _require_single_config_hash(p2, "Protocol-2", errors)

    # ----- Protocol-3 (per target_well × method cartesian) -----
    p3_files = sorted(RESULTS_DIR.glob("summary_protocol3_lowo_*.csv"))
    p3_max = int((cfg.get("protocol3") or {}).get("max_targets") or 6)
    if not p3_files:
        errors.append("no Protocol-3 summary_protocol3_lowo_*.csv")
    else:
        p3_parts: list[pd.DataFrame] = []
        targets: set[str] = set()
        for f in p3_files:
            part = pd.read_csv(f)
            part = _require_pipeline(part, f"Protocol-3/{f.name}", errors)
            _require_clean_git(part, f"Protocol-3/{f.name}", errors)
            _check_protocol_file_hashes(part, f.name, errors)
            if part.empty:
                continue
            if "target_well" not in part.columns or "method" not in part.columns:
                errors.append(f"{f.name}: missing target_well/method")
                continue
            tws = {str(x) for x in part["target_well"].dropna().unique().tolist()}
            if len(tws) != 1:
                errors.append(f"{f.name}: expected one target_well, got {sorted(tws)}")
            else:
                targets |= tws
            missing = REQUIRED_P3_METHODS - set(part["method"].astype(str))
            if missing:
                errors.append(f"{f.name}: missing methods {sorted(missing)}")
            _require_no_duplicate_keys(part, ["method", "seed", "target_well"], f.name, errors)
            for tw in tws:
                for seed in sorted(part["seed"].dropna().unique().tolist()):
                    sub = part[(part["target_well"].astype(str) == tw) & (part["seed"] == seed)]
                    miss = REQUIRED_P3_METHODS - set(sub["method"].astype(str))
                    if miss:
                        errors.append(
                            f"{f.name}: target={tw} seed={seed} missing methods {sorted(miss)}"
                        )
            p3_parts.append(part)
        if len(targets) < p3_max:
            errors.append(
                f"Protocol-3 has {len(targets)} target wells, expected >= {p3_max}"
            )
        if p3_parts:
            p3 = pd.concat(p3_parts, ignore_index=True)
            _require_single_config_hash(p3, "Protocol-3", errors)

    # ----- Table 4 matched leakage -----
    matched = FOLLOWUP / "matched_leakage_study.csv"
    if not matched.exists():
        errors.append("missing results/followup/matched_leakage_study.csv")
    else:
        md = pd.read_csv(matched)
        md = _require_pipeline(md, "matched_leakage", errors)
        _require_clean_git(md, "matched_leakage", errors)
        if not md.empty and len(md) < 5:
            warnings.append(f"matched_leakage has only {len(md)} rows")
        settings = set(md["setting"].astype(str)) if not md.empty and "setting" in md.columns else set()
        for need in (
            "A_naive_depth",
            "B_depth_blocks",
            "C_well_holdout",
            "D_fixed_wells_depth_leak",
            "C_matched_same_test_as_D",
            "D_eq_n_train",
        ):
            if need not in settings:
                errors.append(f"matched_leakage: missing setting {need}")
        for col in ("config_hash", "split_hash", "git_commit"):
            _require_nonblank_col(md, col, "matched_leakage", errors)
        if "setting" in md.columns and "split_hash" in md.columns and not md.empty:
            # Each setting must have its own non-blank split_hash (no shared empty stub).
            for setting, sub in md.groupby(md["setting"].astype(str)):
                if _blank_mask(sub["split_hash"]).any():
                    errors.append(f"matched_leakage: empty split_hash for setting {setting}")

    # ----- Required supplementary follow-ups (provenance + integrity) -----
    EXPECTED_EMBARGO_M = {0.0, 1.0, 5.0, 10.0, 20.0}
    EXPECTED_EMBARGO_SEEDS = set(range(10))
    EXPECTED_EMBARGO_ROWS = len(EXPECTED_EMBARGO_M) * len(EXPECTED_EMBARGO_SEEDS)

    for name in REQUIRED_SUPP_FOLLOWUPS:
        p = FOLLOWUP / name
        if not p.exists():
            errors.append(f"missing required supplementary follow-up: {name}")
            continue
        try:
            fu = pd.read_csv(p)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: failed to read ({exc})")
            continue
        fu = _require_pipeline(fu, name, errors)
        if fu.empty:
            continue

        if name == "embargo_fixed_block_study.csv":
            _require_clean_git(fu, name, errors)
            for col in ("config_hash", "git_commit", "hash_labeled_parquet"):
                _require_nonblank_col(fu, col, name, errors)
            _require_single_hash_col(fu, "config_hash", name, errors)
            if "embargo_m" not in fu.columns or "subsample_seed" not in fu.columns:
                errors.append(f"{name}: missing embargo_m/subsample_seed")
            else:
                if len(fu) != EXPECTED_EMBARGO_ROWS:
                    errors.append(
                        f"{name}: expected {EXPECTED_EMBARGO_ROWS} rows, got {len(fu)}"
                    )
                got_m = {float(x) for x in fu["embargo_m"].tolist()}
                if got_m != EXPECTED_EMBARGO_M:
                    errors.append(
                        f"{name}: embargo_m={sorted(got_m)} != {sorted(EXPECTED_EMBARGO_M)}"
                    )
                for emb in EXPECTED_EMBARGO_M:
                    sub = fu[fu["embargo_m"].astype(float) == emb]
                    seeds = {int(s) for s in sub["subsample_seed"].tolist()}
                    if seeds != EXPECTED_EMBARGO_SEEDS:
                        errors.append(
                            f"{name}: embargo_m={emb} seeds={sorted(seeds)} "
                            f"!= {sorted(EXPECTED_EMBARGO_SEEDS)}"
                        )
                # no duplicate (embargo_m, subsample_seed)
                if fu.duplicated(subset=["embargo_m", "subsample_seed"]).any():
                    errors.append(f"{name}: duplicate embargo_m×subsample_seed rows")
            if "train_val_disjoint" in fu.columns:
                ok = fu["train_val_disjoint"].astype(str).str.lower().isin(
                    {"true", "1", "yes"}
                )
                if not ok.all():
                    errors.append(f"{name}: train_val_disjoint not True on all rows")

        elif name == "embargo_fixed_block_summary.csv":
            if "embargo_m" not in fu.columns:
                errors.append(f"{name}: missing embargo_m")
            else:
                got_m = {float(x) for x in fu["embargo_m"].tolist()}
                if got_m != EXPECTED_EMBARGO_M or len(fu) != len(EXPECTED_EMBARGO_M):
                    errors.append(
                        f"{name}: expected one row per {sorted(EXPECTED_EMBARGO_M)}, "
                        f"got {len(fu)} rows / {sorted(got_m)}"
                    )
            for col in ("macro_mean", "worst_mean", "tv_mean"):
                if col not in fu.columns:
                    errors.append(f"{name}: missing {col}")

        elif name == "while_sparse_prefix_seed42.csv":
            if len(fu) < 5:
                errors.append(f"{name}: expected >=5 diagnostic rows, got {len(fu)}")
            # Prefer full provenance when columns exist (older dumps may lack them).
            if "git_dirty" in fu.columns:
                _require_clean_git(fu, name, errors)
            else:
                warnings.append(f"{name}: missing git_dirty (regenerate for full provenance)")
            if "config_hash" in fu.columns:
                _require_single_hash_col(fu, "config_hash", name, errors)
            else:
                warnings.append(f"{name}: missing config_hash (regenerate for full provenance)")

        elif name == "acceptance_record_protocol1.csv":
            if "record_scope" in fu.columns and (
                ~fu["record_scope"].astype(str).str.contains("primary_multirep_n18", regex=False)
            ).any():
                errors.append(
                    f"{name}: record_scope must be primary_multirep_n18_well_id"
                )
            if "primary_seeds" in fu.columns and (
                ~fu["primary_seeds"].astype(str).str.contains("n=18", regex=False)
            ).any():
                errors.append(f"{name}: primary_seeds must reference multi-rep n=18")
            if "n_closed_reps" in fu.columns and (fu["n_closed_reps"] != 18).any():
                errors.append(f"{name}: n_closed_reps must be 18 for all methods")
            for col in (
                "macro_f1_mean_closed",
                "worst_well_f1_fixed_mean_closed",
                "w10_fixed_mean_closed",
            ):
                if col not in fu.columns:
                    errors.append(f"{name}: missing {col}")
            if len(fu) < 5:
                errors.append(f"{name}: too few method rows ({len(fu)})")

        elif name == "matched_leakage_study.csv":
            # Detailed checks already applied above; pipeline already enforced.
            pass

    # Soft checks for optional companions (canonical names only).
    # Note: depth_embargo_grid.csv is obsolete; paper uses embargo_fixed_block_*.
    for name in ("protocol1_parent_grouped_global.csv",):
        p = FOLLOWUP / name
        if not p.exists():
            alias = FOLLOWUP / "parent_grouped_protocol1.csv"
            if alias.exists():
                p = alias
            else:
                warnings.append(f"optional follow-up missing: {name}")
                continue
        fu = pd.read_csv(p)
        if "pipeline_version" in fu.columns:
            vers = sorted(fu["pipeline_version"].astype(str).unique().tolist())
            if vers != [PIPELINE_VERSION]:
                warnings.append(
                    f"optional {name}: pipeline_version={vers} != {PIPELINE_VERSION!r}"
                )

    # paper multi-seed table + submission figure must match canonical results
    multi_canonical = RESULTS_DIR / "paper_figures" / "table_protocol1_multiseed.csv"
    multi_submission = ROOT / "paper" / "gse" / "figs" / "table_protocol1_multiseed.csv"
    fig_canonical = RESULTS_DIR / "paper_figures" / "fig6_protocol1_multiseed.png"
    fig_submission = ROOT / "paper" / "gse" / "figs" / "fig6_protocol1_multiseed.png"
    multi_paths = [multi_canonical, multi_submission]
    for mp in multi_paths:
        if not mp.exists():
            errors.append(f"missing {mp.relative_to(ROOT)}")
            continue
        mt = pd.read_csv(mp)
        if "seeds" in mt.columns and mt["seeds"].astype(str).str.contains("44").any():
            errors.append(f"{mp.name}: seeds column includes 44")
        if "n_seeds" in mt.columns and (mt["n_seeds"] > len(PROTOCOL1_CLOSED_SEEDS)).any():
            errors.append(f"{mp.name}: n_seeds exceeds closed-set size")
        # Guard against rounded/near-zero-std drift that previously broke Fig.~5.
        for method in ("lgbm_global", "lgbm_hybrid"):
            sub = mt[mt["method"].astype(str) == method] if "method" in mt.columns else mt.iloc[0:0]
            if sub.empty or "macro_std" not in sub.columns:
                continue
            std = float(sub.iloc[0]["macro_std"])
            if std < 1e-3 and int(sub.iloc[0].get("n_seeds", 0) or 0) >= 3:
                errors.append(
                    f"{mp.name}: {method} macro_std={std:.2e} looks collapsed "
                    "(expected sample std over closed seeds ≈0.05–0.07)"
                )

    if multi_canonical.exists() and multi_submission.exists():
        a = pd.read_csv(multi_canonical)
        b = pd.read_csv(multi_submission)
        # Compare on shared identity columns used by the figure/table.
        key_cols = [
            c
            for c in (
                "method",
                "macro_mean",
                "macro_std",
                "worst_mean",
                "worst_std",
                "n_seeds",
                "seeds",
            )
            if c in a.columns and c in b.columns
        ]
        if key_cols:
            a2 = a[key_cols].sort_values("method").reset_index(drop=True)
            b2 = b[key_cols].sort_values("method").reset_index(drop=True)
            if not a2.equals(b2):
                errors.append(
                    "paper/gse/figs/table_protocol1_multiseed.csv differs from "
                    "results/paper_figures/table_protocol1_multiseed.csv"
                )

    if fig_canonical.exists() and fig_submission.exists():
        h_can = _sha256(fig_canonical)
        h_sub = _sha256(fig_submission)
        if h_can != h_sub:
            errors.append(
                "paper/gse/figs/fig6_protocol1_multiseed.png sha256 differs from "
                "results/paper_figures/fig6_protocol1_multiseed.png "
                f"({h_sub[:12]}… != {h_can[:12]}…)"
            )
    elif not fig_submission.exists():
        errors.append("missing paper/gse/figs/fig6_protocol1_multiseed.png")
    elif not fig_canonical.exists():
        warnings.append("missing results/paper_figures/fig6_protocol1_multiseed.png")

    # ----- Protocol-1 multi-rep companion (well_id + parent_grouped; hard) -----
    import importlib.util

    mr_path = ROOT / "scripts" / "check_multirep_r3.py"
    if mr_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("check_multirep_r3", mr_path)
            assert spec is not None and spec.loader is not None
            mr_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mr_mod)
            mr_code, mr_errs, mr_warns = mr_mod.check_multirep(require_complete=True)
            if mr_code == 1:
                for e in mr_errs:
                    errors.append(f"multirep: {e}")
            for w in mr_warns:
                warnings.append(f"multirep: {w}")
            if mr_code == 0 and not mr_warns:
                print("OK: multirep companion gate (require-complete) passed.")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"multirep companion gate failed: {exc}")
    else:
        errors.append("missing scripts/check_multirep_r3.py")

    if warnings:
        print("WARN:")
        for w in warnings:
            print(" ", w)
    if errors:
        print("FAIL: release gate")
        for e in errors:
            print(" ", e)
        print(
            f"\nExpected PIPELINE_VERSION={PIPELINE_VERSION}. "
            "Use configs/paper_r3_release.yaml as the sole full-suite schedule "
            "(see RELEASE_CHECKLIST.md). Multirep: configs/paper_r3_multirep.yaml."
        )
        return 1

    print(
        f"OK: release gate passed "
        f"(pipeline={PIPELINE_VERSION}, "
        f"multirep_primary={_MULTIREP_N_SEEDS} seeds / "
        f"{_MULTIREP_N_CLOSED} closed / {_MULTIREP_N_OPEN} open, "
        f"historical_closed_seeds={list(PROTOCOL1_CLOSED_SEEDS)}, "
        f"n_p1_rows={len(df)})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
