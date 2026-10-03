#!/usr/bin/env python
"""CLI: prepare data and run cross-well DG experiments."""
from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

# allow `py -m` and direct script execution
ROOT = Path(__file__).resolve().parent  # cross_well_dg/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prepare_data import load_curves_all, load_labeled, prepare_dataset
from src.splits import protocol1_random_wells, protocol2_block_holdout, protocol3_leave_one_well_out
from src.train import (
    PIPELINE_VERSION,
    run_naive_leakage_baseline,
    run_protocol_experiment,
    set_curve_context,
)
from src.constants import RESULTS_DIR, PROCESSED_DIR
from src.provenance import freeze_git_provenance

# Prevent concurrent run_experiment instances from thrashing CPU / corrupting resumes.
_RUN_LOCK_PATH = RESULTS_DIR / ".run_experiment.lock"
_LOCK_FH = None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes

            SYNCHRONIZE = 0x00100000
            handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_run_lock(*, force: bool = False, meta: dict | None = None) -> None:
    """Exclusive lock so only one full experiment runner is active."""
    global _LOCK_FH
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if _RUN_LOCK_PATH.exists() and not force:
        try:
            raw = _RUN_LOCK_PATH.read_text(encoding="utf-8")
            old = json.loads(raw) if raw.strip() else {}
            old_pid = int(old.get("pid", 0))
            if _pid_alive(old_pid) and old_pid != os.getpid():
                raise SystemExit(
                    f"Another run_experiment is active (pid={old_pid}, "
                    f"started={old.get('started')}, config={old.get('config')}).\n"
                    f"Wait for it to finish, or delete {_RUN_LOCK_PATH} only if that "
                    f"process is truly dead."
                )
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    # Open+truncate once; keep handle for the process lifetime (Windows-safe).
    _LOCK_FH = open(_RUN_LOCK_PATH, "a+", encoding="utf-8")
    try:
        if os.name == "nt":
            import msvcrt

            _LOCK_FH.seek(0)
            msvcrt.locking(_LOCK_FH.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(_LOCK_FH.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        _LOCK_FH.close()
        _LOCK_FH = None
        raise SystemExit(
            f"Could not acquire run lock at {_RUN_LOCK_PATH}: {exc}. "
            "Another experiment is likely running."
        ) from exc

    payload = meta or {
        "pid": os.getpid(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pipeline_version": PIPELINE_VERSION,
    }
    _LOCK_FH.seek(0)
    _LOCK_FH.truncate()
    _LOCK_FH.write(json.dumps(payload, indent=2))
    _LOCK_FH.flush()

    def _release() -> None:
        global _LOCK_FH
        if _LOCK_FH is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                _LOCK_FH.seek(0)
                msvcrt.locking(_LOCK_FH.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(_LOCK_FH.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            _LOCK_FH.close()
        except OSError:
            pass
        _LOCK_FH = None
        try:
            if _RUN_LOCK_PATH.exists():
                _RUN_LOCK_PATH.unlink()
        except OSError:
            pass

    atexit.register(_release)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_prepare(args: argparse.Namespace) -> None:
    df = prepare_dataset(max_wells=args.max_wells, save=True)
    print(f"Labeled rows: {len(df):,}")
    print(f"Wells: {df['well_id'].nunique()}")
    print(f"Saved to {PROCESSED_DIR}")


def cmd_run(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    # Freeze source-tree git provenance BEFORE lock / any result writes so that
    # tracked summary CSVs and the lock file cannot flip git_dirty mid-run.
    git_snap = freeze_git_provenance(ignore_outputs=True)
    print(
        f"[provenance] git_commit={git_snap.get('git_commit', '')[:12]}… "
        f"git_dirty={git_snap.get('git_dirty')} "
        f"(outputs under results/ ignored for dirty check)"
    )
    acquire_run_lock(
        force=bool(getattr(args, "force_unlock", False)),
        meta={
            "pid": os.getpid(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": str(config_path.resolve()),
            "protocol": args.protocol,
            "pipeline_version": PIPELINE_VERSION,
            "git_commit": git_snap.get("git_commit", ""),
            "git_dirty": git_snap.get("git_dirty", True),
        },
    )
    print(f"[lock] acquired {_RUN_LOCK_PATH} (pid={os.getpid()})")

    cfg = load_config(config_path)
    max_wells = args.max_wells if args.max_wells is not None else cfg.get("max_wells")
    methods = args.methods.split(",") if args.methods else cfg["methods"]
    methods = [m.strip() for m in methods if m.strip()]

    # Absorbed companion configs: refuse unless explicitly allowed (avoids wiping
    # other methods from shared summary_protocol1_*.csv via config_hash resume).
    _ABSORBED = {
        "paper_equal_budget_p1.yaml",
        "paper_r3_wellnorm_norel.yaml",
    }
    if config_path.name in _ABSORBED and not getattr(args, "allow_partial", False):
        raise SystemExit(
            f"Refusing configs/{config_path.name}: it is absorbed into "
            "configs/paper_r3_release.yaml.\n"
            "Running it alone would drop foreign config_hash rows from shared "
            "summary_*.csv files.\n"
            "Use paper_r3_release.yaml, or pass --allow-partial for a deliberate "
            "partial debug run."
        )

    # Canonical paper schedules: refuse partial method/protocol overrides unless forced.
    _CANONICAL = {"paper_r3_release.yaml", "paper_r3_full.yaml"}
    if (
        config_path.name in _CANONICAL
        and not getattr(args, "allow_partial", False)
        and (args.protocol != "all" or args.methods is not None)
    ):
        raise SystemExit(
            f"Refusing partial override of configs/{config_path.name} "
            f"(protocol={args.protocol!r}, methods={args.methods!r}).\n"
            "Use the full schedule so results stay complete, or pass --allow-partial.\n"
            "Preferred release config: configs/paper_r3_release.yaml "
            "(do not chain equal_budget / wellnorm_norel into the same summary CSV)."
        )

    labeled_path = PROCESSED_DIR / "labeled.parquet"
    curves_path = PROCESSED_DIR / "curves_all.parquet"
    if not labeled_path.exists() or not curves_path.exists() or args.reprepare:
        print("Preparing dataset (labeled + full curves)...")
        prepare_dataset(max_wells=max_wells, save=True)
    df = load_labeled()
    curves_all = load_curves_all()
    if max_wells is not None:
        wells = sorted(df["well_id"].unique())[: max_wells]
        df = df[df["well_id"].isin(wells)].copy()
        curves_all = curves_all[curves_all["well_id"].isin(wells)].copy()
        print(f"Smoke subset: {len(wells)} wells, {len(df):,} labeled rows")
    set_curve_context(curves_all)
    print(f"pipeline_version={PIPELINE_VERSION} | curves_all={len(curves_all):,} rows")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_tables = []

    if cfg.get("naive_leakage_baseline", True) and not args.skip_naive:
        print("[run] naive depth-split leakage baseline")
        naive = run_naive_leakage_baseline(
            df, seed=int(cfg.get("seed", 42)), config_path=config_path
        )
        pd.DataFrame(
            [
                {
                    "protocol": "naive_depth_split",
                    "method": naive["method"],
                    **naive["overall"],
                    **naive["well_summary"],
                    "seconds": naive["seconds"],
                    "seed": naive.get("seed", int(cfg.get("seed", 42))),
                    "holdout_blocks": "",
                    "target_well": "",
                    "n_train": naive.get("n_train", ""),
                    "n_test": naive.get("n_test", ""),
                    "n_test_unseen_label_kept": naive.get("n_test_unseen_label_kept", ""),
                    "n_val_dropped_unseen_label": naive.get("n_val_dropped_unseen_label", ""),
                    "uda_target_mode": naive.get("uda_target_mode", "none"),
                    "smoothing_scope": naive.get("smoothing_scope", "none"),
                    "early_stopping_metric": naive.get("early_stopping_metric", ""),
                    "n_estimators_budget": naive.get("n_estimators_budget", ""),
                    "pipeline_version": naive.get("pipeline_version", PIPELINE_VERSION),
                    "feature_mode": naive.get("feature_mode", ""),
                    "use_relative_depth": naive.get("use_relative_depth", ""),
                    "well_z_mode": naive.get("well_z_mode", ""),
                    "n_features": naive.get("n_features", ""),
                    "git_commit": naive.get("git_commit", ""),
                    "git_dirty": naive.get("git_dirty", ""),
                    "config_hash": naive.get("config_hash", ""),
                    "split_hash": naive.get("split_hash", ""),
                    "hash_curves_all_parquet": naive.get("hash_curves_all_parquet", ""),
                    "hash_labeled_parquet": naive.get("hash_labeled_parquet", ""),
                    "hash_dataset_meta_json": naive.get("hash_dataset_meta_json", ""),
                    "feature_param_hash": naive.get("feature_param_hash", ""),
                    "row_index_hash": naive.get("row_index_hash", ""),
                    "test_frac": naive.get("test_frac", ""),
                    "val_frac": naive.get("val_frac", ""),
                    "worst_well_f1_fixed": (naive.get("well_summary_fixed") or {}).get(
                        "worst_well_f1", ""
                    ),
                    "fixed_label_mode": naive.get("fixed_label_mode", ""),
                }
            ]
        ).to_csv(RESULTS_DIR / "summary_naive_depth_split.csv", index=False)
        (RESULTS_DIR / "details_naive_depth_split.json").write_text(
            json.dumps(naive, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )

    seeds = cfg.get("seeds", [cfg.get("seed", 42)])
    if args.seed is not None:
        seeds = [args.seed]

    # Protocol-1
    if cfg.get("protocol1", {}).get("enabled", True) and args.protocol in {"all", "1"}:
        p1 = cfg["protocol1"]
        for seed in seeds:
            split = protocol1_random_wells(
                df,
                seed=int(seed),
                train_frac=p1.get("train_frac", 0.7),
                val_frac=p1.get("val_frac", 0.15),
            )
            table = run_protocol_experiment(
                df, split, methods, curves_all=curves_all, config_path=config_path
            )
            all_tables.append(table)

    # Protocol-2
    if cfg.get("protocol2", {}).get("enabled", True) and args.protocol in {"all", "2"}:
        for holdout in cfg["protocol2"].get("holdout_blocks", [["15"]]):
            for seed in seeds[:1]:  # one seed for block holdout by default
                split = protocol2_block_holdout(df, holdout_blocks=holdout, seed=int(seed))
                table = run_protocol_experiment(
                    df, split, methods, curves_all=curves_all, config_path=config_path
                )
                all_tables.append(table)

    # Protocol-3
    if cfg.get("protocol3", {}).get("enabled", True) and args.protocol in {"all", "3"}:
        p3 = cfg["protocol3"]
        splits = protocol3_leave_one_well_out(
            df,
            max_targets=int(p3.get("max_targets", 8)),
            seed=int(seeds[0]),
            selection=str(p3.get("selection", "spaced")),
        )
        # LOWO: allow hybrid methods; fall back to first two if filter empty
        allow = {
            "lgbm_wellnorm",
            "lgbm_wellnorm_norel",
            "lgbm_full",
            "lgbm_groupdro",
            "lgbm_hybrid",
            "lgbm_hybrid_norel",
            "lgbm_hybrid_while",
            "lgbm_hybrid_sim",
            "lgbm_hybrid_smooth",
            "lgbm_hybrid_plus",
            "lgbm_hybrid_sim_perwell",
            "lgbm_global",
            "lgbm_global_eq120k",
            "rf_global",
            "rf_global_full",
            "xgb_global",
            "mlp_source_only",
            "mlp_coral_uda",
        }
        lowo_methods = [m for m in methods if m in allow]
        if not lowo_methods:
            lowo_methods = methods[:2]
        for split in splits:
            table = run_protocol_experiment(
                df, split, lowo_methods, curves_all=curves_all, config_path=config_path
            )
            all_tables.append(table)

    # merge every summary_*.csv for current pipeline version only
    summary_files = sorted(RESULTS_DIR.glob("summary_*.csv"))
    summary_files = [f for f in summary_files if f.name != "summary_all.csv" and "archive" not in f.name]
    merged = None
    if summary_files:
        parts = []
        for f in summary_files:
            chunk = pd.read_csv(f)
            if "pipeline_version" in chunk.columns:
                chunk = chunk[chunk["pipeline_version"].astype(str) == PIPELINE_VERSION]
            if not chunk.empty:
                parts.append(chunk)
        if parts:
            merged = pd.concat(parts, ignore_index=True)
            merged.to_csv(RESULTS_DIR / "summary_all.csv", index=False)
            print(
                f"Merged {len(parts)} summary files (pipeline={PIPELINE_VERSION}) "
                f"into summary_all.csv"
            )
    elif all_tables:
        merged = pd.concat(all_tables, ignore_index=True)
        merged.to_csv(RESULTS_DIR / "summary_all.csv", index=False)

    if merged is not None:
        print("\n=== Combined summary (head) ===")
        cols = [
            c
            for c in [
                "protocol",
                "method",
                "macro_f1",
                "well_macro_f1_mean",
                "worst_well_f1",
                "worst10pct_well_f1",
                "seconds",
            ]
            if c in merged.columns
        ]
        print(merged[cols].head(20).to_string(index=False))
        print(f"\nResults saved under {RESULTS_DIR}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cross-well domain generalization experiments")
    sub = p.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="Parse LAS and build labeled parquet")
    p_prep.add_argument("--max-wells", type=int, default=None)
    p_prep.set_defaults(func=cmd_prepare)

    p_run = sub.add_parser("run", help="Run experimental protocols")
    p_run.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p_run.add_argument("--max-wells", type=int, default=None, help="Smoke test subset")
    p_run.add_argument("--methods", type=str, default=None, help="Comma-separated method list")
    p_run.add_argument("--protocol", choices=["all", "1", "2", "3"], default="all")
    p_run.add_argument("--seed", type=int, default=None)
    p_run.add_argument("--reprepare", action="store_true")
    p_run.add_argument("--skip-naive", action="store_true")
    p_run.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow --protocol/--methods overrides on release configs, or run absorbed companion YAMLs",
    )
    p_run.add_argument(
        "--force-unlock",
        action="store_true",
        help="Ignore stale lock file (only if no other run_experiment is alive)",
    )
    p_run.set_defaults(func=cmd_run)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
