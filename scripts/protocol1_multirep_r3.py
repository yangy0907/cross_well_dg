"""Protocol-1 multi-rep Round-3 runner (core 6 methods × 20 seeds).

Locked protocol (see ``src/constants.py``):
  - Closed-set a priori: test has zero depths with facies absent from training.
  - Core methods only (PROTOCOL1_CORE_METHODS).
  - Seeds: PROTOCOL1_MULTIREP_SEEDS (includes 42/43/44/45 + 16 predetermined).
  - Split modes: well-id Protocol-1 and parent-grouped Protocol-1 (co-primary
    leakage-sensitive analysis).

Crash-safe resume: per (split_mode, seed, method) rows in CSV; skips completed
provenance-matched rows. Sequential seeds; n_jobs inside model fits as usual.

Usage (PowerShell)::

    py -u scripts/protocol1_multirep_r3.py --split-mode well_id
    py -u scripts/protocol1_multirep_r3.py --split-mode parent_grouped
    py -u scripts/protocol1_multirep_r3.py --split-mode both

Do NOT run embargo / matched / full 16-method / P2 / P3 from this script.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import (  # noqa: E402
    LABEL_COL,
    PROTOCOL1_CORE_METHODS,
    PROTOCOL1_MULTIREP_SEEDS,
    RESULTS_DIR,
    WELL_COL,
)
from src.prepare_data import load_curves_all, load_labeled  # noqa: E402
from src.provenance import freeze_git_provenance  # noqa: E402
from src.splits import (  # noqa: E402
    apply_split,
    protocol1_parent_grouped_wells,
    protocol1_random_wells,
)
from src.train import (  # noqa: E402
    PIPELINE_VERSION,
    run_method_on_split,
    set_curve_context,
    set_run_meta,
)

OUT_DIR = RESULTS_DIR / "multirep"
FOLLOWUP = RESULTS_DIR / "followup"
MULTIREP_CONFIG = ROOT / "configs" / "paper_r3_multirep.yaml"


def _a_priori_regime(parts: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Closed iff test has zero depths with facies absent from train labels."""
    train_labs = set(int(x) for x in parts["train"][LABEL_COL].unique())
    test_labs = parts["test"][LABEL_COL].astype(int)
    unseen_mask = ~test_labs.isin(train_labs)
    n_unseen = int(unseen_mask.sum())
    return {
        "label_regime": "closed_set" if n_unseen == 0 else "open_set",
        "n_test_unseen_label_kept": n_unseen,
        "n_test": int(len(test_labs)),
        "unseen_frac": float(n_unseen) / float(len(test_labs)) if len(test_labs) else 0.0,
        "train_n_classes": int(len(train_labs)),
        "test_n_classes": int(test_labs.nunique()),
    }


def _csv_path(split_mode: str) -> Path:
    return OUT_DIR / f"protocol1_multirep_{split_mode}_core6.csv"


def _details_dir(split_mode: str) -> Path:
    d = OUT_DIR / "details" / split_mode
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_done(path: Path) -> set[tuple[int, str]]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty or "method" not in df.columns:
        return set()
    # Resume only rows matching current pipeline version.
    if "pipeline_version" in df.columns:
        df = df[df["pipeline_version"].astype(str) == PIPELINE_VERSION]
    return {(int(r.seed), str(r.method)) for r in df.itertuples(index=False)}


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if path.exists() and path.stat().st_size > 0:
        df_new.to_csv(path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(path, index=False)


def _make_split(df: pd.DataFrame, split_mode: str, seed: int) -> dict[str, Any]:
    if split_mode == "well_id":
        return protocol1_random_wells(df, seed=seed, save=True)
    if split_mode == "parent_grouped":
        return protocol1_parent_grouped_wells(df, seed=seed, save=True)
    raise ValueError(f"unknown split_mode={split_mode}")


def run_split_mode(
    *,
    df: pd.DataFrame,
    curves: pd.DataFrame,
    split_mode: str,
    seeds: list[int],
    methods: list[str],
    config_path: Path,
    force: bool = False,
) -> None:
    csv_path = _csv_path(split_mode)
    det_dir = _details_dir(split_mode)
    if force and csv_path.exists():
        bak = csv_path.with_suffix(csv_path.suffix + f".preforce_{int(time.time())}.bak")
        csv_path.replace(bak)
        print(f"[force] moved prior CSV → {bak.name}", flush=True)
    done = set() if force else _load_done(csv_path)
    print(
        f"[multirep] mode={split_mode} seeds={len(seeds)} methods={len(methods)} "
        f"done={len(done)} out={csv_path} config={config_path.name}",
        flush=True,
    )

    regime_rows: list[dict[str, Any]] = []
    t_mode0 = time.time()
    for seed in seeds:
        split = _make_split(df, split_mode, seed)
        parts = apply_split(df, split)
        regime = _a_priori_regime(parts)
        regime_rows.append(
            {
                "split_mode": split_mode,
                "seed": seed,
                "protocol": split.get("protocol"),
                "n_train_wells": int(parts["train"][WELL_COL].nunique()),
                "n_val_wells": int(parts["val"][WELL_COL].nunique()),
                "n_test_wells": int(parts["test"][WELL_COL].nunique()),
                **regime,
            }
        )
        print(
            f"[seed] {split_mode} seed={seed} regime={regime['label_regime']} "
            f"n_unseen={regime['n_test_unseen_label_kept']}/{regime['n_test']}",
            flush=True,
        )
        set_curve_context(curves)
        set_run_meta(split=split, config_path=config_path)

        for method in methods:
            key = (seed, method)
            if key in done:
                print(f"[skip] {split_mode} seed={seed} method={method}", flush=True)
                continue
            t0 = time.time()
            try:
                res = run_method_on_split(parts, method, seed=seed)
                # Persist full details JSON per run (crash-safe; overwrite ok).
                det_path = det_dir / f"seed{seed}_{method}.json"
                det_path.write_text(
                    json.dumps(res, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
                row = {
                    "split_mode": split_mode,
                    "protocol": split.get("protocol"),
                    "seed": seed,
                    "method": method,
                    "label_regime": res.get("label_regime", regime["label_regime"]),
                    "a_priori_regime": regime["label_regime"],
                    "macro_f1": res["overall"]["macro_f1"],
                    "accuracy": res["overall"]["accuracy"],
                    "worst_well_f1": res["well_summary"]["worst_well_f1"],
                    "worst10pct_well_f1": res["well_summary"]["worst10pct_well_f1"],
                    "well_macro_f1_mean": res["well_summary"]["well_macro_f1_mean"],
                    "worst_well_f1_fixed": (res.get("well_summary_fixed") or {}).get(
                        "worst_well_f1", ""
                    ),
                    "worst10pct_well_f1_fixed": (res.get("well_summary_fixed") or {}).get(
                        "worst10pct_well_f1", ""
                    ),
                    "n_train": res["n_train"],
                    "n_val": res["n_val"],
                    "n_test": res["n_test"],
                    "n_test_unseen_label_kept": res.get("n_test_unseen_label_kept", 0),
                    "n_val_dropped_unseen_label": res.get("n_val_dropped_unseen_label", 0),
                    "n_val_dropped_unseen": res.get("n_val_dropped_unseen", 0),
                    "n_fixed_labels": res.get("n_fixed_labels", ""),
                    "train_class_names": "|".join(res.get("train_class_names") or []),
                    "seconds": res["seconds"],
                    "pipeline_version": res.get("pipeline_version", PIPELINE_VERSION),
                    "feature_mode": res.get("feature_mode", ""),
                    "use_relative_depth": res.get("use_relative_depth", ""),
                    "well_z_mode": res.get("well_z_mode", ""),
                    "stats_support": res.get("stats_support", ""),
                    "git_commit": res.get("git_commit", ""),
                    "git_dirty": res.get("git_dirty", ""),
                    "config_hash": res.get("config_hash", ""),
                    "config_path": res.get("config_path", str(config_path)),
                    "split_hash": res.get("split_hash", ""),
                    "feature_param_hash": res.get("feature_param_hash", ""),
                    "hash_labeled_parquet": res.get("hash_labeled_parquet", ""),
                    "hash_curves_all_parquet": res.get("hash_curves_all_parquet", ""),
                    "val_open_macro_f1": (res.get("val_open_diagnostics") or {}).get(
                        "macro_f1_including_unseen", ""
                    ),
                    "details_path": str(det_path.relative_to(ROOT)),
                    "wall_seconds": round(time.time() - t0, 2),
                }
                _append_row(csv_path, row)
                done.add(key)
                print(
                    f"[done] {split_mode} seed={seed} method={method} "
                    f"macro={row['macro_f1']:.4f} worst_fixed={row['worst_well_f1_fixed']} "
                    f"sec={row['seconds']}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                err_path = OUT_DIR / f"errors_{split_mode}.log"
                with err_path.open("a", encoding="utf-8") as fh:
                    fh.write(
                        f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S')}] "
                        f"seed={seed} method={method}: {exc}\n"
                    )
                    fh.write(traceback.format_exc())
                print(
                    f"[ERROR] {split_mode} seed={seed} method={method}: {exc}",
                    flush=True,
                )
                raise

    # Write / refresh a priori regime table for this mode.
    regime_path = OUT_DIR / f"protocol1_multirep_{split_mode}_regimes.csv"
    pd.DataFrame(regime_rows).to_csv(regime_path, index=False)
    # Also mirror under followup for paper sync convenience.
    FOLLOWUP.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(regime_rows).to_csv(
        FOLLOWUP / f"protocol1_multirep_{split_mode}_regimes.csv", index=False
    )
    elapsed = time.time() - t_mode0
    print(
        f"[multirep] finished mode={split_mode} in {elapsed/3600:.2f} h → {csv_path}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--split-mode",
        choices=["well_id", "parent_grouped", "both"],
        default="both",
    )
    ap.add_argument(
        "--seeds",
        default="",
        help="Comma-separated override; default PROTOCOL1_MULTIREP_SEEDS",
    )
    ap.add_argument(
        "--methods",
        default="",
        help="Comma-separated override; default PROTOCOL1_CORE_METHODS",
    )
    ap.add_argument("--force", action="store_true", help="Ignore resume CSV")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FOLLOWUP.mkdir(parents=True, exist_ok=True)

    if not MULTIREP_CONFIG.exists():
        raise FileNotFoundError(f"missing multirep config: {MULTIREP_CONFIG}")
    config_path = MULTIREP_CONFIG.resolve()

    git_snap = freeze_git_provenance(ignore_outputs=True)
    print(
        f"[provenance] pipeline={PIPELINE_VERSION} "
        f"git={git_snap.get('git_commit', '')[:12]}… "
        f"dirty={git_snap.get('git_dirty')} "
        f"config={config_path.name}",
        flush=True,
    )

    seeds = (
        [int(x) for x in args.seeds.split(",") if x.strip()]
        if args.seeds.strip()
        else list(PROTOCOL1_MULTIREP_SEEDS)
    )
    methods = (
        [m.strip() for m in args.methods.split(",") if m.strip()]
        if args.methods.strip()
        else list(PROTOCOL1_CORE_METHODS)
    )

    meta = {
        "pipeline_version": PIPELINE_VERSION,
        "seeds": seeds,
        "methods": methods,
        "config_path": str(config_path.relative_to(ROOT)),
        "closed_set_rule": (
            "a_priori: test has zero depths with facies absent from training"
        ),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_snap.get("git_commit", ""),
        "git_dirty": git_snap.get("git_dirty", True),
    }
    (OUT_DIR / "protocol1_multirep_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print("[data] loading labeled + curves_all …", flush=True)
    df = load_labeled()
    curves = load_curves_all()
    print(f"[data] labeled={len(df):,} wells={df[WELL_COL].nunique()}", flush=True)

    modes = (
        ["well_id", "parent_grouped"]
        if args.split_mode == "both"
        else [args.split_mode]
    )
    for mode in modes:
        run_split_mode(
            df=df,
            curves=curves,
            split_mode=mode,
            seeds=seeds,
            methods=methods,
            config_path=config_path,
            force=bool(args.force),
        )

    # Final regime summary across modes.
    frames = []
    for mode in modes:
        p = OUT_DIR / f"protocol1_multirep_{mode}_regimes.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if frames:
        all_reg = pd.concat(frames, ignore_index=True)
        all_reg.to_csv(OUT_DIR / "protocol1_multirep_all_regimes.csv", index=False)
        summary = (
            all_reg.groupby(["split_mode", "label_regime"])
            .size()
            .reset_index(name="n_seeds")
        )
        summary.to_csv(OUT_DIR / "protocol1_multirep_regime_counts.csv", index=False)
        print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
