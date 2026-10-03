"""Build machine-readable + LaTeX-ready minimum acceptance records from CSVs.

Primary record = Protocol-1 closed multi-rep means (well-id; n=18 from the locked
20-seed list). Parent-grouped companion means are noted in metadata only.
A historical three-seed (42/43/45) sidecar is written for SI contrast.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import (  # noqa: E402
    PROTOCOL1_CLOSED_SEEDS,
    PROTOCOL1_CORE_METHODS,
    RESULTS_DIR,
)
from src.train import PIPELINE_VERSION  # noqa: E402

OUT = RESULTS_DIR / "followup"
OUT.mkdir(parents=True, exist_ok=True)
MULTIREP = RESULTS_DIR / "multirep"

DISPLAY = {
    "lgbm_global": ("Inductive DG", "None", "Global-LGBM"),
    "lgbm_wellnorm": ("Test-domain (post)", "whole-well z + rel depth", "WellNorm"),
    "lgbm_wellnorm_norel": (
        "Test-domain (post)",
        "whole-well z (no rel depth)",
        "WellNorm-norel",
    ),
    "lgbm_hybrid_norel": (
        "Test-domain (post)",
        "global + whole-well z (no rel depth)",
        "Hybrid-norel",
    ),
    "lgbm_hybrid": (
        "Test-domain (post)",
        "global + whole-well z + rel depth",
        "Hybrid",
    ),
    "lgbm_hybrid_while": (
        "Test-domain (causal-prefix)",
        "global + causal prefix z",
        "Hybrid-while",
    ),
}

# Primary acceptance methods = multi-rep core family minus score-support diagnostic.
PRIMARY_METHODS = [
    m for m in PROTOCOL1_CORE_METHODS if m != "lgbm_hybrid_smooth"
]

CLOSED_SEEDS_HIST = list(PROTOCOL1_CLOSED_SEEDS)  # 42, 43, 45


def _pm(x: float) -> float:
    return float(f"{float(x):.6g}")


def load_p1() -> pd.DataFrame:
    files = sorted(RESULTS_DIR.glob("summary_protocol1_random_wells_seed*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df[df["pipeline_version"].astype(str) == PIPELINE_VERSION].copy()


def _build_multirep_primary() -> list[dict]:
    summary_path = OUT / "protocol1_multirep_well_id_closed_summary.csv"
    long_path = OUT / "protocol1_multirep_well_id_closed_long.csv"
    core_path = MULTIREP / "protocol1_multirep_well_id_core6.csv"
    meta_path = OUT / "protocol1_multirep_well_id_meta.json"
    if not summary_path.exists():
        print(f"FAIL: missing {summary_path}", file=sys.stderr)
        return []
    summary = pd.read_csv(summary_path)
    if summary.empty:
        print("FAIL: empty well-id closed summary", file=sys.stderr)
        return []

    n_train_mean = None
    if long_path.exists():
        long_df = pd.read_csv(long_path)
        if "n_train" in long_df.columns and not long_df.empty:
            n_train_mean = float(long_df["n_train"].mean())
    elif core_path.exists():
        core = pd.read_csv(core_path)
        closed = core[core["a_priori_regime"].astype(str) == "closed_set"]
        if not closed.empty and "n_train" in closed.columns:
            n_train_mean = float(closed["n_train"].mean())

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    closed_seeds = meta.get("closed_seeds", [])
    open_seeds = meta.get("open_seeds", [44, 127])

    parent_note = ""
    parent_sum = OUT / "protocol1_multirep_parent_grouped_closed_summary.csv"
    if parent_sum.exists():
        parent_note = (
            "Parent-grouped co-primary companion: "
            "protocol1_multirep_parent_grouped_closed_summary.csv"
        )

    rows = []
    for m in PRIMARY_METHODS:
        sub = summary[summary["method"] == m]
        if len(sub) != 1:
            print(
                f"FAIL: method {m} expected 1 closed-summary row, got {len(sub)}",
                file=sys.stderr,
            )
            return []
        r = sub.iloc[0]
        n_closed = int(r["n_closed_reps"])
        if n_closed != 18:
            print(
                f"FAIL: method {m} n_closed_reps={n_closed}, required 18",
                file=sys.stderr,
            )
            return []
        budget, stats, name = DISPLAY.get(m, ("?", "?", m))
        rows.append(
            {
                "method": m,
                "display_name": name,
                "information_budget": budget,
                "target_statistics": stats,
                "split_protocol": "Protocol-1",
                "split_mode_primary": "well_id",
                "primary_seeds": (
                    f"multi-rep closed n=18 "
                    f"(locked list; open {','.join(str(s) for s in open_seeds)} excluded)"
                ),
                "closed_seeds": ",".join(str(s) for s in closed_seeds),
                "open_set_seeds": ",".join(str(s) for s in open_seeds),
                "n_closed_reps": n_closed,
                "n_train_mean_closed": n_train_mean,
                "macro_f1_mean_closed": _pm(r["macro_legacy_mean"]),
                "worst_well_f1_fixed_mean_closed": _pm(r["worst_fixed_mean"]),
                "w10_fixed_mean_closed": _pm(r["w10_fixed_mean"]),
                "macro_f1_std_closed": _pm(r["macro_legacy_std"]),
                "worst_well_f1_fixed_std_closed": _pm(r["worst_fixed_std"]),
                "w10_fixed_std_closed": _pm(r["w10_fixed_std"]),
                "primary_tail_metric": "fixed_taxonomy (train_learnable)",
                "legacy_tail_metric": "well_true (sensitivity; not in this record)",
                "parent_grouped_companion": parent_note,
                "depth_point_as_deployment_evidence": "No",
                "pipeline_version": PIPELINE_VERSION,
                "record_scope": "primary_multirep_n18_well_id",
            }
        )
    return rows


def _build_historical_3seed(df: pd.DataFrame) -> list[dict]:
    """Historical seeds 42/43/45 contrast (not the primary release record)."""
    if df.empty:
        return []
    hist_methods = [
        "lgbm_global",
        "lgbm_wellnorm_norel",
        "lgbm_wellnorm",
        "lgbm_hybrid_norel",
        "lgbm_hybrid",
        "lgbm_hybrid_while",
    ]
    closed = df[df["seed"].isin(CLOSED_SEEDS_HIST)]
    rows = []
    for m in hist_methods:
        sub_c = closed[closed["method"] == m]
        seeds_present = sorted({int(s) for s in sub_c["seed"].tolist()}) if not sub_c.empty else []
        if seeds_present != CLOSED_SEEDS_HIST or len(sub_c) != len(CLOSED_SEEDS_HIST):
            print(
                f"WARN: historical 3-seed skip {m}: seeds={seeds_present}",
                file=sys.stderr,
            )
            continue
        budget, stats, name = DISPLAY.get(m, ("?", "?", m))
        rows.append(
            {
                "method": m,
                "display_name": name,
                "information_budget": budget,
                "target_statistics": stats,
                "split_protocol": "Protocol-1",
                "primary_seeds": "42,43,45 (historical closed-set contrast)",
                "open_set_seed": "44 (stress; not in mean)",
                "n_train_mean_closed": float(sub_c["n_train"].mean()),
                "macro_f1_mean_closed": float(sub_c["macro_f1"].mean()),
                "worst_well_f1_fixed_mean_closed": (
                    float(sub_c["worst_well_f1_fixed"].mean())
                    if "worst_well_f1_fixed" in sub_c.columns
                    else None
                ),
                "w10_fixed_mean_closed": (
                    float(sub_c["worst10pct_well_f1_fixed"].mean())
                    if "worst10pct_well_f1_fixed" in sub_c.columns
                    else None
                ),
                "record_scope": "historical_3seed_contrast",
                "pipeline_version": PIPELINE_VERSION,
            }
        )
    return rows


def main() -> int:
    rows = _build_multirep_primary()
    if not rows:
        return 1
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "acceptance_record_protocol1.csv", index=False)
    (OUT / "acceptance_record_protocol1.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(out.to_string(index=False))
    print(f"wrote {OUT / 'acceptance_record_protocol1.csv'}")

    hist = _build_historical_3seed(load_p1())
    if hist:
        hist_df = pd.DataFrame(hist)
        hist_path = OUT / "acceptance_record_protocol1_historical_3seed.csv"
        hist_df.to_csv(hist_path, index=False)
        print(f"wrote {hist_path} (SI contrast only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
