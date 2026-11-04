"""Protocol-1 open-set diagnostics (seed 44) + Hybrid-norel−WellNorm bootstrap.

Preferred same-budget contrast is Hybrid-norel vs WellNorm (not Full Hybrid).
Optional companions: Full Hybrid−WellNorm and the 2x2 cell Hybrid-norel−WellNorm-norel.

Outputs under results/followup/:
  - protocol1_openset_knownclass.csv
  - protocol1_hybrid_minus_wellnorm_paired.csv
  - protocol1_hybrid_wellnorm_bootstrap.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import RESULTS_DIR
from src.train import PIPELINE_VERSION

OUT = RESULTS_DIR / "followup"
OUT.mkdir(parents=True, exist_ok=True)

# Preferred composition contrast highlighted in the manuscript.
HYBRID_METHOD = "lgbm_hybrid_norel"
WELLNORM_METHOD = "lgbm_wellnorm"


def _load_p1() -> pd.DataFrame:
    files = sorted(RESULTS_DIR.glob("summary_protocol1_random_wells_seed*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "pipeline_version" in df.columns:
        df = df[df["pipeline_version"].astype(str) == PIPELINE_VERSION]
    return df


def openset_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in sorted(df["seed"].unique()):
        sub = df[df["seed"] == seed]
        for _, r in sub.iterrows():
            rows.append(
                {
                    "seed": int(seed),
                    "method": r["method"],
                    "macro_f1": r.get("macro_f1"),
                    "worst_well_f1": r.get("worst_well_f1"),
                    "worst10pct_well_f1": r.get("worst10pct_well_f1"),
                    "n_test_unseen_label_kept": r.get("n_test_unseen_label_kept", 0),
                    "n_val_dropped_unseen_label": r.get("n_val_dropped_unseen_label", 0),
                    "known_macro_f1": r.get("known_macro_f1", ""),
                    "known_worst_well_f1": r.get("known_worst_well_f1", ""),
                    "known_worst10pct_well_f1": r.get("known_worst10pct_well_f1", ""),
                    "pipeline_version": r.get("pipeline_version", ""),
                }
            )
    return pd.DataFrame(rows)


def paired_contrast(
    df: pd.DataFrame,
    left: str,
    right: str,
    *,
    closed_only: bool = False,
) -> pd.DataFrame:
    rows = []
    seeds = [42, 43, 45] if closed_only else sorted(df["seed"].unique())
    for seed in seeds:
        sub = df[df["seed"] == seed].set_index("method")
        if left not in sub.index or right not in sub.index:
            continue
        h, w = sub.loc[left], sub.loc[right]
        g = sub.loc["lgbm_global"] if "lgbm_global" in sub.index else None
        row = {
            "seed": int(seed),
            "left_method": left,
            "right_method": right,
            "hybrid_macro": float(h["macro_f1"]),
            "wellnorm_macro": float(w["macro_f1"]),
            "delta_macro_h_wn": float(h["macro_f1"] - w["macro_f1"]),
            "hybrid_worst": float(h["worst_well_f1"]),
            "wellnorm_worst": float(w["worst_well_f1"]),
            "delta_worst_h_wn": float(h["worst_well_f1"] - w["worst_well_f1"]),
            "hybrid_w10": float(h["worst10pct_well_f1"]),
            "wellnorm_w10": float(w["worst10pct_well_f1"]),
            "delta_w10_h_wn": float(h["worst10pct_well_f1"] - w["worst10pct_well_f1"]),
        }
        if g is not None:
            row.update(
                {
                    "delta_macro_h_global": float(h["macro_f1"] - g["macro_f1"]),
                    "delta_worst_h_global": float(h["worst_well_f1"] - g["worst_well_f1"]),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


# Backward-compatible name used by older docs.
def paired_hybrid_wellnorm(df: pd.DataFrame) -> pd.DataFrame:
    return paired_contrast(df, HYBRID_METHOD, WELLNORM_METHOD, closed_only=False)


def seed_level_bootstrap_ci(
    deltas: np.ndarray, n_boot: int = 5000, seed: int = 0, alpha: float = 0.05
) -> dict:
    """Seed-level bootstrap of mean Δ (resamples the n seed deltas only).

    This is *not* a hierarchical / well-block bootstrap: there is no within-seed
    resampling of wells or depth blocks. With n=4 the CI width should be read
    cautiously and not over-interpreted to three-decimal precision.
    """
    rng = np.random.default_rng(seed)
    deltas = np.asarray(deltas, dtype=float)
    if len(deltas) == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(deltas, size=len(deltas), replace=True)
        boots.append(float(sample.mean()))
    boots = np.sort(boots)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return {
        "mean": float(deltas.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "n": int(len(deltas)),
        "n_boot": n_boot,
        "alpha": alpha,
        "bootstrap_kind": "seed_level",
    }


# Backward-compatible alias (name was historically misleading).
hierarchical_bootstrap_ci = seed_level_bootstrap_ci


def _boot_block(paired: pd.DataFrame, tag: str) -> dict:
    if paired.empty:
        return {"tag": tag, "n": 0}
    out = {
        "tag": tag,
        "left_method": paired["left_method"].iloc[0],
        "right_method": paired["right_method"].iloc[0],
        "delta_macro_h_wn": seed_level_bootstrap_ci(paired["delta_macro_h_wn"].to_numpy()),
        "delta_worst_h_wn": seed_level_bootstrap_ci(paired["delta_worst_h_wn"].to_numpy()),
        "delta_w10_h_wn": seed_level_bootstrap_ci(paired["delta_w10_h_wn"].to_numpy()),
    }
    if "delta_worst_h_global" in paired.columns:
        out["delta_worst_h_global"] = seed_level_bootstrap_ci(
            paired["delta_worst_h_global"].to_numpy()
        )
    return out


def main() -> None:
    df = _load_p1()
    if df.empty:
        print(f"No Protocol-1 rows for {PIPELINE_VERSION}; run paper_r3_release first.")
        return
    ot = openset_table(df)
    ot.to_csv(OUT / "protocol1_openset_knownclass.csv", index=False)

    paired_all = paired_contrast(df, HYBRID_METHOD, WELLNORM_METHOD, closed_only=False)
    paired_closed = paired_contrast(df, HYBRID_METHOD, WELLNORM_METHOD, closed_only=True)
    paired_full = paired_contrast(df, "lgbm_hybrid", WELLNORM_METHOD, closed_only=False)
    paired_2x2 = paired_contrast(
        df, HYBRID_METHOD, "lgbm_wellnorm_norel", closed_only=True
    )

    paired_all.to_csv(OUT / "protocol1_hybrid_minus_wellnorm_paired.csv", index=False)
    paired_closed.to_csv(
        OUT / "protocol1_hybrid_norel_wellnorm_paired_closed.csv", index=False
    )

    boot = {
        "pipeline_version": PIPELINE_VERSION,
        "preferred_contrast": f"{HYBRID_METHOD}-{WELLNORM_METHOD}",
        "note": (
            "Primary multi-seed means in the manuscript use closed-set seeds 42/43/45; "
            "seed 44 is an open-set stress with no rejection mechanism."
        ),
        "hybrid_norel_minus_wellnorm_all4": _boot_block(paired_all, "all4"),
        "hybrid_norel_minus_wellnorm_closed": _boot_block(paired_closed, "closed"),
        "full_hybrid_minus_wellnorm_all4": _boot_block(paired_full, "full_hybrid_all4"),
        "hybrid_norel_minus_wellnorm_norel_closed": _boot_block(paired_2x2, "2x2_closed"),
        # Keep legacy keys pointing at the preferred contrast (all-seed) for old readers.
        "delta_macro_h_wn": seed_level_bootstrap_ci(
            paired_all["delta_macro_h_wn"].to_numpy()
        )
        if not paired_all.empty
        else {},
        "delta_worst_h_wn": seed_level_bootstrap_ci(
            paired_all["delta_worst_h_wn"].to_numpy()
        )
        if not paired_all.empty
        else {},
        "delta_w10_h_wn": seed_level_bootstrap_ci(paired_all["delta_w10_h_wn"].to_numpy())
        if not paired_all.empty
        else {},
    }
    (OUT / "protocol1_hybrid_wellnorm_bootstrap.json").write_text(
        json.dumps(boot, indent=2), encoding="utf-8"
    )
    print(paired_all.to_string(index=False))
    print(json.dumps(boot, indent=2))


if __name__ == "__main__":
    main()
