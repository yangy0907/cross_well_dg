"""Post-hoc C_common Macro + per-class + well-level F1 figure (closed seeds).

Uses existing details JSON for seeds 42/43/45 (no retrain).

Outputs under results/followup/:
  - protocol1_ccommon_macro_closed.csv
  - protocol1_ccommon_per_class_closed.csv
  - per_class_f1_protocol1_closed_seeds.csv  (legacy + filtered micro avg)
  - fig_followup_well_f1_distribution.png / .pdf

C_common = intersection of train-learnable *class names* across analysis splits.
Reports (i) legacy per-split pooled Macro, (ii) Macro over C_common (missing
class F1=0), (iii) per-class F1 table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import (  # noqa: E402
    LITHOLOGY_MAP,
    PROTOCOL1_CLOSED_SEEDS,
    PROTOCOL1_CORE_METHODS,
    RESULTS_DIR,
)

OUT = RESULTS_DIR / "followup"
FIGS = RESULTS_DIR / "paper_figures"
PAPER_FIGS = ROOT / "paper" / "gse" / "figs"

_SKIP = {"accuracy", "macro avg", "weighted avg", "micro avg", "samples avg"}

METHOD_LABELS = {
    "lgbm_global": "Global",
    "lgbm_wellnorm": "WellNorm",
    "lgbm_hybrid_norel": "Hybrid-norel",
    "lgbm_hybrid": "Hybrid",
    "lgbm_hybrid_while": "Hybrid-while",
    "lgbm_hybrid_smooth": "Hybrid+Smooth",
}


def _load_details(seed: int) -> list[dict]:
    path = RESULTS_DIR / f"details_protocol1_random_wells_seed{seed}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _train_class_names(rec: dict) -> set[str]:
    if rec.get("train_class_names"):
        return set(rec["train_class_names"])
    names = set()
    for code_s in rec.get("label_mapping", {}):
        code = int(code_s)
        names.add(LITHOLOGY_MAP.get(code, str(code)))
    return names


def _filter_per_class(pc: dict) -> dict:
    return {k: v for k, v in pc.items() if k not in _SKIP and isinstance(v, dict)}


def _ccommon_macro_from_per_class(pc: dict, c_common: list[str]) -> float:
    """Mean F1 over C_common; missing class → 0."""
    f1s = []
    for name in c_common:
        if name in pc and "f1" in pc[name]:
            f1s.append(float(pc[name]["f1"]))
        else:
            f1s.append(0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)

    # --- C_common from train-learnable names across closed seeds (any core method) ---
    train_sets: dict[int, set[str]] = {}
    for seed in PROTOCOL1_CLOSED_SEEDS:
        rows = _load_details(seed)
        # Prefer Global; any method shares train mapping within a split.
        rec = next(r for r in rows if r["method"] == "lgbm_global")
        train_sets[seed] = _train_class_names(rec)
        assert int(rec.get("n_test_unseen_label_kept", 0)) == 0, seed

    c_common = sorted(set.intersection(*train_sets.values()))
    print("C_common (", len(c_common), "):", c_common)
    for seed, s in train_sets.items():
        print(f"  seed {seed} train classes ({len(s)}):", sorted(s))
        print(f"  seed {seed} only:", sorted(s - set(c_common)))

    macro_rows = []
    per_class_rows = []
    well_rows = []

    for seed in PROTOCOL1_CLOSED_SEEDS:
        rows = _load_details(seed)
        for rec in rows:
            method = rec["method"]
            if method not in PROTOCOL1_CORE_METHODS:
                continue
            if not str(rec.get("pipeline_version", "")).startswith("r3_"):
                raise SystemExit(f"non-r3: seed={seed} method={method}")
            pc = _filter_per_class(rec.get("per_class") or {})
            legacy_macro = float(rec["overall"]["macro_f1"])
            c_macro = _ccommon_macro_from_per_class(pc, c_common)
            worst_fixed = float(
                (rec.get("well_summary_fixed") or {}).get("worst_well_f1")
                or float("nan")
            )
            macro_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "macro_f1_legacy": legacy_macro,
                    "macro_f1_ccommon": c_macro,
                    "worst_well_f1_fixed": worst_fixed,
                    "n_ccommon": len(c_common),
                    "n_train_classes": len(_train_class_names(rec)),
                    "n_per_class_reported": len(pc),
                }
            )
            for cls_name in c_common:
                stats = pc.get(cls_name) or {}
                code = next(
                    (c for c, n in LITHOLOGY_MAP.items() if n == cls_name),
                    -1,
                )
                per_class_rows.append(
                    {
                        "seed": seed,
                        "method": METHOD_LABELS.get(method, method),
                        "method_key": method,
                        "code": code,
                        "class": cls_name,
                        "support": int(stats.get("support", 0)),
                        "f1": float(stats.get("f1", 0.0)),
                        "macro_f1_legacy": legacy_macro,
                        "macro_f1_ccommon": c_macro,
                        "in_split_report": cls_name in pc,
                    }
                )
            # well-level F1 (legacy well_true table)
            for wrow in rec.get("well_table") or []:
                well_rows.append(
                    {
                        "seed": seed,
                        "method": METHOD_LABELS.get(method, method),
                        "method_key": method,
                        "well_id": wrow["well_id"],
                        "n": wrow["n"],
                        "macro_f1": float(wrow["macro_f1"]),
                        "taxonomy": "well_true",
                    }
                )
            for wrow in rec.get("well_table_fixed") or []:
                well_rows.append(
                    {
                        "seed": seed,
                        "method": METHOD_LABELS.get(method, method),
                        "method_key": method,
                        "well_id": wrow["well_id"],
                        "n": wrow["n"],
                        "macro_f1": float(wrow["macro_f1"]),
                        "taxonomy": "fixed",
                    }
                )

    macro_df = pd.DataFrame(macro_rows)
    pc_df = pd.DataFrame(per_class_rows)
    well_df = pd.DataFrame(well_rows)

    macro_path = OUT / "protocol1_ccommon_macro_closed.csv"
    pc_path = OUT / "protocol1_ccommon_per_class_closed.csv"
    # Also refresh legacy per_class export without micro avg leak.
    legacy_pc = pc_df[pc_df["seed"] == 42][
        ["method", "code", "class", "support", "f1", "macro_f1_legacy"]
    ].rename(columns={"macro_f1_legacy": "macro_f1"})
    legacy_pc.to_csv(OUT / "per_class_f1_protocol1_seed42.csv", index=False)
    pc_df.to_csv(OUT / "per_class_f1_protocol1_closed_seeds.csv", index=False)
    macro_df.to_csv(macro_path, index=False)
    pc_df.to_csv(pc_path, index=False)

    # Mean±std over closed seeds
    agg = (
        macro_df.groupby("method", as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            macro_legacy_mean=("macro_f1_legacy", "mean"),
            macro_legacy_std=("macro_f1_legacy", "std"),
            macro_ccommon_mean=("macro_f1_ccommon", "mean"),
            macro_ccommon_std=("macro_f1_ccommon", "std"),
            worst_fixed_mean=("worst_well_f1_fixed", "mean"),
            worst_fixed_std=("worst_well_f1_fixed", "std"),
        )
        .sort_values("macro_legacy_mean", ascending=False)
    )
    agg.to_csv(OUT / "protocol1_ccommon_macro_closed_summary.csv", index=False)
    print(agg.to_string(index=False))

    meta = {
        "c_common": c_common,
        "closed_seeds": list(PROTOCOL1_CLOSED_SEEDS),
        "train_sets": {str(k): sorted(v) for k, v in train_sets.items()},
        "note": (
            "C_common = intersection of train-learnable class names across "
            "closed analysis splits; missing class F1=0 in Macro_Ccommon"
        ),
    }
    (OUT / "protocol1_ccommon_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    # --- Well-level F1 distribution figure (closed seeds, well_true) ---
    plot_df = well_df[well_df["taxonomy"] == "well_true"].copy()
    if plot_df.empty:
        # Reconstruct from well_table only (fixed may be absent in old details)
        plot_df = well_df.copy()
    order = [
        METHOD_LABELS[m]
        for m in PROTOCOL1_CORE_METHODS
        if METHOD_LABELS[m] in set(plot_df["method"])
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    data = [plot_df.loc[plot_df["method"] == m, "macro_f1"].values for m in order]
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor("#4C78A8")
        b.set_alpha(0.55)
    # overlay box
    ax.boxplot(
        data,
        widths=0.18,
        patch_artist=True,
        boxprops=dict(facecolor="white", alpha=0.85),
        medianprops=dict(color="#E45756", linewidth=1.5),
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
    )
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("Per-well Macro-F1 (well_true)")
    ax.set_title(
        "Protocol-1 closed seeds 42/43/45 — well-level F1 distribution (core 6)"
    )
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.0, color="0.7", lw=0.6)
    fig.tight_layout()
    for dest in (OUT, FIGS, PAPER_FIGS):
        fig.savefig(dest / "fig_followup_well_f1_distribution.png", dpi=200)
        fig.savefig(dest / "fig_followup_well_f1_distribution.pdf")
    plt.close(fig)
    well_df.to_csv(OUT / "protocol1_well_f1_closed_long.csv", index=False)
    print("wrote", macro_path)
    print("wrote fig_followup_well_f1_distribution.png")


if __name__ == "__main__":
    main()
