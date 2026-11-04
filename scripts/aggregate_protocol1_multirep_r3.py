"""Aggregate Protocol-1 multi-rep core-6 results after overnight runs.

Builds closed-set mean±std (legacy Macro, C_common Macro, Worst-fixed, W10),
paired HN−Global / Hybrid−HN deltas, C_common class appearance counts,
writes followup CSVs and a Pareto scatter figure. Does not retrain.

Usage::
    py scripts/aggregate_protocol1_multirep_r3.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import (  # noqa: E402
    LITHOLOGY_MAP,
    PROTOCOL1_CORE_METHODS,
    PROTOCOL1_MULTIREP_SEEDS,
    RESULTS_DIR,
)
from src.train import PIPELINE_VERSION  # noqa: E402

OUT = RESULTS_DIR / "multirep"
FOLLOWUP = RESULTS_DIR / "followup"
FIGS = RESULTS_DIR / "paper_figures"
PAPER_FIGS = ROOT / "paper" / "gse" / "figs"

METHOD_LABELS = {
    "lgbm_global": "Global",
    "lgbm_wellnorm": "WellNorm",
    "lgbm_hybrid_norel": "Hybrid-norel",
    "lgbm_hybrid": "Hybrid",
    "lgbm_hybrid_while": "Hybrid-while",
    "lgbm_hybrid_smooth": "Hybrid+Smooth",
}
_SKIP = {"accuracy", "macro avg", "weighted avg", "micro avg", "samples avg"}

def _ccommon_from_details(split_mode: str, closed_seeds: list[int]) -> list[str]:
    det_dir = OUT / "details" / split_mode
    train_sets: list[set[str]] = []
    for seed in closed_seeds:
        path = det_dir / f"seed{seed}_lgbm_global.json"
        if not path.exists():
            cands = list(det_dir.glob(f"seed{seed}_*.json"))
            if not cands:
                continue
            path = cands[0]
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("train_class_names"):
            train_sets.append(set(rec["train_class_names"]))
        else:
            names = {
                LITHOLOGY_MAP.get(int(k), str(k)) for k in rec.get("label_mapping", {})
            }
            train_sets.append(names)
    if not train_sets:
        return []
    return sorted(set.intersection(*train_sets))


def _macro_ccommon(rec: dict, c_common: list[str]) -> float:
    pc = {
        k: v
        for k, v in (rec.get("per_class") or {}).items()
        if k not in _SKIP and isinstance(v, dict)
    }
    f1s = [float(pc[c]["f1"]) if c in pc and "f1" in pc[c] else 0.0 for c in c_common]
    return float(np.mean(f1s)) if f1s else float("nan")


def _macro_ccommon_supported(rec: dict, c_common: list[str]) -> float:
    """C_common restricted to classes with test support > 0 in this split."""
    pc = {
        k: v
        for k, v in (rec.get("per_class") or {}).items()
        if k not in _SKIP and isinstance(v, dict)
    }
    f1s = []
    for c in c_common:
        if c not in pc:
            continue
        support = float(pc[c].get("support", 0) or 0)
        if support <= 0:
            continue
        f1s.append(float(pc[c].get("f1", 0.0)))
    return float(np.mean(f1s)) if f1s else float("nan")


def _class_appearance(
    split_mode: str, closed_seeds: list[int], c_common: list[str]
) -> pd.DataFrame:
    det_dir = OUT / "details" / split_mode
    test_counts: Counter[str] = Counter()
    for seed in closed_seeds:
        path = det_dir / f"seed{seed}_lgbm_global.json"
        if not path.exists():
            cands = list(det_dir.glob(f"seed{seed}_*.json"))
            if not cands:
                continue
            path = cands[0]
        rec = json.loads(path.read_text(encoding="utf-8"))
        pc = {
            k: v
            for k, v in (rec.get("per_class") or {}).items()
            if k not in _SKIP and isinstance(v, dict)
        }
        for name in c_common:
            if name in pc and float(pc[name].get("support", 0) or 0) > 0:
                test_counts[name] += 1
    rows = [
        {
            "split_mode": split_mode,
            "class_name": name,
            "n_closed_splits_with_test_support": int(test_counts.get(name, 0)),
            "n_closed_splits": len(closed_seeds),
        }
        for name in c_common
    ]
    return pd.DataFrame(rows)


def _paired_deltas(closed: pd.DataFrame, split_mode: str) -> pd.DataFrame:
    pairs = [
        ("lgbm_hybrid_norel", "lgbm_global", "HN-Global"),
        ("lgbm_hybrid", "lgbm_hybrid_norel", "Hybrid-HN"),
    ]
    rows = []
    for left, right, name in pairs:
        d_macro: list[float] = []
        d_worst: list[float] = []
        for seed in sorted(closed["seed"].unique()):
            a = closed[(closed["seed"] == seed) & (closed["method"] == left)]
            b = closed[(closed["seed"] == seed) & (closed["method"] == right)]
            if a.empty or b.empty:
                continue
            d_macro.append(float(a["macro_f1"].iloc[0] - b["macro_f1"].iloc[0]))
            wa = float(
                pd.to_numeric(a["worst_well_f1_fixed"], errors="coerce").iloc[0]
            )
            wb = float(
                pd.to_numeric(b["worst_well_f1_fixed"], errors="coerce").iloc[0]
            )
            d_worst.append(wa - wb)
        if not d_macro:
            continue
        dm = np.asarray(d_macro, dtype=float)
        dw = np.asarray(d_worst, dtype=float)
        rows.append(
            {
                "split_mode": split_mode,
                "contrast": name,
                "left": left,
                "right": right,
                "n_closed_reps": len(dm),
                "delta_macro_mean": float(dm.mean()),
                "delta_macro_std": float(dm.std(ddof=1)) if len(dm) > 1 else 0.0,
                "delta_macro_wins": int((dm > 0).sum()),
                "delta_worst_fixed_mean": float(dw.mean()),
                "delta_worst_fixed_std": float(dw.std(ddof=1)) if len(dw) > 1 else 0.0,
                "delta_worst_fixed_wins": int((dw > 0).sum()),
                "note": (
                    "Overlapping Protocol-1 reps; sample std is split sensitivity, "
                    "not population SE"
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_mode(split_mode: str) -> pd.DataFrame | None:
    csv_path = OUT / f"protocol1_multirep_{split_mode}_core6.csv"
    if not csv_path.exists():
        print("missing", csv_path)
        return None
    df = pd.read_csv(csv_path)
    df = df[df["pipeline_version"].astype(str) == PIPELINE_VERSION].copy()
    regime_col = "a_priori_regime" if "a_priori_regime" in df.columns else "label_regime"
    closed_seeds = sorted(
        int(s) for s in df.loc[df[regime_col] == "closed_set", "seed"].unique()
    )
    open_seeds = sorted(
        int(s) for s in df.loc[df[regime_col] == "open_set", "seed"].unique()
    )
    print(f"[{split_mode}] closed={closed_seeds} open={open_seeds}")

    c_common = _ccommon_from_details(split_mode, closed_seeds)
    print(f"[{split_mode}] C_common ({len(c_common)}):", c_common)

    c_macros = []
    c_macros_sup = []
    det_dir = OUT / "details" / split_mode
    for r in df.itertuples(index=False):
        path = det_dir / f"seed{int(r.seed)}_{r.method}.json"
        if path.exists() and c_common:
            rec = json.loads(path.read_text(encoding="utf-8"))
            c_macros.append(_macro_ccommon(rec, c_common))
            c_macros_sup.append(_macro_ccommon_supported(rec, c_common))
        else:
            c_macros.append(np.nan)
            c_macros_sup.append(np.nan)
    df["macro_f1_ccommon"] = c_macros
    df["macro_f1_ccommon_supported"] = c_macros_sup

    closed = df[df[regime_col] == "closed_set"].copy()
    rows = []
    for method in PROTOCOL1_CORE_METHODS:
        sub = closed[closed["method"] == method]
        if sub.empty:
            continue
        rows.append(
            {
                "split_mode": split_mode,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "n_closed_reps": len(sub),
                "macro_legacy_mean": float(sub["macro_f1"].mean()),
                "macro_legacy_std": float(sub["macro_f1"].std(ddof=1))
                if len(sub) > 1
                else 0.0,
                "macro_ccommon_mean": float(sub["macro_f1_ccommon"].mean()),
                "macro_ccommon_std": float(sub["macro_f1_ccommon"].std(ddof=1))
                if len(sub) > 1
                else 0.0,
                "macro_ccommon_supported_mean": float(
                    sub["macro_f1_ccommon_supported"].mean()
                ),
                "macro_ccommon_supported_std": float(
                    sub["macro_f1_ccommon_supported"].std(ddof=1)
                )
                if len(sub) > 1
                else 0.0,
                "worst_fixed_mean": float(
                    pd.to_numeric(sub["worst_well_f1_fixed"], errors="coerce").mean()
                ),
                "worst_fixed_std": float(
                    pd.to_numeric(sub["worst_well_f1_fixed"], errors="coerce").std(
                        ddof=1
                    )
                )
                if len(sub) > 1
                else 0.0,
                "w10_fixed_mean": float(
                    pd.to_numeric(sub["worst10pct_well_f1_fixed"], errors="coerce").mean()
                ),
                "w10_fixed_std": float(
                    pd.to_numeric(sub["worst10pct_well_f1_fixed"], errors="coerce").std(
                        ddof=1
                    )
                )
                if len(sub) > 1
                else 0.0,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(
        FOLLOWUP / f"protocol1_multirep_{split_mode}_closed_summary.csv", index=False
    )
    closed.to_csv(
        FOLLOWUP / f"protocol1_multirep_{split_mode}_closed_long.csv", index=False
    )

    appear = _class_appearance(split_mode, closed_seeds, c_common)
    appear.to_csv(
        FOLLOWUP / f"protocol1_multirep_{split_mode}_ccommon_appearance.csv",
        index=False,
    )
    paired = _paired_deltas(closed, split_mode)
    paired.to_csv(
        FOLLOWUP / f"protocol1_multirep_{split_mode}_paired_deltas.csv", index=False
    )
    print(paired.to_string(index=False))
    print(appear.to_string(index=False))

    meta = {
        "split_mode": split_mode,
        "pipeline_version": PIPELINE_VERSION,
        "closed_seeds": closed_seeds,
        "open_seeds": open_seeds,
        "c_common": c_common,
        "n_expected_seeds": len(PROTOCOL1_MULTIREP_SEEDS),
        "n_expected_methods": len(PROTOCOL1_CORE_METHODS),
        "std_note": (
            "Overlapping Protocol-1 reps; sample std is split sensitivity, "
            "not population SE"
        ),
    }
    (FOLLOWUP / f"protocol1_multirep_{split_mode}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return summary


def _fmt_pm(mean: float, std: float) -> str:
    return f"${mean:.3f}\\pm{std:.3f}$"


def write_multirep_latex_table(summary: pd.DataFrame, split_mode: str) -> None:
    """Emit a peer table fragment with Macro / C_common / Worst-fixed / W10."""
    if summary is None or summary.empty:
        return
    required = {
        "macro_legacy_mean",
        "macro_legacy_std",
        "macro_ccommon_mean",
        "macro_ccommon_std",
        "worst_fixed_mean",
        "worst_fixed_std",
        "w10_fixed_mean",
        "w10_fixed_std",
    }
    missing = required - set(summary.columns)
    if missing:
        raise SystemExit(f"closed summary missing columns for W10 peer table: {sorted(missing)}")

    lines = [
        "% Auto-generated by scripts/aggregate_protocol1_multirep_r3.py — do not edit by hand.",
        f"% split_mode={split_mode}; columns: Macro, C_common, Worst-fixed, W10-fixed",
        r"\begin{tabular*}{\tblwidth}{@{}lcccc@{}}",
        r"\toprule",
        r"Method & Macro & $C_{\mathrm{common}}$ & Worst-fixed & W10-fixed \\",
        r"\midrule",
    ]
    diag = {"lgbm_hybrid_smooth"}
    primary = summary[~summary["method"].isin(diag)]
    smooth = summary[summary["method"].isin(diag)]
    for _, r in primary.iterrows():
        label = str(r["method_label"])
        if label == "Global":
            label = "Global-LGBM (inductive)" if split_mode == "well_id" else "Global-LGBM"
        elif label == "WellNorm":
            label = "WellNorm-LGBM (test-domain)" if split_mode == "well_id" else "WellNorm-LGBM"
        elif label == "Hybrid-norel":
            label = "Hybrid-norel (test-domain)" if split_mode == "well_id" else "Hybrid-norel"
        elif label == "Hybrid":
            label = "Hybrid (test-domain)" if split_mode == "well_id" else "Hybrid"
        elif label == "Hybrid-while":
            label = "Hybrid-while (causal-prefix)" if split_mode == "well_id" else "Hybrid-while"
        lines.append(
            f"{label} & {_fmt_pm(r['macro_legacy_mean'], r['macro_legacy_std'])} & "
            f"{_fmt_pm(r['macro_ccommon_mean'], r['macro_ccommon_std'])} & "
            f"{_fmt_pm(r['worst_fixed_mean'], r['worst_fixed_std'])} & "
            f"{_fmt_pm(r['w10_fixed_mean'], r['w10_fixed_std'])} \\\\"
        )
    if not smooth.empty:
        lines.append(r"\midrule")
        for _, r in smooth.iterrows():
            lab = "Hybrid+Smooth (score-support diag.)" if split_mode == "well_id" else "Hybrid+Smooth (diag.)"
            lines.append(
                f"{lab} & {_fmt_pm(r['macro_legacy_mean'], r['macro_legacy_std'])} & "
                f"{_fmt_pm(r['macro_ccommon_mean'], r['macro_ccommon_std'])} & "
                f"{_fmt_pm(r['worst_fixed_mean'], r['worst_fixed_std'])} & "
                f"{_fmt_pm(r['w10_fixed_mean'], r['w10_fixed_std'])} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular*}"])
    text = "\n".join(lines) + "\n"
    for dest in (FOLLOWUP, FIGS, PAPER_FIGS, OUT):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"table_protocol1_multirep_{split_mode}.tex").write_text(text, encoding="utf-8")
    # Flat CSV twin with W10 for paper-table sync / audits.
    cols = [
        "method",
        "method_label",
        "n_closed_reps",
        "macro_legacy_mean",
        "macro_legacy_std",
        "macro_ccommon_mean",
        "macro_ccommon_std",
        "worst_fixed_mean",
        "worst_fixed_std",
        "w10_fixed_mean",
        "w10_fixed_std",
    ]
    summary[cols].to_csv(FIGS / f"table_protocol1_multirep_{split_mode}.csv", index=False)
    summary[cols].to_csv(PAPER_FIGS / f"table_protocol1_multirep_{split_mode}.csv", index=False)


def pareto_fig(summary: pd.DataFrame, split_mode: str) -> None:
    if summary is None or summary.empty:
        return
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for _, r in summary.iterrows():
        label = str(r["method_label"])
        ax.errorbar(
            r["worst_fixed_mean"],
            r["macro_legacy_mean"],
            xerr=r["worst_fixed_std"],
            yerr=r["macro_legacy_std"],
            fmt="o",
            capsize=3,
            label=label,
        )
    ax.set_xlabel("Worst-fixed Macro-F1 (mean ± std, closed reps)")
    ax.set_ylabel("Pooled Macro-F1 legacy (mean ± std, closed reps)")
    ax.set_title(f"Protocol-1 {split_mode}: Macro vs Worst-fixed (Pareto)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    for dest in (FOLLOWUP, FIGS, PAPER_FIGS, OUT):
        dest.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest / f"fig_followup_pareto_{split_mode}.png", dpi=200)
        fig.savefig(dest / f"fig_followup_pareto_{split_mode}.pdf")
    plt.close(fig)


def main() -> None:
    FOLLOWUP.mkdir(parents=True, exist_ok=True)
    frames = []
    paired_frames = []
    for mode in ("well_id", "parent_grouped"):
        s = aggregate_mode(mode)
        if s is not None:
            frames.append(s)
            write_multirep_latex_table(s, mode)
            pareto_fig(s, mode)
            print(s.to_string(index=False))
            p = FOLLOWUP / f"protocol1_multirep_{mode}_paired_deltas.csv"
            if p.exists():
                paired_frames.append(pd.read_csv(p))
    if frames:
        all_s = pd.concat(frames, ignore_index=True)
        all_s.to_csv(FOLLOWUP / "protocol1_multirep_closed_summary_all.csv", index=False)
        print("wrote protocol1_multirep_closed_summary_all.csv")
    if paired_frames:
        all_p = pd.concat(paired_frames, ignore_index=True)
        all_p.to_csv(FOLLOWUP / "protocol1_multirep_paired_deltas_all.csv", index=False)
        print("wrote protocol1_multirep_paired_deltas_all.csv")


if __name__ == "__main__":
    main()
