"""Regenerate Fig.9 (LOWO selection) and Fig.10 (per-class F1) with high-contrast legends."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "followup"
PAPER = ROOT / "results" / "paper_figures"
GSE = ROOT / "paper" / "gse" / "figs"
DPI = 300

# Cool vs warm: high luminance + hue contrast (print-safe)
C1 = "#1f4e79"  # deep steel blue
C2 = "#d94801"  # strong orange


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig, name: str) -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    GSE.mkdir(parents=True, exist_ok=True)
    for folder in (PAPER, GSE, OUT):
        fig.savefig(folder / name, dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"saved {name}", flush=True)


def fig9_lowo() -> None:
    path = OUT / "lowo_primary_vs_alt_means.csv"
    out = pd.read_csv(path)
    methods = ["lgbm_wellnorm", "lgbm_hybrid", "lgbm_hybrid_smooth", "lgbm_hybrid_plus"]
    methods = [m for m in methods if m in set(out["method"])]
    a = out[out["selection"] == "primary"].set_index("method")
    b = out[out["selection"] == "spaced_alt"].set_index("method")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(methods))
    w = 0.36
    ax.bar(
        x - w / 2,
        [a.loc[m, "mean"] if m in a.index else 0 for m in methods],
        w,
        label="Primary LOWO",
        color=C1,
        edgecolor="black",
        linewidth=0.4,
        zorder=3,
    )
    ax.bar(
        x + w / 2,
        [b.loc[m, "mean"] if m in b.index else 0 for m in methods],
        w,
        label="Alt spaced LOWO",
        color=C2,
        edgecolor="black",
        linewidth=0.4,
        zorder=3,
    )
    METHOD_LABELS = {
        "lgbm_wellnorm": "WellNorm-LGBM",
        "lgbm_hybrid": "Hybrid",
        "lgbm_hybrid_smooth": "Hybrid+Smooth",
        "lgbm_hybrid_plus": "Hybrid+Plus",
    }
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=20, ha="right")
    ax.set_ylabel("Mean Macro-F1")
    ax.set_title("LOWO selection sensitivity (6 wells each)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#dddddd", lw=0.6)
    _save(fig, "fig_followup_lowo_selection.png")


def fig10_perclass() -> None:
    path = OUT / "per_class_f1_protocol1_seed42.csv"
    tab = pd.read_csv(path)
    # expect columns: class / lithology, method, f1, support (flexible)
    cols = {c.lower(): c for c in tab.columns}
    method_col = cols.get("method")
    f1_col = cols.get("f1") or cols.get("macro_f1") or cols.get("per_class_f1")
    name_col = cols.get("class_name") or cols.get("lithology") or cols.get("label") or cols.get("class")
    support_col = cols.get("support") or cols.get("n") or cols.get("count")
    if method_col is None or f1_col is None or name_col is None:
        raise SystemExit(f"unexpected columns: {tab.columns.tolist()}")

    wide = tab.pivot_table(index=name_col, columns=method_col, values=f1_col, aggfunc="first")
    if support_col:
        supports = tab.groupby(name_col)[support_col].max()
        order = supports.sort_values().index.tolist()
    else:
        order = list(wide.index)
        supports = None

    # keep Global / Hybrid only
    for need in ("Global", "Hybrid"):
        if need not in wide.columns:
            # try lgbm_ names
            mapping = {c: c for c in wide.columns}
            for c in list(wide.columns):
                cl = str(c).lower()
                if "global" in cl:
                    mapping[c] = "Global"
                elif "hybrid" in cl and "plus" not in cl and "sim" not in cl and "smooth" not in cl:
                    mapping[c] = "Hybrid"
            wide = wide.rename(columns=mapping)
            wide = wide.groupby(level=0, axis=1).first()

    order = [c for c in order if c in wide.index]
    labels = []
    for c in order:
        if supports is not None and c in supports.index:
            labels.append(f"{c}\n(n={int(supports.loc[c])})")
        else:
            labels.append(str(c))

    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.bar(
        x - w / 2,
        [float(wide.loc[c, "Global"]) if "Global" in wide.columns else 0 for c in order],
        w,
        label="Global-LGBM",
        color=C1,
        edgecolor="black",
        linewidth=0.4,
        zorder=3,
    )
    ax.bar(
        x + w / 2,
        [float(wide.loc[c, "Hybrid"]) if "Hybrid" in wide.columns else 0 for c in order],
        w,
        label="Hybrid",
        color=C2,
        edgecolor="black",
        linewidth=0.4,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Per-class F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Protocol-1 seed 42: per-class F1 (Global-LGBM vs Hybrid)")
    ax.legend(frameon=False, loc="upper right")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#dddddd", lw=0.6)
    _save(fig, "fig_followup_per_class_f1.png")


def main() -> None:
    _style()
    fig9_lowo()
    fig10_perclass()


if __name__ == "__main__":
    main()
