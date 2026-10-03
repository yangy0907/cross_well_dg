"""Generate publication-quality figures for the paper (English labels, 300 dpi)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Embed TrueType (Type 42) fonts for publisher PDF compliance.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "DejaVu Sans"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR
from src.train import PIPELINE_VERSION

PAPER_DIR = RESULTS_DIR / "paper_figures"
# Submission figs dir; tests must monkeypatch this away from the real tree.
PAPER_GSE_FIGS = ROOT / "paper" / "gse" / "figs"
DPI = 300

# Consistent palette
C_BASE = "#4C78A8"
C_OURS = "#F58518"
C_WORST = "#E45756"
C_WELL = "#54A24B"
C_GRAY = "#9D9D9D"

DISPLAY = {
    "naive_depth_split_lgbm": "Naive depth-split",
    "lgbm_global": "Global-LGBM",
    "lgbm_global_eq120k": "Global-LGBM (≤120k)",
    "rf_global": "Global-RF",
    "rf_global_full": "Global-RF (full)",
    "xgb_global": "Global-XGB",
    "lgbm_wellnorm": "WellNorm-LGBM",
    "lgbm_conf": "ConfWeight-LGBM",
    "lgbm_full": "Full-LGBM",
    "lgbm_groupdro": "GroupDRO-LGBM",
    "mlp_coral_uda": "MLP+CORAL",
    "lgbm_hybrid": "Hybrid",
    "lgbm_hybrid_norel": "Hybrid-norel",
    "lgbm_hybrid_while": "Hybrid-while",
    "lgbm_hybrid_sim": "Hybrid+Sim",
    "lgbm_hybrid_sim_perwell": "Hybrid+Sim (per-well)",
    "lgbm_hybrid_smooth": "Hybrid+Smooth",
    "lgbm_hybrid_plus": "Hybrid+Plus",
}

# Main-text bars: no Hybrid+Sim / Plus (those scores are historical asymmetric-support
# diagnostics in the Supplementary material only).
MAIN_METHOD_ORDER = [
    "lgbm_global",
    "lgbm_wellnorm",
    "lgbm_hybrid",
    "lgbm_hybrid_smooth",
    "lgbm_hybrid_while",
]

# Protocol-1 multi-seed table/fig: include Hybrid-norel + inductive refs
# (rf_global_full / xgb_global only if present in summary CSVs).
MULTISEED_METHOD_ORDER = [
    "lgbm_global",
    "lgbm_global_eq120k",
    "rf_global",
    "rf_global_full",
    "xgb_global",
    "lgbm_wellnorm",
    "lgbm_hybrid_norel",
    "lgbm_hybrid",
    "lgbm_hybrid_smooth",
    "lgbm_hybrid_while",
]

# Supplementary ablation may still show historical Sim/Plus if present in CSV.
ABLATION_ORDER = [
    "lgbm_global",
    "lgbm_wellnorm",
    "lgbm_hybrid",
    "lgbm_hybrid_smooth",
    "lgbm_hybrid_while",
    "lgbm_hybrid_sim",
    "lgbm_hybrid_plus",
]

# Primary Protocol-3 LOWO targets (exclude spaced_alt selection used only in Table 6 / Fig. lowosel)
PRIMARY_LOWO_WELLS = {
    "15/9-23",
    "17/11-1",
    "29/3-1",
    "33/9-1",
    "34/10-16 R",
    "7/1-1",
}


def _load_all() -> pd.DataFrame:
    """Load summary CSVs; prefer current PIPELINE_VERSION, else latest r3_* rows.

    Preferring an exact match keeps release figures locked. Falling back to any
    ``r3_*`` rows lets closed-set figure fixes regenerate before the mandatory
    provenance-locked re-run (see RELEASE_CHECKLIST.md).
    """
    files = [
        f
        for f in sorted(RESULTS_DIR.glob("summary_*.csv"))
        if f.name != "summary_all.csv" and "archive" not in str(f)
    ]
    if not files:
        raise FileNotFoundError(f"No summary_*.csv under {RESULTS_DIR}")
    all_parts = []
    for f in files:
        chunk = pd.read_csv(f)
        if "pipeline_version" not in chunk.columns:
            print(f"[figures] skip {f.name}: missing pipeline_version")
            continue
        all_parts.append(chunk)
    if not all_parts:
        raise RuntimeError(f"No usable summary_*.csv under {RESULTS_DIR}")
    raw = pd.concat(all_parts, ignore_index=True)
    cur = raw[raw["pipeline_version"].astype(str) == PIPELINE_VERSION]
    if not cur.empty:
        print(f"[figures] using pipeline_version={PIPELINE_VERSION} ({len(cur)} rows)")
        return cur.copy()
    r3 = raw[raw["pipeline_version"].astype(str).str.startswith("r3_")]
    if r3.empty:
        raise RuntimeError(
            f"No rows with pipeline_version={PIPELINE_VERSION!r} or r3_* fallback. "
            "Re-run experiments or bump filter."
        )
    vers = sorted(r3["pipeline_version"].astype(str).unique())
    print(
        f"[figures] WARN: no {PIPELINE_VERSION!r}; falling back to {vers} "
        f"({len(r3)} rows). Re-run required before release."
    )
    return r3.copy()


FIGSIZE_WIDE = (8.2, 4.2)
FIGSIZE_PANEL = (9.2, 4.0)


def _style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": DPI,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def _polish(ax, ylabel: str | None = None, title: str | None = None, ylim=None):
    """Shared axis look for all quantitative paper figures."""
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=8)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="both", length=3.5)


def _method_color(method: str) -> str:
    return C_OURS if "hybrid" in method else C_BASE


def _worst_col(df: pd.DataFrame) -> str:
    """Primary tail metric: fixed-taxonomy worst-well when available."""
    if "worst_well_f1_fixed" in df.columns and df["worst_well_f1_fixed"].notna().any():
        return "worst_well_f1_fixed"
    return "worst_well_f1"


def _worst_label(col: str) -> str:
    return "Worst-fixed F1" if col.endswith("_fixed") else "Worst-true F1 (sens.)"


def _save(fig: plt.Figure, name: str):
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    folders = [PAPER_DIR, FIGURES_DIR]
    if PAPER_GSE_FIGS is not None:
        PAPER_GSE_FIGS.mkdir(parents=True, exist_ok=True)
        folders.append(PAPER_GSE_FIGS)
    fig.tight_layout()
    for folder in folders:
        fig.savefig(folder / name, dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"saved {name}")


def fig_leakage_contrast(df: pd.DataFrame):
    naive = df[df["protocol"] == "naive_depth_split"]
    p1 = df[df["protocol"] == "protocol1_random_wells"]
    p1s = p1[p1["seed"] == 42] if (not p1.empty and (p1["seed"] == 42).any()) else p1
    best = None
    if not p1s.empty:
        hit = p1s[p1s["method"] == "lgbm_global"]
        if not hit.empty:
            best = hit.drop_duplicates("method", keep="last").iloc[0]
    naive_macro = None
    if not naive.empty:
        naive_macro = float(naive.drop_duplicates("method", keep="last").iloc[-1]["macro_f1"])
    # Fallback: matched leakage study A/C (Global-LGBM; pipeline r3 numbers)
    matched = RESULTS_DIR / "followup" / "matched_leakage_study.csv"
    if (naive_macro is None or best is None) and matched.exists():
        mdf = pd.read_csv(matched)
        if naive_macro is None and (mdf["setting"] == "A_naive_depth").any():
            naive_macro = float(mdf.loc[mdf["setting"] == "A_naive_depth", "macro_f1"].iloc[0])
        if best is None and (mdf["setting"] == "C_well_holdout").any():
            best = {
                "method": "lgbm_global",
                "macro_f1": float(mdf.loc[mdf["setting"] == "C_well_holdout", "macro_f1"].iloc[0]),
            }
    if naive_macro is None or best is None:
        return

    method_name = best["method"] if isinstance(best, dict) else str(best["method"])
    p1_macro = float(best["macro_f1"])
    labels = [
        "Naive depth-split\n(same-well depth split)",
        f"Protocol-1 cross-well\n({DISPLAY.get(method_name, 'Global-LGBM')})",
    ]
    vals = [float(naive_macro), p1_macro]
    colors = [C_GRAY, C_OURS]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    bars = ax.bar(labels, vals, color=colors, width=0.48, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    _polish(
        ax,
        ylabel="Macro-F1",
        title="Optimistic same-well depth split versus cross-well evaluation",
        ylim=(0, max(vals) * 1.22),
    )
    _save(fig, "fig1_leakage_vs_crosswell.png")


def fig_protocol1_main(df: pd.DataFrame):
    p1 = df[df["protocol"] == "protocol1_random_wells"].copy()
    if p1.empty:
        return
    # prefer seed 42 for main bar; if multi-seed, aggregate later
    p1 = p1[p1["seed"] == 42] if (p1["seed"] == 42).any() else p1
    methods = [m for m in MAIN_METHOD_ORDER if m in set(p1["method"])]
    g = p1[p1["method"].isin(methods)].drop_duplicates("method", keep="last")
    g = g.set_index("method").loc[methods].reset_index()

    labels = [DISPLAY.get(m, m) for m in g["method"]]
    x = np.arange(len(g))
    w = 0.36
    wcol = _worst_col(g)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    colors = [_method_color(m) for m in g["method"]]
    ax.bar(x - w / 2, g["macro_f1"], w, label="Macro-F1", color=colors, alpha=0.95, zorder=3)
    ax.bar(x + w / 2, g[wcol], w, label=_worst_label(wcol), color=C_WORST, alpha=0.85, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(frameon=False, loc="upper right")
    ymax = max(g["macro_f1"].max(), g[wcol].max()) * 1.28
    _polish(ax, ylabel="Score", title="Protocol-1: random well hold-out (seed=42)", ylim=(0, ymax))
    _save(fig, "fig2_protocol1_macro_worst.png")

    # well-mean comparison
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(x, g["macro_f1"], "o-", color=C_BASE, label="Macro-F1", lw=1.8, ms=6, zorder=3)
    ax.plot(x, g["well_macro_f1_mean"], "s--", color=C_WELL, label="Mean well Macro-F1", lw=1.6, ms=5, zorder=3)
    ax.plot(x, g[wcol], "^:", color=C_WORST, label=_worst_label(wcol), lw=1.6, ms=6, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(frameon=False)
    _polish(ax, ylabel="Score", title="Protocol-1: metric breakdown", ylim=(0, 0.55))
    _save(fig, "fig2b_protocol1_metric_breakdown.png")


def fig_ablation(df: pd.DataFrame):
    p1 = df[(df["protocol"] == "protocol1_random_wells") & (df["seed"] == 42)].copy()
    if p1.empty:
        return
    methods = [m for m in MAIN_METHOD_ORDER if m in set(p1["method"])]
    g = p1[p1["method"].isin(methods)].drop_duplicates("method", keep="last")
    g = g.set_index("method").loc[methods].reset_index()

    labels = [DISPLAY.get(m, m) for m in g["method"]]
    wcol = _worst_col(g)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_PANEL, sharey=False)
    axes[0].plot(range(len(g)), g["macro_f1"], "o-", color=C_OURS, lw=2, ms=6, zorder=3)
    axes[0].set_xticks(range(len(g)))
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    _polish(axes[0], ylabel="Macro-F1", title="Ablation: Macro-F1")

    axes[1].plot(range(len(g)), g[wcol], "o-", color=C_WORST, lw=2, ms=6, zorder=3)
    axes[1].set_xticks(range(len(g)))
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    _polish(axes[1], ylabel=_worst_label(wcol), title=f"Ablation: {_worst_label(wcol)}")
    fig.suptitle("Component ablation on Protocol-1", y=1.02, fontsize=11)
    _save(fig, "fig3_ablation_protocol1.png")

    # save table
    tab_cols = ["method", "macro_f1", "well_macro_f1_mean", wcol, "seconds"]
    if "worst_well_f1" in g.columns and wcol != "worst_well_f1":
        tab_cols.insert(-1, "worst_well_f1")
    tab = g[[c for c in tab_cols if c in g.columns]].copy()
    tab["display"] = tab["method"].map(DISPLAY)
    tab.to_csv(PAPER_DIR / "table_ablation_protocol1.csv", index=False)


def fig_protocol2(df: pd.DataFrame):
    p2 = df[df["protocol"] == "protocol2_block_holdout"].copy()
    if p2.empty:
        return
    methods = [m for m in MAIN_METHOD_ORDER if m in set(p2["method"])]
    blocks = sorted(p2["holdout_blocks"].astype(str).unique())

    fig, axes = plt.subplots(1, len(blocks), figsize=(4.4 * len(blocks), 4.2), sharey=True)
    if len(blocks) == 1:
        axes = [axes]
    for ax, blk in zip(axes, blocks):
        g = p2[p2["holdout_blocks"].astype(str) == blk]
        g = g[g["method"].isin(methods)].drop_duplicates("method", keep="last")
        g = g.set_index("method").reindex(methods).dropna(subset=["macro_f1"]).reset_index()
        labels = [DISPLAY.get(m, m) for m in g["method"]]
        colors = [_method_color(m) for m in g["method"]]
        ax.barh(range(len(g)), g["macro_f1"], color=colors, zorder=3, height=0.68)
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Macro-F1")
        ax.set_title(f"Hold-out quadrant {blk}")
        ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        xmax = float(g["macro_f1"].max()) * 1.18
        ax.set_xlim(0, xmax)
        for i, v in enumerate(g["macro_f1"]):
            ax.text(v + 0.008 * xmax, i, f"{v:.3f}", va="center", fontsize=8, color="#333333")
    fig.suptitle("Protocol-2: quadrant hold-out generalization", y=1.02, fontsize=11)
    _save(fig, "fig4_protocol2_block_holdout.png")


def fig_protocol3(df: pd.DataFrame):
    p3 = df[df["protocol"] == "protocol3_lowo"].copy()
    if p3.empty:
        return
    # Keep only the primary six LOWO wells; spaced_alt is reported separately.
    p3["target_well"] = p3["target_well"].astype(str).str.replace("~", " ", regex=False).str.strip()
    p3 = p3[p3["target_well"].isin(PRIMARY_LOWO_WELLS)].copy()
    if p3.empty:
        return
    # Prefer a method order consistent with Protocol-1 panels (no Sim/Plus in main figs)
    focus = [
        m
        for m in ["lgbm_wellnorm", "lgbm_global"] + MAIN_METHOD_ORDER
        if m in set(p3["method"])
    ]
    focus = list(dict.fromkeys(focus))
    g = (
        p3[p3["method"].isin(focus)]
        .groupby("method", as_index=False)
        .agg(
            macro_f1=("macro_f1", "mean"),
            macro_std=("macro_f1", "std"),  # pandas default ddof=1 (sample std)
            n=("macro_f1", "size"),
        )
    )
    order = [m for m in focus if m in set(g["method"])]
    g = g.set_index("method").loc[order].reset_index()
    g["macro_std"] = g["macro_std"].fillna(0.0)

    n_wells = int(p3["target_well"].nunique())
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    x = np.arange(len(g))
    colors = [_method_color(m) for m in g["method"]]
    ax.bar(
        x,
        g["macro_f1"],
        yerr=g["macro_std"],
        color=colors,
        capsize=3,
        zorder=3,
        alpha=0.95,
        error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#444444"},
    )
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY.get(m, m) for m in g["method"]], rotation=25, ha="right")
    ymax = max(0.25, float((g["macro_f1"] + g["macro_std"]).max()) * 1.22)
    _polish(
        ax,
        ylabel="Mean Macro-F1 over target wells",
        title=f"Protocol-3: leave-one-well-out (n = {n_wells} wells)",
        ylim=(0, ymax),
    )
    _save(fig, "fig5_protocol3_lowo_mean.png")

    # per-well grouped bars for key methods (primary wells only)
    key = [
        m
        for m in [
            "lgbm_wellnorm",
            "lgbm_global",
            "lgbm_hybrid",
            "lgbm_hybrid_smooth",
            "lgbm_hybrid_while",
        ]
        if m in set(p3["method"])
    ]
    if len(key) >= 2:
        well_order = [w for w in ["15/9-23", "17/11-1", "29/3-1", "33/9-1", "34/10-16 R", "7/1-1"] if w in set(p3["target_well"])]
        pivot = p3[p3["method"].isin(key)].pivot_table(
            index="target_well", columns="method", values="macro_f1", aggfunc="last"
        )
        pivot = pivot.reindex(index=well_order, columns=key)
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        width = 0.8 / len(key)
        xpos = np.arange(len(pivot.index))
        palette = [C_BASE, C_OURS, "#ECAA38", "#72B7B2", "#B279A2"]
        for i, m in enumerate(key):
            ax.bar(
                xpos + i * width - 0.4 + width / 2,
                pivot[m].to_numpy(),
                width=width * 0.92,
                label=DISPLAY.get(m, m),
                color=palette[i % len(palette)],
                zorder=3,
            )
        ax.set_xticks(xpos)
        ax.set_xticklabels([str(w) for w in pivot.index], rotation=25, ha="right")
        ax.legend(frameon=False, fontsize=8, ncol=2)
        _polish(
            ax,
            ylabel="Macro-F1",
            title="Protocol-3: per-well Macro-F1 (primary six wells)",
            ylim=(0, max(0.2, float(np.nanmax(pivot.to_numpy())) * 1.18)),
        )
        _save(fig, "fig5b_protocol3_per_well.png")
        pivot.to_csv(PAPER_DIR / "table_protocol3_per_well.csv")


def fig_multiseed(df: pd.DataFrame):
    from src.constants import PROTOCOL1_CLOSED_SEEDS

    p1 = df[df["protocol"] == "protocol1_random_wells"].copy()
    # Primary aggregation: closed-set seeds only (exclude open-set seed 44).
    p1 = p1[p1["seed"].isin(PROTOCOL1_CLOSED_SEEDS)].copy()
    if p1.empty or p1["seed"].nunique() < 2:
        return
    if set(p1["seed"].unique()) & {44}:
        raise AssertionError("fig_multiseed must not include open-set seed 44")
    present = set(p1["method"])
    focus = [m for m in MULTISEED_METHOD_ORDER if m in present]
    focus = list(dict.fromkeys(focus))
    rows = []
    for m in focus:
        sub = p1[p1["method"] == m]
        has_fixed = "worst_well_f1_fixed" in sub.columns and sub["worst_well_f1_fixed"].notna().any()
        rows.append(
            {
                "method": m,
                "macro_mean": sub["macro_f1"].mean(),
                "macro_std": sub["macro_f1"].std(ddof=1) if len(sub) > 1 else 0.0,
                # Primary tail = fixed taxonomy; legacy true kept as sensitivity.
                "worst_mean": (
                    float(sub["worst_well_f1_fixed"].mean())
                    if has_fixed
                    else float(sub["worst_well_f1"].mean())
                ),
                "worst_std": (
                    float(sub["worst_well_f1_fixed"].std(ddof=1))
                    if has_fixed and len(sub) > 1
                    else (float(sub["worst_well_f1"].std(ddof=1)) if len(sub) > 1 else 0.0)
                ),
                "worst_true_mean": float(sub["worst_well_f1"].mean()),
                "worst_true_std": (
                    float(sub["worst_well_f1"].std(ddof=1)) if len(sub) > 1 else 0.0
                ),
                "worst_fixed_mean": (
                    float(sub["worst_well_f1_fixed"].mean()) if has_fixed else float("nan")
                ),
                "worst_fixed_std": (
                    float(sub["worst_well_f1_fixed"].std(ddof=1))
                    if has_fixed and len(sub) > 1
                    else 0.0
                ),
                "n_seeds": int(sub["seed"].nunique()),
                "seeds": ",".join(str(int(s)) for s in sorted(sub["seed"].unique())),
            }
        )
    g = pd.DataFrame(rows)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = PAPER_DIR / "table_protocol1_multiseed.csv"
    g.to_csv(out_csv, index=False)
    if PAPER_GSE_FIGS is not None:
        PAPER_GSE_FIGS.mkdir(parents=True, exist_ok=True)
        g.to_csv(PAPER_GSE_FIGS / "table_protocol1_multiseed.csv", index=False)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    x = np.arange(len(g))
    colors = [_method_color(m) for m in g["method"]]
    bars = ax.bar(
        x,
        g["macro_mean"],
        yerr=g["macro_std"],
        color=colors,
        capsize=3,
        alpha=0.95,
        zorder=3,
        error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#444444"},
    )
    for bar, n in zip(bars, g["n_seeds"]):
        if int(n) < 2:
            bar.set_hatch("//")
            bar.set_edgecolor("#555555")
            bar.set_linewidth(0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY.get(m, m) for m in g["method"]], rotation=25, ha="right")
    n_seeds = int(g["n_seeds"].max())
    seed_label = ",".join(str(s) for s in PROTOCOL1_CLOSED_SEEDS)
    ymax = float((g["macro_mean"] + g["macro_std"]).max()) * 1.22
    _polish(
        ax,
        ylabel="Macro-F1 (mean ± sample std)",
        title=f"Protocol-1 closed-set Macro-F1 (seeds {seed_label}; n = {n_seeds})",
        ylim=(0, ymax),
    )
    _save(fig, "fig6_protocol1_multiseed.png")

def fig_class_and_drift():
    # class distribution
    cc = PROCESSED_DIR / "class_counts.csv"
    if cc.exists():
        d = pd.read_csv(cc)
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        ax.bar(np.arange(len(d)), d["count"], color=C_BASE, zorder=3)
        ax.set_xticks(np.arange(len(d)))
        ax.set_xticklabels(d["lithology_name"], rotation=30, ha="right")
        _polish(ax, ylabel="Sample count", title="FORCE 2020 lithofacies class distribution")
        _save(fig, "fig0_class_distribution.png")

    prior = RESULTS_DIR / "drift_class_prior_p1.csv"
    if prior.exists():
        d = pd.read_csv(prior)
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        x = np.arange(len(d))
        w = 0.38
        ax.bar(x - w / 2, d["train_frac"], w, label="Train wells", color=C_BASE, zorder=3)
        ax.bar(x + w / 2, d["test_frac"], w, label="Test wells", color=C_OURS, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(d["lithology"], rotation=30, ha="right")
        ax.legend(frameon=False)
        _polish(ax, ylabel="Fraction", title="Protocol-1 class prior shift (train vs test)")
        _save(fig, "fig0b_class_prior_shift.png")

    zdiff = RESULTS_DIR / "drift_curve_zdiff_p1.csv"
    if zdiff.exists():
        d = pd.read_csv(zdiff)
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        colors = [C_WORST if abs(v) > 0.3 else C_BASE for v in d["z_diff_test_minus_train"]]
        ax.bar(np.arange(len(d)), d["z_diff_test_minus_train"], color=colors, zorder=3)
        ax.axhline(0, color="black", lw=0.8, zorder=2)
        ax.set_xticks(np.arange(len(d)))
        ax.set_xticklabels(d["curve"], rotation=30, ha="right")
        _polish(ax, ylabel="Mean z-diff (test − train)", title="Protocol-1 feature drift across wells")
        _save(fig, "fig0c_feature_drift.png")


def fig_worst_wells():
    path = RESULTS_DIR / "details_protocol1_random_wells_seed42.json"
    if not path.exists():
        return
    details = json.loads(path.read_text(encoding="utf-8"))
    # pick hybrid_sim or hybrid_plus or hybrid
    prefer = ["lgbm_hybrid_sim", "lgbm_hybrid_plus", "lgbm_hybrid", "lgbm_global"]
    item = None
    for name in prefer:
        item = next((d for d in details if d.get("method") == name), None)
        if item:
            break
    if not item:
        return
    wt = pd.DataFrame(item["well_table"]).sort_values("macro_f1")
    show = pd.concat([wt.head(5), wt.tail(3)])
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    colors = [C_WORST if i < 5 else C_WELL for i in range(len(show))]
    ax.barh(range(len(show)), show["macro_f1"], color=colors, zorder=3, height=0.7)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels(show["well_id"])
    ax.invert_yaxis()
    ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Well Macro-F1")
    ax.set_title(f"Protocol-1 well-wise F1 ({DISPLAY.get(item['method'], item['method'])})")
    _save(fig, "fig7_worst_best_wells.png")
    show.to_csv(PAPER_DIR / "table_worst_best_wells.csv", index=False)


def fig_method_overview():
    """Information-budget and evaluation overview for the protocol paper."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    def rounded_box(x, y, w, h, face, edge="#34495E", lw=1.0, radius=0.10):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=f"round,pad=0.03,rounding_size={radius}",
                linewidth=lw,
                edgecolor=edge,
                facecolor=face,
            )
        )

    def arrow(start, end, *, dashed=False, color="#566573", width=1.15):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=width,
                linestyle="--" if dashed else "-",
                color=color,
            )
        )

    ax.text(
        6.0,
        8.28,
        "Leakage-restricted baselines under three information budgets",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#243746",
    )

    rounded_box(0.35, 7.52, 11.30, 0.48, "#244A5A", edge="#244A5A", lw=0.8, radius=0.08)
    ax.text(
        6.0,
        7.76,
        r"LEAKAGE BARRIER: target lithofacies labels $y_t$ are hidden until final scoring",
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color="white",
    )

    rounded_box(0.45, 6.35, 5.25, 0.88, "#EAF2F8", edge="#4C78A8")
    ax.text(3.075, 6.93, r"Labelled source wells  $W_s$", ha="center", va="center", fontsize=9.3, fontweight="bold", color="#2C5F89")
    ax.text(3.075, 6.58, r"curves $x_s$  +  facies $y_s$  +  confidence $c_s$", ha="center", va="center", fontsize=8.4, color="#2F3E46")

    rounded_box(6.30, 6.35, 5.25, 0.88, "#F7F3E8", edge="#B07C25")
    ax.text(8.925, 6.93, r"Unlabelled target well(s)  $W_t$", ha="center", va="center", fontsize=9.3, fontweight="bold", color="#8A5A14")
    ax.text(8.925, 6.58, r"curves $x_t$  +  measured depth $d_t$;  no $y_t$", ha="center", va="center", fontsize=8.4, color="#2F3E46")

    ax.text(6.0, 6.02, "Select and disclose the admissible information budget", ha="center", va="center", fontsize=8.7, color="#5D6D7E")
    ax.plot([3.10, 0.22, 0.22], [6.35, 5.92, 3.20], color="#4C78A8", lw=1.0, zorder=0)
    ax.plot([8.90, 11.78, 11.78], [6.35, 5.92, 3.20], color="#B07C25", lw=1.0, ls="--", zorder=0)

    rows = [
        (
            5.05,
            "1  Inductive DG",
            "Global-LGBM",
            "Global curves + missingness masks\nsource-fitted statistics only",
            "NO TARGET\nAGGREGATES",
            "#DCEAF5",
            "#3F6F96",
        ),
        (
            3.95,
            "2  Post-drilling",
            "Hybrid-full · target normalisation",
            "Global + whole-well z-scores\noptional relative depth",
            "FULL TARGET\nINTERVAL",
            "#FCE8D2",
            "#B76E20",
        ),
        (
            2.85,
            "3  While-drilling",
            "Hybrid-while · causal normalisation",
            "Global + causal-prefix z-scores\nno total depth / relative depth",
            "PAST + CURRENT\nONLY",
            "#E3F0DF",
            "#4F7B45",
        ),
    ]
    for y, setting, method, description, access, face, edge in rows:
        rounded_box(0.45, y, 11.10, 0.90, face, edge=edge, lw=1.0, radius=0.08)
        ax.text(0.72, y + 0.62, setting, ha="left", va="center", fontsize=8.9, fontweight="bold", color=edge)
        ax.text(0.72, y + 0.28, method, ha="left", va="center", fontsize=7.8, color="#34495E")
        ax.plot([3.80, 3.80], [y + 0.15, y + 0.75], color=edge, lw=0.7, alpha=0.55)
        ax.text(4.08, y + 0.45, description, ha="left", va="center", fontsize=7.9, color="#2F3E46", linespacing=1.25)
        rounded_box(9.30, y + 0.18, 2.00, 0.54, "#FFFFFF", edge=edge, lw=0.75, radius=0.07)
        ax.text(10.30, y + 0.45, access, ha="center", va="center", fontsize=6.8, fontweight="bold", color=edge, linespacing=1.05)

    # Solid source flow applies to every row; dashed target flow for post/while only.
    for y in (5.50, 4.40, 3.30):
        arrow((0.22, y), (0.45, y), color="#4C78A8", width=0.9)
    for y in (4.40, 3.30):
        arrow((11.78, y), (11.55, y), dashed=True, color="#B07C25", width=0.9)

    arrow((6.0, 2.75), (3.125, 2.15), color="#566573")
    rounded_box(0.55, 1.35, 5.15, 0.76, "#EAF2F8", edge="#4C78A8")
    ax.text(3.125, 1.84, "Shared CPU LightGBM backbone", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#2C5F89")
    ax.text(3.125, 1.53, "main-text feature variants  |  early stopping", ha="center", va="center", fontsize=7.8, color="#2F3E46")

    rounded_box(6.30, 1.35, 5.15, 0.76, "#F7EEF4", edge="#9A6687")
    ax.text(8.875, 1.84, "Optional post-processing", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#7A4F6E")
    ax.text(8.875, 1.53, "centred depth smoothing — OFFLINE only", ha="center", va="center", fontsize=8.0, color="#2F3E46")
    arrow((5.70, 1.73), (6.30, 1.73), dashed=True, color="#9A6687")

    ax.text(
        6.0,
        1.05,
        "Unequal-budget references (not shown): Global-RF and appendix CORAL-MLP",
        ha="center",
        va="center",
        fontsize=7.6,
        color="#5D6D7E",
        style="italic",
    )

    arrow((3.125, 1.34), (3.125, 0.88), color="#566573")
    arrow((8.875, 1.34), (8.875, 0.88), dashed=True, color="#9A6687")
    rounded_box(1.15, 0.12, 9.70, 0.72, "#F3F5F5", edge="#566573", lw=1.1)
    ax.text(6.0, 0.60, "Evaluation:  P1 random-well  |  P2 quadrant hold-out  |  P3 LOWO", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#34495E")
    ax.text(6.0, 0.32, "Macro-F1  +  well-level tail risk (worst-well / worst-10%)", ha="center", va="center", fontsize=8.1, color="#566573")

    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.02)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    folders = [PAPER_DIR, FIGURES_DIR]
    if PAPER_GSE_FIGS is not None:
        PAPER_GSE_FIGS.mkdir(parents=True, exist_ok=True)
        folders.append(PAPER_GSE_FIGS)
    for folder in folders:
        fig.savefig(folder / "figM_method_overview.png", dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
        fig.savefig(folder / "figM_method_overview.pdf", bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print("saved figM_method_overview.png/.pdf")


def write_caption_readme():
    text = """# Paper figures (300 dpi)

Generated by `scripts/make_paper_figures.py` (fig0–fig7, figM); fig8/fig9 by `scripts/make_case_figure.py`.

| File | Suggested use in paper |
|------|----------------------|
| figM_method_overview.png | Method overview (Fig. M) |
| fig0_class_distribution.png | Data / Experimental setup |
| fig0b_class_prior_shift.png | Domain shift evidence |
| fig0c_feature_drift.png | Domain shift evidence |
| fig1_leakage_vs_crosswell.png | Motivation: evaluation protocol |
| fig2_protocol1_macro_worst.png | Main result Protocol-1 |
| fig2b_protocol1_metric_breakdown.png | Supplementary / analysis |
| fig3_ablation_protocol1.png | Ablation study |
| fig4_protocol2_block_holdout.png | Main result Protocol-2 |
| fig5_protocol3_lowo_mean.png | Main result Protocol-3 |
| fig5b_protocol3_per_well.png | Protocol-3 detail |
| fig6_protocol1_multiseed.png | Robustness (if ≥2 seeds) |
| fig7_worst_best_wells.png | Case / failure analysis |
| fig8_case_well_strips.png | Case study strips |
| fig9_protocol1_worst_strips.png | Protocol-1 worst-well True/Global-LGBM/Hybrid+Plus strips |

Tables: `table_*.csv` in this folder.
"""
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    (PAPER_DIR / "README.md").write_text(text, encoding="utf-8")


def main():
    _style()
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_all()
    df.to_csv(RESULTS_DIR / "summary_all.csv", index=False)

    fig_method_overview()
    fig_class_and_drift()
    fig_leakage_contrast(df)
    fig_protocol1_main(df)
    fig_ablation(df)
    fig_protocol2(df)
    fig_protocol3(df)
    fig_multiseed(df)
    fig_worst_wells()
    write_caption_readme()
    print(f"\nAll paper figures -> {PAPER_DIR}")


if __name__ == "__main__":
    main()
