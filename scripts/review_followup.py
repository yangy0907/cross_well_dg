"""Reviewer follow-up analyses: per-class F1, sensitivity, seed stats, hard wells.

Outputs under results/followup/ and results/paper_figures/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import FIGURES_DIR, LITHOLOGY_MAP, RESULTS_DIR, WELL_COL
from src.features import FeaturePipeline
from src.metrics import evaluate_split
from src.models import LGBMConfig, LGBMLithologyModel
from src.prepare_data import load_labeled
from src.postprocess import similarity_sample_weights, smooth_predictions_by_well
from src.splits import apply_split, protocol1_random_wells
from src.train import _feature_routing

OUT = RESULTS_DIR / "followup"

_MODE_TO_METHOD = {
    "global": "lgbm_global",
    "hybrid": "lgbm_hybrid",
    "wellnorm": "lgbm_wellnorm",
}
PAPER = RESULTS_DIR / "paper_figures"
DPI = 300


def _style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": DPI,
        }
    )


def _save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for folder in (OUT, PAPER, FIGURES_DIR):
        fig.savefig(folder / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {name}", flush=True)


def multiseed_stats() -> pd.DataFrame:
    rows = []
    for seed in (42, 43, 44, 45):
        p = RESULTS_DIR / f"summary_protocol1_random_wells_seed{seed}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["seed"] = seed
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    all_df = pd.concat(rows, ignore_index=True)
    focus = [
        "lgbm_global",
        "lgbm_wellnorm",
        "lgbm_hybrid",
        "lgbm_hybrid_sim",
        "lgbm_hybrid_smooth",
        "lgbm_hybrid_plus",
    ]
    all_df = all_df[all_df["method"].isin(focus)].copy()

    # paired deltas vs Global on shared seeds
    pivot = all_df.pivot_table(index="seed", columns="method", values="macro_f1")
    stats_rows = []
    for m in focus:
        sub = all_df[all_df["method"] == m]
        if sub.empty:
            continue
        rec = {
            "method": m,
            "n_seeds": int(sub["seed"].nunique()),
            "macro_mean": float(sub["macro_f1"].mean()),
            "macro_std": float(sub["macro_f1"].std(ddof=1)) if len(sub) > 1 else 0.0,
            "worst_mean": float(sub["worst_well_f1"].mean()),
            "worst_std": float(sub["worst_well_f1"].std(ddof=1)) if len(sub) > 1 else 0.0,
        }
        if m != "lgbm_global" and "lgbm_global" in pivot.columns and m in pivot.columns:
            paired = pivot[["lgbm_global", m]].dropna()
            if len(paired) >= 2:
                delta = paired[m] - paired["lgbm_global"]
                rec["delta_vs_global_mean"] = float(delta.mean())
                rec["delta_vs_global_std"] = float(delta.std(ddof=1))
                # one-sided: how often hybrid beats global
                rec["n_seeds_beat_global"] = int((delta > 0).sum())
                rec["n_seeds_paired"] = int(len(delta))
                # simple paired t (n small; report cautiously)
                if len(delta) >= 2 and delta.std(ddof=1) > 1e-12:
                    t = float(delta.mean() / (delta.std(ddof=1) / np.sqrt(len(delta))))
                    rec["paired_t_vs_global"] = t
                else:
                    rec["paired_t_vs_global"] = float("nan")
        stats_rows.append(rec)

    out = pd.DataFrame(stats_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "table_multiseed_stats.csv", index=False)
    all_df.to_csv(OUT / "protocol1_all_seeds_long.csv", index=False)
    print(out.to_string(index=False), flush=True)
    return out


def hard_well_diagnosis() -> None:
    """Compare well-level F1 across methods for known hard wells + seed44."""
    records = []
    for seed in (42, 43, 44, 45):
        p = RESULTS_DIR / f"details_protocol1_random_wells_seed{seed}.json"
        if not p.exists():
            continue
        details = json.loads(p.read_text(encoding="utf-8"))
        for item in details:
            method = item.get("method")
            if method not in {
                "lgbm_global",
                "lgbm_wellnorm",
                "lgbm_hybrid",
                "lgbm_hybrid_sim",
                "lgbm_hybrid_plus",
                "lgbm_hybrid_smooth",
            }:
                continue
            for row in item.get("well_table", []):
                records.append(
                    {
                        "seed": seed,
                        "method": method,
                        "well_id": row["well_id"],
                        "n": row["n"],
                        "macro_f1": row["macro_f1"],
                    }
                )
    if not records:
        print("no well tables", flush=True)
        return
    df = pd.DataFrame(records)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "well_level_all_seeds.csv", index=False)

    # LOWO hard wells from paper narrative
    lowo_hard = ["15/9-23", "7/1-1", "17/11-1"]
    # Protocol-3 details
    lowo_rows = []
    for p in sorted(RESULTS_DIR.glob("summary_protocol3_lowo_*_seed42.csv")):
        well = p.name.replace("summary_protocol3_lowo_well", "").replace("_seed42.csv", "")
        well = well.replace("_", "/", 1) if well.count("_") >= 1 else well
        # filenames like well15_9-23
        stem = p.stem  # summary_protocol3_lowo_well15_9-23_seed42
        wpart = stem.split("lowo_well")[-1].replace("_seed42", "")
        well_id = wpart.replace("_", "/", 1)
        t = pd.read_csv(p)
        t["target_well"] = well_id
        lowo_rows.append(t)
    if lowo_rows:
        lowo = pd.concat(lowo_rows, ignore_index=True)
        lowo.to_csv(OUT / "lowo_per_well_seed42.csv", index=False)
        focus = lowo[lowo["target_well"].isin(lowo_hard) | lowo["target_well"].str.contains("15/9-23|7/1-1|17/11-1", regex=True)]
        # also match filename-style ids
        print("LOWO sample:", lowo[["method", "macro_f1", "target_well"]].head(20).to_string(index=False), flush=True)

    # For each seed, list 3 worst wells under Global and Hybrid
    worst_notes = []
    for seed, g in df.groupby("seed"):
        for method in ("lgbm_global", "lgbm_hybrid", "lgbm_hybrid_plus"):
            sub = g[g["method"] == method].nsmallest(3, "macro_f1")
            for _, r in sub.iterrows():
                worst_notes.append(r.to_dict())
    pd.DataFrame(worst_notes).to_csv(OUT / "worst_wells_by_seed_method.csv", index=False)

    # seed44: Hybrid vs Global per well
    s44 = df[df["seed"] == 44]
    if not s44.empty and {"lgbm_global", "lgbm_hybrid"} <= set(s44["method"]):
        a = s44[s44["method"] == "lgbm_global"][["well_id", "macro_f1", "n"]].rename(
            columns={"macro_f1": "global_f1"}
        )
        b = s44[s44["method"] == "lgbm_hybrid"][["well_id", "macro_f1"]].rename(
            columns={"macro_f1": "hybrid_f1"}
        )
        m = a.merge(b, on="well_id")
        m["delta"] = m["hybrid_f1"] - m["global_f1"]
        m = m.sort_values("delta")
        m.to_csv(OUT / "seed44_hybrid_minus_global_by_well.csv", index=False)
        print("seed44 worst Hybrid deltas:\n", m.head(8).to_string(index=False), flush=True)

        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.axvline(0, color="#888", lw=0.8)
        colors = np.where(m["delta"] >= 0, "#2a6f6f", "#b85c38")
        ax.barh(m["well_id"].astype(str), m["delta"], color=colors, zorder=3)
        ax.set_xlabel("Hybrid − Global Macro-F1 (Protocol-1 seed 44)")
        ax.set_title("Seed-44 well-level gap contributes to multi-seed variability")
        _save(fig, "fig_followup_seed44_well_delta.png")


def _prepare_p1(seed: int = 42):
    df = load_labeled()
    split_path = ROOT / "data" / "splits" / f"protocol1_seed{seed}.json"
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
    else:
        split = protocol1_random_wells(df, seed=seed)
    parts = apply_split(df, split)
    return parts, split


def _fit_predict(
    parts,
    *,
    feat_mode: str,
    use_sim: bool = False,
    use_conf: bool = False,
    use_smooth: bool = False,
    temperature: float = 1.0,
    window: int = 7,
    seed: int = 42,
    n_estimators: int = 180,
):
    pipe_mode, use_rel = _feature_routing(_MODE_TO_METHOD.get(feat_mode, "lgbm_hybrid"))
    pipe = FeaturePipeline(mode=pipe_mode, use_missing_mask=True, use_relative_depth=use_rel)
    tr = pipe.fit_transform(parts["train"])
    va = pipe.transform(parts["val"])
    te = pipe.transform(parts["test"])
    classes = sorted(pd.unique(tr.y).tolist())
    mapping = {int(c): i for i, c in enumerate(classes)}
    inv = {i: int(c) for c, i in mapping.items()}

    def filt(b):
        m = np.array([int(v) in mapping for v in b.y])
        y = np.array([mapping[int(v)] for v in b.y[m]], dtype=np.int32)
        return b.X[m], y, b.conf[m], b.wells[m], b.depths[m]

    Xtr, ytr, ctr, wtr, _ = filt(tr)
    Xva, yva, cva, wva, dva = filt(va)
    Xte, yte, cte, wte, dte = filt(te)

    ext = None
    if use_sim:
        ext = similarity_sample_weights(
            Xtr, wtr, Xte, wte, feature_names=tr.feature_names, temperature=temperature
        )
    model = LGBMLithologyModel(
        config=LGBMConfig(random_state=seed, n_estimators=n_estimators),
        use_confidence=use_conf,
        external_weights=ext,
    )
    model.fit(Xtr, ytr, ctr, wtr, X_val=Xva, y_val=yva)
    pred = model.predict(Xte)
    if use_smooth:
        pred = smooth_predictions_by_well(pred, wte, dte, window=window)
    return yte, pred, wte, inv


def per_class_f1(seed: int = 42) -> None:
    print("per-class F1 (Protocol-1 seed42)...", flush=True)
    parts, _ = _prepare_p1(seed)
    configs = [
        ("Global", dict(feat_mode="global")),
        ("Hybrid", dict(feat_mode="hybrid")),
        ("Hybrid+Plus", dict(feat_mode="hybrid", use_sim=True, use_conf=True, use_smooth=True)),
    ]
    rows = []
    for name, kw in configs:
        t0 = time.time()
        yte, pred, wte, inv = _fit_predict(parts, seed=seed, **kw)
        labels = sorted(inv.keys())
        f1s = f1_score(yte, pred, labels=labels, average=None, zero_division=0)
        macro = float(f1_score(yte, pred, average="macro", zero_division=0))
        print(f"  {name}: macro={macro:.3f} ({time.time()-t0:.0f}s)", flush=True)
        for lab, f1 in zip(labels, f1s):
            code = inv[lab]
            rows.append(
                {
                    "method": name,
                    "code": code,
                    "class": LITHOLOGY_MAP.get(code, str(code)),
                    "support": int((yte == lab).sum()),
                    "f1": float(f1),
                    "macro_f1": macro,
                }
            )
    tab = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "per_class_f1_protocol1_seed42.csv", index=False)

    # plot rare vs common for Hybrid vs Global
    wide = tab.pivot_table(index=["class", "support"], columns="method", values="f1").reset_index()
    wide = wide.sort_values("support")
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(wide))
    w = 0.36
    ax.bar(x - w / 2, wide.get("Global", 0), w, label="Global", color="#1f4e79", edgecolor="black", linewidth=0.4, zorder=3)
    ax.bar(x + w / 2, wide.get("Hybrid", 0), w, label="Hybrid", color="#d94801", edgecolor="black", linewidth=0.4, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c}\n(n={int(s)})" for c, s in zip(wide["class"], wide["support"])],
        rotation=30,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Per-class F1")
    ax.set_title("Protocol-1 seed 42: per-class F1 (Global vs Hybrid)")
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.05)
    _save(fig, "fig_followup_per_class_f1.png")


def sensitivity_temperature(seed: int = 42) -> None:
    print("temperature sensitivity...", flush=True)
    parts, _ = _prepare_p1(seed)
    rows = []
    for T in (0.5, 1.0, 2.0, 4.0):
        t0 = time.time()
        yte, pred, wte, inv = _fit_predict(
            parts, feat_mode="hybrid", use_sim=True, temperature=T, seed=seed, n_estimators=180
        )
        ev = evaluate_split(yte, pred, wte)
        macro = float(ev["overall"]["macro_f1"])
        worst = float(ev["well_summary"]["worst_well_f1"])
        rows.append({"temperature": T, "macro_f1": macro, "worst_well_f1": worst, "seconds": time.time() - t0})
        print(f"  T={T}: macro={macro:.3f} worst={worst:.3f}", flush=True)
    tab = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "sensitivity_temperature.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(tab["temperature"], tab["macro_f1"], "o-", color="#2a6f6f", label="Macro-F1")
    ax.plot(tab["temperature"], tab["worst_well_f1"], "s--", color="#b85c38", label="Worst-well F1")
    ax.set_xlabel("Similarity temperature T")
    ax.set_ylabel("F1")
    ax.set_title("Hybrid+Sim: temperature sensitivity (P1 seed 42)")
    ax.legend(frameon=False)
    _save(fig, "fig_followup_temperature.png")


def sensitivity_smooth_window(seed: int = 42) -> None:
    print("smooth-window sensitivity...", flush=True)
    parts, _ = _prepare_p1(seed)
    _, use_rel = _feature_routing("lgbm_hybrid")
    pipe = FeaturePipeline(mode="hybrid", use_missing_mask=True, use_relative_depth=use_rel)
    tr = pipe.fit_transform(parts["train"])
    te = pipe.transform(parts["test"])
    va = pipe.transform(parts["val"])
    classes = sorted(pd.unique(tr.y).tolist())
    mapping = {int(c): i for i, c in enumerate(classes)}

    def filt(b):
        m = np.array([int(v) in mapping for v in b.y])
        y = np.array([mapping[int(v)] for v in b.y[m]], dtype=np.int32)
        return b.X[m], y, b.conf[m], b.wells[m], b.depths[m]

    Xtr, ytr, ctr, wtr, _ = filt(tr)
    Xte, yte, cte, wte, dte = filt(te)
    Xva, yva, cva, wva, dva = filt(va)
    model = LGBMLithologyModel(config=LGBMConfig(random_state=seed, n_estimators=180))
    model.fit(Xtr, ytr, ctr, wtr, X_val=Xva, y_val=yva)
    pred0 = model.predict(Xte)

    out_rows = []
    for window in (1, 3, 5, 7, 11, 15):
        pred = pred0.copy() if window <= 1 else smooth_predictions_by_well(pred0, wte, dte, window=window)
        ev = evaluate_split(yte, pred, wte)
        out_rows.append(
            {
                "window": window,
                "macro_f1": float(ev["overall"]["macro_f1"]),
                "worst_well_f1": float(ev["well_summary"]["worst_well_f1"]),
            }
        )
        print(
            f"  window={window}: macro={out_rows[-1]['macro_f1']:.3f} "
            f"worst={out_rows[-1]['worst_well_f1']:.3f}",
            flush=True,
        )
    tab = pd.DataFrame(out_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "sensitivity_smooth_window.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(tab["window"], tab["macro_f1"], "o-", color="#2a6f6f", label="Macro-F1")
    ax.plot(tab["window"], tab["worst_well_f1"], "s--", color="#b85c38", label="Worst-well F1")
    ax.set_xlabel("Majority-smooth window (samples)")
    ax.set_ylabel("F1")
    ax.set_title("Hybrid + depth smoothing: window sensitivity (P1 seed 42)")
    ax.legend(frameon=False)
    _save(fig, "fig_followup_smooth_window.png")


def main():
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== multiseed stats ===", flush=True)
    multiseed_stats()
    print("=== hard wells ===", flush=True)
    hard_well_diagnosis()
    print("=== per-class ===", flush=True)
    per_class_f1(42)
    print("=== temperature ===", flush=True)
    sensitivity_temperature(42)
    print("=== smooth window ===", flush=True)
    sensitivity_smooth_window(42)
    print("DONE followup", flush=True)


if __name__ == "__main__":
    main()
