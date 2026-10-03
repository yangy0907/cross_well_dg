"""Round-2 diagnostics: rare-class confusion, seed44 mechanism, LOWO alt summary."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import CORE_CURVES, FIGURES_DIR, LITHOLOGY_MAP, RESULTS_DIR
from src.features import FeaturePipeline
from src.models import LGBMConfig, LGBMLithologyModel
from src.prepare_data import load_labeled
from src.splits import apply_split
from src.train import _feature_routing

OUT = RESULTS_DIR / "followup"
PAPER = RESULTS_DIR / "paper_figures"
DPI = 300
RARE = {"Coal", "Basement", "Dolomite", "Chalk", "Tuff", "Anhydrite"}
_MODE_TO_METHOD = {
    "global": "lgbm_global",
    "hybrid": "lgbm_hybrid",
    "wellnorm": "lgbm_wellnorm",
}


def _save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for folder in (OUT, PAPER, FIGURES_DIR):
        fig.savefig(folder / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {name}", flush=True)


def rare_class_confusion(seed: int = 42):
    print("rare-class confusion (Global vs Hybrid)...", flush=True)
    df = load_labeled()
    split = json.loads((ROOT / "data" / "splits" / f"protocol1_seed{seed}.json").read_text(encoding="utf-8"))
    parts = apply_split(df, split)

    def run(feat_mode: str):
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
            return b.X[m], y, b.conf[m], b.wells[m]

        Xtr, ytr, ctr, wtr = filt(tr)
        Xva, yva, cva, wva = filt(va)
        Xte, yte, cte, wte = filt(te)
        model = LGBMLithologyModel(config=LGBMConfig(random_state=seed, n_estimators=180))
        model.fit(Xtr, ytr, ctr, wtr, X_val=Xva, y_val=yva)
        pred = model.predict(Xte)
        names = [LITHOLOGY_MAP.get(inv[i], str(inv[i])) for i in range(len(inv))]
        cm = confusion_matrix(yte, pred, labels=list(range(len(inv))))
        return yte, pred, names, cm, inv

    yg, pg, names, cm_g, inv = run("global")
    yh, ph, _, cm_h, _ = run("hybrid")

    # focus rows: rare classes present
    rare_idx = [i for i, n in enumerate(names) if n in RARE]
    # also shale as sink
    shale_i = next((i for i, n in enumerate(names) if n == "Shale"), None)

    rows = []
    for i in rare_idx:
        support = int((yg == i).sum())
        if support == 0:
            continue
        # where do true rare go under Global / Hybrid
        for method, pred in (("Global", pg), ("Hybrid", ph)):
            m = yg == i
            pred_c = pred[m]
            top, counts = np.unique(pred_c, return_counts=True)
            order = np.argsort(-counts)
            top3 = [(names[int(top[j])], int(counts[j])) for j in order[:3]]
            rows.append(
                {
                    "true_class": names[i],
                    "support": support,
                    "method": method,
                    "f1": float(f1_score(yg == i, pred == i, zero_division=0)),
                    "recall": float((pred_c == i).mean()),
                    "top1_pred": top3[0][0] if top3 else "",
                    "top1_n": top3[0][1] if top3 else 0,
                    "top2_pred": top3[1][0] if len(top3) > 1 else "",
                    "shale_frac": float((pred_c == shale_i).mean()) if shale_i is not None else np.nan,
                }
            )
    tab = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "rare_class_sink_protocol1_seed42.csv", index=False)
    print(tab.to_string(index=False), flush=True)

    # heatmap of rare→predicted (Hybrid)
    if rare_idx:
        sub = cm_h[np.ix_(rare_idx, list(range(len(names))))]
        # row-normalize
        sub_n = sub.astype(float)
        rs = sub_n.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        sub_n = sub_n / rs
        fig, ax = plt.subplots(figsize=(8.5, 3.8))
        im = ax.imshow(sub_n, aspect="auto", cmap="YlOrBr", vmin=0, vmax=1)
        ax.set_yticks(range(len(rare_idx)))
        ax.set_yticklabels([names[i] for i in rare_idx])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_title("Hybrid: rare-class confusion (row-normalized, P1 seed 42)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(pred|true)")
        _save(fig, "fig_followup_rare_confusion.png")


def seed44_mechanism():
    print("seed44 mechanism diagnosis...", flush=True)
    delta_p = OUT / "seed44_hybrid_minus_global_by_well.csv"
    if not delta_p.exists():
        print("missing seed44 delta table", flush=True)
        return
    delta = pd.read_csv(delta_p)
    df = load_labeled()
    split = json.loads((ROOT / "data" / "splits" / "protocol1_seed44.json").read_text(encoding="utf-8"))
    train_wells = set(split["train_wells"])
    test_wells = set(split["test_wells"])

    # missingness + class entropy per test well
    rows = []
    curves = [c for c in CORE_CURVES if c in df.columns]
    train_df = df[df["well_id"].isin(train_wells)]
    train_prior = train_df["lithology"].value_counts(normalize=True)

    for _, r in delta.iterrows():
        w = r["well_id"]
        wdf = df[df["well_id"] == w]
        miss = float(np.mean([wdf[c].isna().mean() for c in curves])) if curves else np.nan
        prior = wdf["lithology"].value_counts(normalize=True)
        # TV distance to train prior
        keys = sorted(set(train_prior.index) | set(prior.index))
        tv = 0.5 * sum(abs(float(prior.get(k, 0)) - float(train_prior.get(k, 0))) for k in keys)
        # mean GR/RHOB if present
        gr = float(pd.to_numeric(wdf["GR"], errors="coerce").mean()) if "GR" in wdf else np.nan
        rows.append(
            {
                "well_id": w,
                "delta_hybrid_minus_global": r["delta"],
                "n": r["n"],
                "missing_frac": miss,
                "class_tv_to_train": tv,
                "n_classes": int(wdf["lithology"].nunique()),
                "gr_mean": gr,
            }
        )
    tab = pd.DataFrame(rows).sort_values("delta_hybrid_minus_global")
    tab.to_csv(OUT / "seed44_mechanism_by_well.csv", index=False)
    print(tab.head(10).to_string(index=False), flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    axes[0].scatter(tab["missing_frac"], tab["delta_hybrid_minus_global"], c="#2a6f6f", s=36, zorder=3)
    axes[0].axhline(0, color="#888", lw=0.8)
    axes[0].set_xlabel("Mean curve missingness")
    axes[0].set_ylabel("Hybrid − Global Macro-F1")
    axes[0].set_title("Seed 44: missingness vs hybrid gap")
    axes[1].scatter(tab["class_tv_to_train"], tab["delta_hybrid_minus_global"], c="#b85c38", s=36, zorder=3)
    axes[1].axhline(0, color="#888", lw=0.8)
    axes[1].set_xlabel("Class-prior TV distance to train")
    axes[1].set_ylabel("Hybrid − Global Macro-F1")
    axes[1].set_title("Seed 44: prior shift vs hybrid gap")
    fig.tight_layout()
    _save(fig, "fig_followup_seed44_mechanism.png")

    # correlations
    corr = {
        "corr_delta_missing": float(tab["delta_hybrid_minus_global"].corr(tab["missing_frac"])),
        "corr_delta_tv": float(tab["delta_hybrid_minus_global"].corr(tab["class_tv_to_train"])),
    }
    (OUT / "seed44_mechanism_corr.json").write_text(json.dumps(corr, indent=2), encoding="utf-8")
    print(corr, flush=True)


def summarize_lowo_alt():
    print("LOWO primary vs alt summary...", flush=True)
    primary = list(RESULTS_DIR.glob("summary_protocol3_lowo_well*_seed42.csv"))
    # exclude alt-tagged if any slipped into pattern
    primary = [p for p in primary if "_spaced_alt_" not in p.name]
    alt = list(RESULTS_DIR.glob("summary_protocol3_lowo_spaced_alt_well*_seed42.csv"))
    rows = []
    for label, files in (("primary", primary), ("spaced_alt", alt)):
        if not files:
            continue
        df = pd.concat([pd.read_csv(f).assign(file=f.name) for f in files], ignore_index=True)
        # recover well from filename
        g = df.groupby("method")["macro_f1"].agg(["mean", "std", "count"]).reset_index()
        g["selection"] = label
        rows.append(g)
        wells = sorted({f.name for f in files})
        print(f"{label}: {len(files)} wells/files", flush=True)
    if not rows:
        print("no LOWO summaries yet", flush=True)
        return
    out = pd.concat(rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "lowo_primary_vs_alt_means.csv", index=False)
    print(out.to_string(index=False), flush=True)

    # bar compare if both present
    if {"primary", "spaced_alt"} <= set(out["selection"]):
        methods = ["lgbm_wellnorm", "lgbm_hybrid", "lgbm_hybrid_smooth", "lgbm_hybrid_plus"]
        methods = [m for m in methods if m in set(out["method"])]
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        x = np.arange(len(methods))
        w = 0.36
        a = out[out["selection"] == "primary"].set_index("method")
        b = out[out["selection"] == "spaced_alt"].set_index("method")
        ax.bar(x - w / 2, [a.loc[m, "mean"] if m in a.index else 0 for m in methods], w, label="Primary LOWO", color="#1f4e79", edgecolor="black", linewidth=0.4)
        ax.bar(x + w / 2, [b.loc[m, "mean"] if m in b.index else 0 for m in methods], w, label="Alt spaced LOWO", color="#d94801", edgecolor="black", linewidth=0.4)
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
        ax.legend(frameon=False)
        _save(fig, "fig_followup_lowo_selection.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rare_class_confusion(42)
    seed44_mechanism()
    summarize_lowo_alt()
    print("DONE revision_round2", flush=True)


if __name__ == "__main__":
    main()
