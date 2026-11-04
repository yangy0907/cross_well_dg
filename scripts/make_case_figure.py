"""Plot lithofacies prediction case panels for selected wells (paper figure).

Memory-safe: trains on a subsample, predicts only case wells, and renders
strips via imshow (not per-sample fill_betweenx).

Outputs:
  - fig8_case_well_strips.png
  - fig9_protocol1_worst_strips.png  (True | Global-LGBM | Hybrid+Plus on worst wells)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import FIGURES_DIR, RESULTS_DIR, WELL_COL, LITHOLOGY_MAP
from src.features import FeaturePipeline
from src.models import LGBMConfig, LGBMLithologyModel
from src.prepare_data import load_labeled
from src.postprocess import similarity_sample_weights
from src.splits import apply_split
from src.train import _feature_routing

PAPER_DIR = RESULTS_DIR / "paper_figures"
DPI = 300
MAX_TRAIN = 80_000
MAX_DEPTH_ROWS = 2_500

# Stable categorical colours for FORCE codes present in plots
_FACIES_COLORS = {
    30000: "#7fc97f",
    65000: "#beaed4",
    65030: "#fdc086",
    70000: "#ffff99",
    70032: "#386cb0",
    74000: "#f0027f",
    80000: "#bf5b17",
    86000: "#666666",
    88000: "#1b9e77",
    90000: "#d95f02",
    93000: "#7570b3",
    99000: "#e7298a",
}

def _style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "figure.dpi": 120,
            "savefig.dpi": DPI,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _worst_wells(n: int = 2) -> list:
    details = RESULTS_DIR / "details_protocol1_random_wells_seed42.json"
    if not details.exists():
        return []
    det = json.loads(details.read_text(encoding="utf-8"))
    item = next((d for d in det if d.get("method") == "lgbm_hybrid_plus"), None)
    if item is None:
        item = next((d for d in det if d.get("method") == "lgbm_hybrid_sim"), None)
    if item is None:
        return []
    wt = pd.DataFrame(item["well_table"]).sort_values("macro_f1")
    return wt["well_id"].head(n).tolist()


def _select_wells(split: dict) -> list:
    details = RESULTS_DIR / "details_protocol1_random_wells_seed42.json"
    wells: list = []
    if details.exists():
        det = json.loads(details.read_text(encoding="utf-8"))
        item = next((d for d in det if d.get("method") == "lgbm_hybrid_sim"), None)
        if item is None:
            item = next((d for d in det if "hybrid" in str(d.get("method"))), det[-1])
        wt = pd.DataFrame(item["well_table"]).sort_values("macro_f1")
        wells = wt["well_id"].head(2).tolist()
        if len(wt) > 5:
            wells.append(wt["well_id"].iloc[len(wt) // 2])
    if not wells:
        wells = split["test_wells"][:3]
    return wells


def _train_predict(parts, method="lgbm_hybrid_sim", seed=42, feat_mode=None):
    feat_mode, use_rel, well_z_mode = _feature_routing(method)
    pipe = FeaturePipeline(
        mode=feat_mode,
        use_missing_mask=True,
        use_relative_depth=use_rel,
        well_z_mode=well_z_mode,
    )
    print(f"fitting features ({feat_mode}, rel_depth={use_rel}, well_z={well_z_mode})...", flush=True)
    tr = pipe.fit_transform(parts["train"])
    te = pipe.transform(parts["test"])
    classes = sorted(pd.unique(tr.y).tolist())
    mapping = {int(c): i for i, c in enumerate(classes)}
    inv = {i: int(c) for c, i in mapping.items()}

    def filt(b):
        m = np.array([int(v) in mapping for v in b.y])
        y = np.array([mapping[int(v)] for v in b.y[m]], dtype=np.int32)
        return b.X[m], y, b.conf[m], b.wells[m], b.depths[m]

    Xtr, ytr, ctr, wtr, _ = filt(tr)
    Xte, yte, cte, wte, dte = filt(te)
    ext = None
    if method in {"lgbm_hybrid_sim", "lgbm_hybrid_plus"}:
        print("similarity weights...", flush=True)
        ext = similarity_sample_weights(Xtr, wtr, Xte, wte, feature_names=tr.feature_names)
    model = LGBMLithologyModel(
        config=LGBMConfig(random_state=seed, n_estimators=180),
        use_confidence=method == "lgbm_hybrid_plus",
        external_weights=ext,
    )
    print(f"training {method} on {len(ytr)} rows...", flush=True)
    model.fit(Xtr, ytr, ctr, wtr)
    print(f"predicting {len(yte)} rows...", flush=True)
    pred = model.predict(Xte)
    return dte, yte, pred, wte, inv


def _palette_from_inv(inv: dict) -> tuple[np.ndarray, list]:
    """Return RGBA palette indexed by remapped class id, and legend handles info."""
    n = max(inv.keys()) + 1 if inv else 1
    palette = np.zeros((max(n, 20), 4), dtype=np.float32)
    legend_items = []
    fallback = plt.cm.tab20(np.linspace(0, 1, 20))
    for i, code in inv.items():
        rgba = matplotlib.colors.to_rgba(_FACIES_COLORS.get(code, fallback[i % 20]))
        palette[i] = rgba
        legend_items.append((LITHOLOGY_MAP.get(code, str(code)), rgba))
    # unique by name order
    seen = {}
    for name, rgba in legend_items:
        seen.setdefault(name, rgba)
    return palette, list(seen.items())


def _add_facies_legend(fig, legend_items):
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=rgba, edgecolor="none", label=name) for name, rgba in legend_items]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(6, max(1, len(handles))),
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.02),
    )


def _downsample_many(depth: np.ndarray, cols: list[np.ndarray], max_rows: int):
    n = len(depth)
    if n <= max_rows:
        return depth, cols
    idx = np.linspace(0, n - 1, max_rows).astype(np.int64)
    return depth[idx], [c[idx] for c in cols]


def _downsample_strip(depth: np.ndarray, yt: np.ndarray, yp: np.ndarray, max_rows: int):
    depth, cols = _downsample_many(depth, [yt, yp], max_rows)
    return depth, cols[0], cols[1]


def _draw_multi_strips(ax, depth, columns: list[np.ndarray], labels: list[str], palette):
    n_pal = len(palette)
    k = len(columns)
    img = np.empty((len(depth), k, 4), dtype=np.float32)
    for j, col in enumerate(columns):
        img[:, j] = palette[np.asarray(col, dtype=np.int64) % n_pal]
    dmin, dmax = float(depth.min()), float(depth.max())
    ax.imshow(
        img,
        aspect="auto",
        extent=[0.0, float(k), dmax, dmin],
        interpolation="nearest",
        origin="upper",
    )
    ax.set_xlim(-0.1, k + 0.1)
    ax.set_ylim(dmax, dmin)
    ax.set_xticks(np.arange(k) + 0.5)
    ax.set_xticklabels(labels, fontsize=8)
    ax.tick_params(axis="y", length=3)


def _save(fig, name: str):
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    gse_dir = ROOT / "paper" / "gse" / "figs"
    gse_dir.mkdir(parents=True, exist_ok=True)
    out = PAPER_DIR / name
    print(f"saving {out} @ {DPI} dpi...", flush=True)
    for folder in (PAPER_DIR, FIGURES_DIR, gse_dir):
        fig.savefig(folder / name, dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"saved {out}", flush=True)


def _prepare_parts(wells: list):
    split = json.loads((ROOT / "data" / "splits" / "protocol1_seed42.json").read_text(encoding="utf-8"))
    print("loading labeled data...", flush=True)
    df = load_labeled()
    parts = apply_split(df, split)
    rng = np.random.default_rng(42)
    tr = parts["train"]
    if len(tr) > MAX_TRAIN:
        tr = tr.iloc[rng.choice(len(tr), MAX_TRAIN, replace=False)].copy()
    te = parts["test"]
    te = te[te[WELL_COL].isin(wells)].copy()
    print(f"train subsample={len(tr)}, case test rows={len(te)}", flush=True)
    return {"train": tr, "val": parts["val"], "test": te}, wells


def make_fig8():
    split = json.loads((ROOT / "data" / "splits" / "protocol1_seed42.json").read_text(encoding="utf-8"))
    wells = _select_wells(split)
    parts_s, wells = _prepare_parts(wells)
    dte, yte, pred, wte, inv = _train_predict(parts_s, method="lgbm_hybrid_sim", feat_mode="hybrid")

    n = min(3, len(wells))
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 5.4), sharey=False)
    if n == 1:
        axes = [axes]
    palette, legend_items = _palette_from_inv(inv)

    for ax, well in zip(axes, wells[:n]):
        m = wte == well
        if m.sum() == 0:
            ax.set_title(str(well) + " (missing)")
            continue
        order = np.argsort(dte[m])
        depth = dte[m][order]
        yt = yte[m][order]
        yp = pred[m][order]
        depth, yt, yp = _downsample_strip(depth, yt, yp, MAX_DEPTH_ROWS)
        print(f"drawing {well}: {len(depth)} rows", flush=True)
        _draw_multi_strips(ax, depth, [yt, yp], ["True", "Hybrid+Sim"], palette)
        ax.set_ylabel("Depth (m)")
        ax.set_title(str(well), fontsize=10)

    fig.suptitle("Lithofacies strips: ground truth versus Hybrid+Sim", fontsize=11, y=1.02)
    _add_facies_legend(fig, legend_items)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, "fig8_case_well_strips.png")


def make_fig9_worst():
    """Protocol-1 worst wells: True | Global | Hybrid+Plus."""
    wells = _worst_wells(2)
    if len(wells) < 1:
        print("skip fig9: no worst-well list", flush=True)
        return
    parts_s, wells = _prepare_parts(wells)

    d_g, y_g, p_g, w_g, inv_g = _train_predict(parts_s, method="lgbm_global", feat_mode="global")
    d_h, y_h, p_h, w_h, inv_h = _train_predict(parts_s, method="lgbm_hybrid_plus", feat_mode="hybrid")
    # Prefer hybrid inv for legend (superset of codes seen on these wells)
    inv = inv_h if len(inv_h) >= len(inv_g) else inv_g
    palette, legend_items = _palette_from_inv(inv)

    n = len(wells)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 5.6), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, well in zip(axes, wells):
        mg = w_g == well
        mh = w_h == well
        if mg.sum() == 0 or mh.sum() == 0:
            ax.set_title(str(well) + " (missing)")
            continue
        og = np.argsort(d_g[mg])
        oh = np.argsort(d_h[mh])
        depth = d_g[mg][og]
        yt = y_g[mg][og]
        pg = p_g[mg][og]
        ph = p_h[mh][oh]
        mlen = min(len(depth), len(ph))
        depth, yt, pg, ph = depth[:mlen], yt[:mlen], pg[:mlen], ph[:mlen]
        depth, cols = _downsample_many(depth, [yt, pg, ph], MAX_DEPTH_ROWS)
        yt, pg, ph = cols
        print(f"worst-well strip {well}: {len(depth)} rows", flush=True)
        _draw_multi_strips(ax, depth, [yt, pg, ph], ["True", "Global-LGBM", "Hybrid+Plus"], palette)
        ax.set_ylabel("Depth (m)")
        ax.set_title(f"{well} (Protocol-1 worst)", fontsize=10)

    fig.suptitle("Protocol-1 worst wells: True vs Global-LGBM vs Hybrid+Plus", fontsize=11, y=1.02)
    _add_facies_legend(fig, legend_items)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, "fig9_protocol1_worst_strips.png")


def main():
    _style()
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    make_fig8()
    make_fig9_worst()


if __name__ == "__main__":
    main()
