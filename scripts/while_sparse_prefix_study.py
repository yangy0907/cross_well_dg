"""Sparse-prefix while-drilling stress test (Protocol-1 seed 42).

Two complementary views on pipeline r3 Hybrid-while:

1) Early-bit windows: score only labelled depths whose full-curve metres
   from well top are below W (W in {50, 100, 200, 500}).
2) Sliding causal window: recompute Hybrid-while well-z using only the last
   ``cap`` full-curve samples before each depth (cap in {50, 100, 500, full}).

Outputs:
  results/followup/while_sparse_prefix_seed42.csv
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("LGBM_NUM_THREADS", "2")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import DEPTH_COL, RESULTS_DIR, WELL_COL
from src.features import FeaturePipeline
from src.metrics import evaluate_split
from src.models import LGBMConfig, LGBMLithologyModel
from src.prepare_data import load_curves_all, load_labeled
from src.splits import apply_split, protocol1_random_wells
from src.train import PIPELINE_VERSION, set_curve_context

OUT = RESULTS_DIR / "followup"
OUT.mkdir(parents=True, exist_ok=True)


def _causal_z_window(
    depths_query: np.ndarray,
    depths_full: np.ndarray,
    values_full: np.ndarray,
    cap: int | None,
) -> np.ndarray:
    """Causal z with optional trailing window of ``cap`` valid samples."""
    order = np.argsort(depths_full, kind="mergesort")
    d_ord = depths_full[order]
    s_ord = values_full[order]
    valid = ~np.isnan(s_ord)
    z_ord = np.zeros(len(s_ord), dtype=np.float64)
    vals: list[float] = []
    for i in range(len(s_ord)):
        if valid[i]:
            vals.append(float(s_ord[i]))
            if cap is not None and len(vals) > cap:
                vals = vals[-cap:]
        if not vals:
            z_ord[i] = 0.0
            continue
        arr = np.asarray(vals, dtype=np.float64)
        mu = float(arr.mean())
        sd = float(arr.std(ddof=0))
        if sd < 1e-12:
            sd = 1.0
        z_ord[i] = (float(s_ord[i]) - mu) / (sd + 1e-6) if valid[i] else 0.0

    z_out = np.zeros(len(depths_query), dtype=np.float64)
    j = -1
    for i, dq in enumerate(depths_query):
        while j + 1 < len(d_ord) and d_ord[j + 1] <= dq + 1e-9:
            j += 1
        if j >= 0:
            z_out[i] = z_ord[j]
    return z_out


def _apply_windowed_well_z(
    pipe: FeaturePipeline,
    bundle_X: np.ndarray,
    feat_names: list[str],
    out_df: pd.DataFrame,
    base_curves: list[str],
    cap: int | None,
) -> np.ndarray:
    """Replace *_wn columns with windowed causal z (row-aligned to out_df)."""
    X = bundle_X.copy()
    name_to_i = {n: i for i, n in enumerate(feat_names)}
    ctx = pipe.curve_context
    assert ctx is not None
    ctx_by = {str(w): g for w, g in ctx.groupby(WELL_COL, sort=False)}
    out = out_df.reset_index(drop=True)
    assert len(out) == len(X)
    for c in base_curves:
        col = f"{c}_wn"
        if col not in name_to_i:
            continue
        j = name_to_i[col]
        z = np.zeros(len(out), dtype=np.float64)
        for w, idx in out.groupby(WELL_COL, sort=False).groups.items():
            positions = np.asarray(idx, dtype=np.int64)
            g_out = out.iloc[positions]
            g_ctx = ctx_by.get(str(w))
            if g_ctx is None or c not in g_ctx.columns:
                continue
            z_w = _causal_z_window(
                g_out[DEPTH_COL].to_numpy(dtype=np.float64),
                g_ctx[DEPTH_COL].to_numpy(dtype=np.float64),
                g_ctx[c].to_numpy(dtype=np.float64),
                cap,
            )
            z[positions] = z_w
        X[:, j] = z
    return X


def _pack(y, mapping, unk_id):
    return np.array(
        [mapping[int(v)] if int(v) in mapping else unk_id for v in y], dtype=np.int32
    )


def main() -> None:
    print("[sparse] load", flush=True)
    df = load_labeled()
    curves = load_curves_all()
    set_curve_context(curves)
    split = protocol1_random_wells(df, seed=42)
    parts = apply_split(df, split)

    pipe = FeaturePipeline(
        mode="hybrid",
        use_relative_depth=False,
        well_z_mode="causal",
        curve_context=curves,
    )
    t0 = time.time()
    tr = pipe.fit_transform(parts["train"])
    va = pipe.transform(parts["val"])
    te = pipe.transform(parts["test"])
    print(f"[sparse] featurize {time.time()-t0:.1f}s", flush=True)

    classes = sorted(int(c) for c in pd.unique(tr.y) if int(c) >= 0)
    mapping = {c: i for i, c in enumerate(classes)}
    unk = len(mapping)
    mask_tr = np.array([int(v) in mapping for v in tr.y])
    mask_va = np.array([int(v) in mapping for v in va.y])
    Xtr, ytr = tr.X[mask_tr], np.array([mapping[int(v)] for v in tr.y[mask_tr]], dtype=np.int32)
    Xva, yva = va.X[mask_va], np.array([mapping[int(v)] for v in va.y[mask_va]], dtype=np.int32)
    yte = _pack(te.y, mapping, unk)

    # Reference Hybrid-while (full expanding causal)
    model = LGBMLithologyModel(config=LGBMConfig(random_state=42, n_estimators=180))
    model.fit(Xtr, ytr, np.ones(len(ytr)), tr.wells[mask_tr], X_val=Xva, y_val=yva)
    pred_full = model.predict(te.X)
    ev_full = evaluate_split(yte, pred_full, te.wells)

    # Metres from well top on full curve
    well_depths = {
        str(w): np.sort(g[DEPTH_COL].to_numpy(dtype=np.float64))
        for w, g in curves.groupby(WELL_COL, sort=False)
    }
    metres = np.zeros(len(te.wells), dtype=np.float64)
    for i, (w, d) in enumerate(zip(te.wells.astype(str), te.depths)):
        arr = well_depths.get(str(w))
        if arr is None or arr.size == 0:
            continue
        metres[i] = float(d - arr[0])

    rows = []
    rows.append(
        {
            "study": "reference_hybrid_while",
            "param": "full_expanding",
            "n_rows": int(len(yte)),
            "macro_f1": ev_full["overall"]["macro_f1"],
            "worst_well_f1": ev_full["well_summary"]["worst_well_f1"],
            "pipeline_version": PIPELINE_VERSION,
        }
    )

    for w in (50, 100, 200, 500):
        m = metres < float(w)
        if m.sum() < 50:
            continue
        ev = evaluate_split(yte[m], pred_full[m], te.wells[m])
        rows.append(
            {
                "study": "early_bit_window",
                "param": f"metres_lt_{w}",
                "n_rows": int(m.sum()),
                "macro_f1": ev["overall"]["macro_f1"],
                "worst_well_f1": ev["well_summary"]["worst_well_f1"],
                "pipeline_version": PIPELINE_VERSION,
            }
        )
        print(f"[sparse] early <{w}m n={m.sum()} macro={ev['overall']['macro_f1']:.3f}", flush=True)

    # Sliding-window causal features + retrain
    base_curves = [n[: -len("_wn")] for n in tr.feature_names if n.endswith("_wn")]
    for cap in (50, 100, 500, None):
        tag = "full" if cap is None else str(cap)
        print(f"[sparse] window cap={tag}", flush=True)
        t1 = time.time()
        Xtr_w = _apply_windowed_well_z(
            pipe, tr.X, list(tr.feature_names), parts["train"], base_curves, cap
        )
        Xva_w = _apply_windowed_well_z(
            pipe, va.X, list(va.feature_names), parts["val"], base_curves, cap
        )
        Xte_w = _apply_windowed_well_z(
            pipe, te.X, list(te.feature_names), parts["test"], base_curves, cap
        )
        mdl = LGBMLithologyModel(config=LGBMConfig(random_state=42, n_estimators=180))
        mdl.fit(
            Xtr_w[mask_tr],
            ytr,
            np.ones(len(ytr)),
            tr.wells[mask_tr],
            X_val=Xva_w[mask_va],
            y_val=yva,
        )
        pred = mdl.predict(Xte_w)
        ev = evaluate_split(yte, pred, te.wells)
        rows.append(
            {
                "study": "sliding_causal_window",
                "param": f"cap_{tag}",
                "n_rows": int(len(yte)),
                "macro_f1": ev["overall"]["macro_f1"],
                "worst_well_f1": ev["well_summary"]["worst_well_f1"],
                "pipeline_version": PIPELINE_VERSION,
                "seconds": round(time.time() - t1, 2),
            }
        )
        print(
            f"[sparse] cap={tag} macro={ev['overall']['macro_f1']:.3f} "
            f"worst={ev['well_summary']['worst_well_f1']:.3f} ({time.time()-t1:.0f}s)",
            flush=True,
        )

    out = OUT / "while_sparse_prefix_seed42.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[sparse] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
