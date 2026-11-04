"""While-drilling cold-start curves under full-curve causal Hybrid-while.

For Protocol-1 seed 42: score Hybrid-while predictions stratified by how many
full-curve prefix samples (and drilled metres) precede each labeled test depth.

Also reports a simple cold-start policy: require min_prefix in {0,10,50,100}
before trusting well-z (else fall back to global features only — reported as
coverage of depths meeting the threshold).

Output: results/followup/while_coldstart_seed42.csv
"""
from __future__ import annotations

import os
import sys
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


def _build_well_depth_index(curves: pd.DataFrame) -> dict[str, np.ndarray]:
    """Sorted depth arrays per well for O(log n) prefix lookups."""
    idx: dict[str, np.ndarray] = {}
    for well, g in curves.groupby(WELL_COL, sort=False):
        depths = np.sort(g[DEPTH_COL].to_numpy(dtype=np.float64))
        idx[str(well)] = depths
    return idx


def _prefix_stats_fast(
    well_depths: dict[str, np.ndarray], well: str, depth: float
) -> tuple[int, float]:
    arr = well_depths.get(str(well))
    if arr is None or arr.size == 0:
        return 0, 0.0
    # counts of depths <= depth
    n = int(np.searchsorted(arr, depth + 1e-9, side="right"))
    if n <= 0:
        return 0, 0.0
    metres = float(arr[n - 1] - arr[0])
    return n, metres


def main() -> None:
    print("[coldstart] load data", flush=True)
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
    print("[coldstart] featurize", flush=True)
    tr = pipe.fit_transform(parts["train"])
    va = pipe.transform(parts["val"])
    te = pipe.transform(parts["test"])

    classes = sorted(pd.unique(tr.y).tolist())
    mapping = {int(c): i for i, c in enumerate(classes)}
    ytr = np.array([mapping[int(v)] for v in tr.y], dtype=np.int32)
    yva = np.array(
        [mapping[int(v)] for v in va.y if int(v) in mapping],
        dtype=np.int32,
    )
    # align va X
    mva = np.array([int(v) in mapping for v in va.y])
    Xva = va.X[mva]
    yte = np.array(
        [mapping[int(v)] if int(v) in mapping else -1 for v in te.y], dtype=np.int32
    )
    keep = yte >= 0
    Xte, yte, wte, dte = te.X[keep], yte[keep], te.wells[keep], te.depths[keep]

    print(f"[coldstart] fit Hybrid-while n_train={len(ytr)} n_test={len(yte)}", flush=True)
    n_jobs = int(os.environ.get("COLDSTART_LGBM_N_JOBS", "2"))
    print(f"coldstart memory-safe: LGBM n_jobs={n_jobs}", flush=True)
    model = LGBMLithologyModel(
        config=LGBMConfig(random_state=42, n_estimators=180, n_jobs=n_jobs)
    )
    model.fit(tr.X, ytr, tr.conf, tr.wells, X_val=Xva, y_val=yva)
    pred = model.predict(Xte)

    print("[coldstart] prefix index + score bins", flush=True)
    well_depths = _build_well_depth_index(curves)
    n_pref = np.empty(len(dte), dtype=np.int64)
    m_pref = np.empty(len(dte), dtype=np.float64)
    for i, (w, d) in enumerate(zip(wte, dte)):
        n_pref[i], m_pref[i] = _prefix_stats_fast(well_depths, str(w), float(d))

    rows = []
    # Bin by prefix sample count
    bins = [(0, 10), (10, 50), (50, 100), (100, 500), (500, 10**9)]
    for lo, hi in bins:
        m = (n_pref >= lo) & (n_pref < hi)
        if not m.any():
            continue
        ev = evaluate_split(yte[m], pred[m], wte[m])
        rows.append(
            {
                "bin": f"n_prefix[{lo},{hi})",
                "n_rows": int(m.sum()),
                "macro_f1": ev["overall"]["macro_f1"],
                "worst_well_f1": ev["well_summary"]["worst_well_f1"],
                "pipeline_version": PIPELINE_VERSION,
            }
        )

    # Bin by drilled metres
    mbins = [(0, 10), (10, 50), (50, 100), (100, 500), (500, 10**9)]
    for lo, hi in mbins:
        m = (m_pref >= lo) & (m_pref < hi)
        if not m.any():
            continue
        ev = evaluate_split(yte[m], pred[m], wte[m])
        rows.append(
            {
                "bin": f"metres[{lo},{hi})",
                "n_rows": int(m.sum()),
                "macro_f1": ev["overall"]["macro_f1"],
                "worst_well_f1": ev["well_summary"]["worst_well_f1"],
                "pipeline_version": PIPELINE_VERSION,
            }
        )

    for min_n in [0, 10, 50, 100]:
        frac = float((n_pref >= min_n).mean())
        rows.append(
            {
                "bin": f"coldstart_min_prefix>={min_n}_coverage",
                "n_rows": int((n_pref >= min_n).sum()),
                "macro_f1": frac,
                "worst_well_f1": float("nan"),
                "pipeline_version": PIPELINE_VERSION,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "while_coldstart_seed42.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
