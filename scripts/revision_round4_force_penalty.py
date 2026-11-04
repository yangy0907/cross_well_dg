"""Build FORCE penalty matrix from official starter-notebook dump and score Protocol-1."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import RESULTS_DIR
from src.features import FeaturePipeline
from src.metrics import summarize_well_f1, well_wise_f1
from src.models import LGBMConfig, LGBMLithologyModel
from src.prepare_data import load_labeled
from src.splits import apply_split

OUT = RESULTS_DIR / "followup"
DATA = ROOT / "data"

# Official FORCE label index order (test_code.py / starter notebook)
LITHOLOGY_TO_IDX = {
    30000: 0,   # Sandstone
    65030: 1,   # Sandstone/Shale
    65000: 2,   # Shale
    80000: 3,   # Marl
    74000: 4,   # Dolomite
    70000: 5,   # Limestone
    70032: 6,   # Chalk
    88000: 7,   # Halite
    86000: 8,   # Anhydrite
    99000: 9,   # Tuff
    90000: 10,  # Coal
    93000: 11,  # Basement
}

# From official starter_notebook.ipynb text/plain dump of A (12 x 12); A[true, pred]
PENALTY = np.array(
    [
        [0.0, 2.0, 3.5, 3.0, 3.75, 3.5, 3.5, 4.0, 4.0, 2.5, 3.875, 3.25],
        [2.0, 0.0, 2.375, 2.75, 4.0, 3.75, 3.75, 3.875, 4.0, 3.0, 3.75, 3.0],
        [3.5, 2.375, 0.0, 2.0, 3.5, 3.5, 3.75, 4.0, 4.0, 2.75, 3.25, 3.0],
        [3.0, 2.75, 2.0, 0.0, 2.5, 2.0, 2.25, 4.0, 4.0, 3.375, 3.75, 3.25],
        [3.75, 4.0, 3.5, 2.5, 0.0, 2.625, 2.875, 3.75, 3.25, 3.0, 4.0, 3.625],
        [3.5, 3.75, 3.5, 2.0, 2.625, 0.0, 1.375, 4.0, 3.75, 3.5, 4.0, 3.625],
        [3.5, 3.75, 3.75, 2.25, 2.875, 1.375, 0.0, 4.0, 3.75, 3.125, 4.0, 3.75],
        [4.0, 3.875, 4.0, 4.0, 3.75, 4.0, 4.0, 0.0, 2.75, 3.75, 3.75, 4.0],
        [4.0, 4.0, 4.0, 4.0, 3.25, 3.75, 3.75, 2.75, 0.0, 4.0, 4.0, 3.875],
        [2.5, 3.0, 2.75, 3.375, 3.0, 3.5, 3.125, 3.75, 4.0, 0.0, 2.5, 3.25],
        [3.875, 3.75, 3.25, 3.75, 4.0, 4.0, 4.0, 3.75, 4.0, 2.5, 0.0, 4.0],
        [3.25, 3.0, 3.0, 3.25, 3.625, 3.625, 3.75, 4.0, 3.875, 3.25, 4.0, 0.0],
    ],
    dtype=np.float64,
)


def force_penalty_score(y_true_codes: np.ndarray, y_pred_codes: np.ndarray) -> float:
    """Official FORCE score: mean(-A[true, pred]); closer to 0 is better."""
    yt = np.asarray([LITHOLOGY_TO_IDX[int(v)] for v in y_true_codes], dtype=np.int64)
    yp = np.asarray([LITHOLOGY_TO_IDX[int(v)] for v in y_pred_codes], dtype=np.int64)
    return float((-PENALTY[yt, yp]).mean())


def _run(parts, *, feat_mode: str, use_rel: bool, seed: int) -> dict:
    pipe = FeaturePipeline(
        mode=feat_mode, use_missing_mask=True, use_relative_depth=use_rel
    )
    tr = pipe.fit_transform(parts["train"])
    va = pipe.transform(parts["val"])
    te = pipe.transform(parts["test"])

    classes = sorted(pd.unique(tr.y).tolist())
    mapping = {int(c): i for i, c in enumerate(classes)}
    inv = {i: int(c) for c, i in mapping.items()}
    fixed_idx = list(range(len(classes)))

    def filt(b):
        m = np.array([int(v) in mapping for v in b.y])
        y = np.array([mapping[int(v)] for v in b.y[m]], dtype=np.int32)
        return b.X[m], y, b.conf[m], b.wells[m]

    Xtr, ytr, ctr, wtr = filt(tr)
    Xva, yva, _, _ = filt(va)
    Xte, yte, _, wte = filt(te)

    model = LGBMLithologyModel(config=LGBMConfig(random_state=seed, n_estimators=180))
    model.fit(Xtr, ytr, ctr, wtr, X_val=Xva, y_val=yva)
    pred = model.predict(Xte)

    # map back to FORCE codes for penalty
    yte_code = np.array([inv[int(v)] for v in yte], dtype=np.int64)
    pred_code = np.array([inv[int(v)] for v in pred], dtype=np.int64)
    # unseen test labels kept outside mapping are already filtered

    wt = well_wise_f1(yte, pred, wte, label_mode="well_true")
    fx = well_wise_f1(yte, pred, wte, label_mode="fixed", global_labels=fixed_idx)
    sw, sf = summarize_well_f1(wt), summarize_well_f1(fx)

    from sklearn.metrics import f1_score

    return {
        "macro_f1_pooled": float(
            f1_score(yte, pred, average="macro", zero_division=0, labels=sorted(np.unique(yte)))
        ),
        "force_penalty": force_penalty_score(yte_code, pred_code),
        "worst_well_true": sw["worst_well_f1"],
        "worst_well_fixed": sf["worst_well_f1"],
        "well_mean_true": sw["well_macro_f1_mean"],
        "well_mean_fixed": sf["well_macro_f1_mean"],
        "n_test": int(len(yte)),
        "n_features": int(Xtr.shape[1]),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    np.save(DATA / "force_penalty_matrix.npy", PENALTY)

    # sanity: perfect score is 0
    assert abs(force_penalty_score(np.array([30000, 65000]), np.array([30000, 65000]))) < 1e-12

    df = load_labeled()
    rows = []
    # Reuse existing fixed-taxonomy CSV for text; retrain only for penalty on seeds 42-45
    for seed in (42, 43, 44, 45):
        split = json.loads(
            (ROOT / "data" / "splits" / f"protocol1_seed{seed}.json").read_text(encoding="utf-8")
        )
        parts = apply_split(df, split)
        for name, mode, rel in (
            ("Global-LGBM", "global", False),
            ("Hybrid", "hybrid", True),
        ):
            t0 = time.time()
            print(f"seed={seed} {name} ...", flush=True)
            rec = _run(parts, feat_mode=mode, use_rel=rel, seed=seed)
            rec.update(
                {
                    "seed": seed,
                    "method": name,
                    "seconds": round(time.time() - t0, 1),
                }
            )
            rows.append(rec)
            print(
                f"  macro={rec['macro_f1_pooled']:.3f} "
                f"penalty={rec['force_penalty']:.4f} "
                f"worst_fixed={rec['worst_well_fixed']:.3f} "
                f"({rec['seconds']}s)",
                flush=True,
            )

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "force_penalty_protocol1.csv", index=False)
    print("wrote", OUT / "force_penalty_protocol1.csv", flush=True)

    g = tab[tab.method == "Global-LGBM"].set_index("seed")
    h = tab[tab.method == "Hybrid"].set_index("seed")
    for metric in ("force_penalty", "worst_well_fixed", "macro_f1_pooled"):
        # penalty: closer to 0 is better => Hybrid wins if Hybrid > Global (less negative)
        if metric == "force_penalty":
            wins = int((h[metric] > g[metric]).sum())
            delta = (h[metric] - g[metric]).mean()
        else:
            wins = int((h[metric] > g[metric]).sum())
            delta = (h[metric] - g[metric]).mean()
        print(f"{metric}: Hybrid better {wins}/4; mean Δ={delta:+.4f}", flush=True)


if __name__ == "__main__":
    main()
