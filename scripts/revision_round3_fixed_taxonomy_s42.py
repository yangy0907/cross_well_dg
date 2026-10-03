"""Fast seed-42 only: fixed-taxonomy + Hybrid without relative depth."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import RESULTS_DIR
from src.features import FeaturePipeline
from src.metrics import summarize_well_f1, well_wise_f1
from src.models import LGBMConfig, LGBMLithologyModel
from src.prepare_data import load_labeled
from src.splits import apply_split

OUT = RESULTS_DIR / "followup"


def _run_one(parts, *, feat_mode: str, use_rel: bool, seed: int, train_labels: list[int]) -> dict:
    pipe = FeaturePipeline(mode=feat_mode, use_missing_mask=True, use_relative_depth=use_rel)
    tr = pipe.fit_transform(parts["train"])
    va = pipe.transform(parts["val"])
    te = pipe.transform(parts["test"])
    classes = sorted(pd.unique(tr.y).tolist())
    mapping = {int(c): i for i, c in enumerate(classes)}
    fixed_idx = [mapping[c] for c in train_labels if c in mapping]

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
    wt = well_wise_f1(yte, pred, wte, label_mode="well_true")
    fx = well_wise_f1(yte, pred, wte, label_mode="fixed", global_labels=fixed_idx)
    sw, sf = summarize_well_f1(wt), summarize_well_f1(fx)
    return {
        "macro_f1_pooled": float(
            f1_score(yte, pred, average="macro", zero_division=0, labels=sorted(np.unique(yte)))
        ),
        "worst_well_true": sw["worst_well_f1"],
        "worst_well_fixed": sf["worst_well_f1"],
        "well_mean_true": sw["well_macro_f1_mean"],
        "well_mean_fixed": sf["well_macro_f1_mean"],
        "worst_well_id_true": str(wt.iloc[0]["well_id"]),
        "worst_well_id_fixed": str(fx.iloc[0]["well_id"]),
        "n_test_wells": int(len(wt)),
        "n_fixed_labels": int(len(fixed_idx)),
        "n_features": int(Xtr.shape[1]),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_labeled()
    seed = 42
    split = json.loads((ROOT / "data" / "splits" / f"protocol1_seed{seed}.json").read_text(encoding="utf-8"))
    parts = apply_split(df, split)
    train_labels = sorted(pd.unique(parts["train"]["lithology"]).astype(int).tolist())
    rows = []
    for name, mode, rel in (
        ("Global-LGBM", "global", False),
        ("Hybrid", "hybrid", True),
        ("Hybrid-no-reldepth", "hybrid", False),
    ):
        t0 = time.time()
        print(f"{name} ...", flush=True)
        rec = _run_one(parts, feat_mode=mode, use_rel=rel, seed=seed, train_labels=train_labels)
        rec.update(
            {
                "seed": seed,
                "method": name,
                "feat_mode": mode,
                "use_relative_depth": rel,
                "seconds": round(time.time() - t0, 1),
            }
        )
        rows.append(rec)
        print(
            f"  pooled={rec['macro_f1_pooled']:.3f} "
            f"true_worst={rec['worst_well_true']:.3f} "
            f"fixed_worst={rec['worst_well_fixed']:.3f} "
            f"({rec['seconds']}s)",
            flush=True,
        )
    tab = pd.DataFrame(rows)
    path = OUT / "fixed_taxonomy_protocol1_seed42.csv"
    tab.to_csv(path, index=False)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
