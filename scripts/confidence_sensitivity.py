"""Confidence sensitivity using FORCE codes 1=high, 2=mid, 3=low."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import FIGURES_DIR, RESULTS_DIR, SPLITS_DIR
from src.features import FeaturePipeline
from src.metrics import evaluate_split
from src.models import LGBMConfig, LGBMLithologyModel
from src.prepare_data import load_labeled
from src.splits import apply_split


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_labeled()
    split = json.loads((SPLITS_DIR / "protocol1_seed42.json").read_text(encoding="utf-8"))
    parts = apply_split(df, split)

    rng = np.random.default_rng(42)

    def _sub(d, n):
        if len(d) <= n:
            return d
        return d.iloc[rng.choice(len(d), size=n, replace=False)].copy()

    train = _sub(parts["train"], 150_000)
    val = _sub(parts["val"], 30_000)
    test = _sub(parts["test"], 50_000)

    pipe = FeaturePipeline(use_well_norm=True, use_missing_mask=True, use_relative_depth=True)
    tr = pipe.fit_transform(train)
    va = pipe.transform(val)
    te = pipe.transform(test)

    classes = sorted(pd.unique(tr.y).tolist())
    mapping = {int(c): i for i, c in enumerate(classes)}

    def enc_filter(X, y, *extras):
        mask = np.array([int(v) in mapping for v in y])
        y2 = np.array([mapping[int(v)] for v in y[mask]], dtype=np.int32)
        out = [X[mask], y2]
        for e in extras:
            out.append(e[mask])
        return tuple(out)

    Xtr, ytr, ctr, wtr = enc_filter(tr.X, tr.y, tr.conf, tr.wells)
    Xva, yva = enc_filter(va.X, va.y)
    Xte, yte, wte = enc_filter(te.X, te.y, te.wells)

    # settings: keep conf <= k (1=high only, <=2, all)
    settings = [
        ("conf<=1 (high only)", ctr <= 1.0, False),
        ("conf<=2", ctr <= 2.0, False),
        ("all + inverse-conf weight", np.ones(len(ctr), dtype=bool), True),
        ("all uniform", np.ones(len(ctr), dtype=bool), False),
    ]
    rows = []
    for name, keep, use_conf in settings:
        if keep.sum() < 1000:
            continue
        model = LGBMLithologyModel(
            config=LGBMConfig(n_estimators=150, random_state=42),
            use_confidence=use_conf,
            use_well_balance=False,
        )
        model.fit(Xtr[keep], ytr[keep], ctr[keep], wtr[keep], X_val=Xva, y_val=yva)
        pred = model.predict(Xte)
        ev = evaluate_split(yte, pred, wte)
        row = {"setting": name, "n_train_used": int(keep.sum()), **ev["overall"], **ev["well_summary"]}
        rows.append(row)
        print(row)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "confidence_sensitivity.csv", index=False)
    if not out.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(out))
        ax.bar(x - 0.15, out["macro_f1"], 0.3, label="macro F1", color="#2f6f8f")
        ax.bar(x + 0.15, out["worst_well_f1"], 0.3, label="worst-well F1", color="#c4733b")
        ax.set_xticks(x)
        ax.set_xticklabels(out["setting"], rotation=20, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Confidence usage sensitivity (subsampled)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "confidence_sensitivity.png", dpi=150)
        plt.close(fig)
    print(f"Saved {RESULTS_DIR / 'confidence_sensitivity.csv'}")


if __name__ == "__main__":
    main()
