"""Domain drift diagnosis for Protocol-1 train vs test wells."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import CORE_CURVES, FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR, SPLITS_DIR
from src.prepare_data import load_labeled


def _mean_vector(df: pd.DataFrame, curves: list[str]) -> np.ndarray:
    vals = []
    for c in curves:
        s = pd.to_numeric(df[c], errors="coerce")
        vals.append(float(s.mean()) if s.notna().any() else 0.0)
    return np.asarray(vals, dtype=np.float64)


def _mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    """Unbiased MMD^2 estimate on small samples (rows = samples)."""
    n = min(len(x), 2000)
    m = min(len(y), 2000)
    rng = np.random.default_rng(42)
    xs = x[rng.choice(len(x), n, replace=False)]
    ys = y[rng.choice(len(y), m, replace=False)]
    # standardize by pooled std
    pooled = np.vstack([xs, ys])
    std = pooled.std(axis=0) + 1e-6
    xs = xs / std
    ys = ys / std

    def k(a, b):
        # ||a-b||^2 via (a^2 + b^2 - 2ab)
        aa = np.sum(a * a, axis=1)[:, None]
        bb = np.sum(b * b, axis=1)[None, :]
        dist = aa + bb - 2 * (a @ b.T)
        return np.exp(-gamma * np.clip(dist, 0, None))

    kxx = k(xs, xs)
    kyy = k(ys, ys)
    kxy = k(xs, ys)
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    mmd2 = kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.mean()
    return float(max(mmd2, 0.0))


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_labeled()
    split_path = SPLITS_DIR / "protocol1_seed42.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split_path}; run protocol1 first.")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train = df[df["well_id"].isin(split["train_wells"])]
    test = df[df["well_id"].isin(split["test_wells"])]

    curves = [c for c in CORE_CURVES if c in df.columns]
    # drop curves that are almost all missing
    curves = [c for c in curves if train[c].notna().mean() > 0.05 and test[c].notna().mean() > 0.05]

    # 1) mean difference heatmap-like bar
    mu_tr = _mean_vector(train, curves)
    mu_te = _mean_vector(test, curves)
    # z-diff using train std
    std_tr = np.array(
        [
            float(pd.to_numeric(train[c], errors="coerce").std(ddof=0) or 1.0) + 1e-6
            for c in curves
        ]
    )
    zdiff = (mu_te - mu_tr) / std_tr
    drift_df = pd.DataFrame({"curve": curves, "z_diff_test_minus_train": zdiff})
    drift_df.to_csv(RESULTS_DIR / "drift_curve_zdiff_p1.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#c4733b" if abs(v) > 0.3 else "#2f6f8f" for v in zdiff]
    ax.bar(curves, zdiff, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Mean z-diff (test − train)")
    ax.set_title("Protocol-1 feature drift (curve means)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "drift_curve_zdiff_p1.png", dpi=150)
    plt.close(fig)

    # 2) class prior shift
    def prior(d: pd.DataFrame) -> pd.Series:
        return d["lithology_name"].value_counts(normalize=True)

    p_tr, p_te = prior(train), prior(test)
    names = sorted(set(p_tr.index) | set(p_te.index))
    prior_df = pd.DataFrame(
        {
            "lithology": names,
            "train_frac": [float(p_tr.get(n, 0.0)) for n in names],
            "test_frac": [float(p_te.get(n, 0.0)) for n in names],
        }
    )
    prior_df.to_csv(RESULTS_DIR / "drift_class_prior_p1.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, prior_df["train_frac"], width=w, label="train", color="#2f6f8f")
    ax.bar(x + w / 2, prior_df["test_frac"], width=w, label="test", color="#c4733b")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Fraction")
    ax.set_title("Protocol-1 lithofacies prior shift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "drift_class_prior_p1.png", dpi=150)
    plt.close(fig)

    # 3) rough MMD on core numeric matrix (imputed medians)
    use = [c for c in ["GR", "RHOB", "NPHI", "DTC", "RDEP", "RMED", "PEF", "CALI"] if c in curves]
    Xtr = train[use].apply(pd.to_numeric, errors="coerce")
    Xte = test[use].apply(pd.to_numeric, errors="coerce")
    med = Xtr.median()
    Xtr = Xtr.fillna(med).to_numpy(dtype=np.float64)
    Xte = Xte.fillna(med).to_numpy(dtype=np.float64)
    mmd2 = _mmd_rbf(Xtr, Xte, gamma=0.5)
    meta = {
        "protocol": "protocol1_seed42",
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_train_wells": int(train["well_id"].nunique()),
        "n_test_wells": int(test["well_id"].nunique()),
        "mmd2_core_curves": mmd2,
        "curves_used_for_mmd": use,
    }
    (RESULTS_DIR / "drift_meta_p1.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))
    print(f"Figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
