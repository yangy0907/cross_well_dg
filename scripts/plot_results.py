"""Build paper-ready result figures and ablation table from summary CSVs."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import FIGURES_DIR, RESULTS_DIR

METHOD_ORDER = [
    "naive_depth_split_lgbm",
    "lgbm_global",
    "lgbm_wellnorm",
    "lgbm_missmask",
    "lgbm_conf",
    "lgbm_full",
    "lgbm_groupdro",
    "mlp_coral_uda",
]

ABLATION_MAP = {
    "lgbm_global": "no well-norm (weak)",
    "lgbm_wellnorm": "+ well-norm + rel-depth (A)",
    "lgbm_missmask": "+ missing masks (A+B)",
    "lgbm_conf": "+ confidence weights (A+B+C)",
    "lgbm_full": "+ well balance (full)",
    "lgbm_groupdro": "+ GroupDRO (worst-well)",
    "mlp_coral_uda": "MLP + CORAL (UDA)",
}


def _load_all() -> pd.DataFrame:
    files = sorted(RESULTS_DIR.glob("summary_*.csv"))
    files = [f for f in files if f.name != "summary_all.csv" and "archive" not in f.name]
    if not files:
        raise FileNotFoundError("No summary_*.csv found in results/")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_all()
    df.to_csv(RESULTS_DIR / "summary_all.csv", index=False)

    # Protocol-1 method comparison
    p1 = df[df["protocol"] == "protocol1_random_wells"].copy()
    if not p1.empty:
        # keep latest seed42 rows unique by method
        p1 = p1.sort_values("seed").drop_duplicates("method", keep="last")
        order = [m for m in METHOD_ORDER if m in set(p1["method"])]
        p1 = p1.set_index("method").loc[order].reset_index()

        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = range(len(p1))
        ax.bar(x, p1["worst_well_f1"], color="#c4733b", alpha=0.85, label="worst-well F1")
        ax.plot(x, p1["macro_f1"], "o-", color="#1f3a5f", label="macro F1")
        if "well_macro_f1_mean" in p1.columns:
            ax.plot(x, p1["well_macro_f1_mean"], "s--", color="#2f6f8f", label="mean well F1")
        ax.set_xticks(list(x))
        ax.set_xticklabels(p1["method"], rotation=30, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Protocol-1 cross-well results")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "protocol1_methods.png", dpi=150)
        plt.close(fig)

        abl = p1.copy()
        abl["ablation"] = abl["method"].map(ABLATION_MAP).fillna(abl["method"])
        cols = [
            c
            for c in [
                "method",
                "ablation",
                "macro_f1",
                "well_macro_f1_mean",
                "worst_well_f1",
                "worst10pct_well_f1",
                "accuracy",
                "seconds",
            ]
            if c in abl.columns
        ]
        abl[cols].to_csv(RESULTS_DIR / "ablation_protocol1.csv", index=False)

    # Leakage contrast
    naive = df[df["protocol"] == "naive_depth_split"]
    if not naive.empty and not p1.empty:
        row_n = naive.iloc[-1]
        row_p = p1.sort_values("macro_f1", ascending=False).iloc[0]
        contrast = pd.DataFrame(
            [
                {
                    "setting": "naive_depth_split",
                    "macro_f1": row_n.get("macro_f1"),
                    "worst_well_f1": row_n.get("worst_well_f1"),
                },
                {
                    "setting": f"cross_well_best({row_p['method']})",
                    "macro_f1": row_p.get("macro_f1"),
                    "worst_well_f1": row_p.get("worst_well_f1"),
                },
            ]
        )
        contrast.to_csv(RESULTS_DIR / "leakage_vs_crosswell.csv", index=False)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(
            contrast["setting"],
            contrast["macro_f1"],
            color=["#888888", "#2f6f8f"],
        )
        ax.set_ylabel("macro F1")
        ax.set_title("Optimistic leakage vs cross-well evaluation")
        plt.xticks(rotation=15, ha="right")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "leakage_contrast.png", dpi=150)
        plt.close(fig)

    # Protocol-2
    p2 = df[df["protocol"] == "protocol2_block_holdout"].copy()
    if not p2.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for blk, g in p2.groupby("holdout_blocks"):
            g = g.drop_duplicates("method", keep="last")
            ax.plot(g["method"], g["macro_f1"], "o-", label=f"holdout {blk}")
        ax.set_ylabel("macro F1")
        ax.set_title("Protocol-2 quadrant holdout")
        plt.xticks(rotation=30, ha="right")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "protocol2_methods.png", dpi=150)
        plt.close(fig)

    # Protocol-3 well-wise
    p3 = df[df["protocol"] == "protocol3_lowo"].copy()
    if not p3.empty and "target_well" in p3.columns:
        pivot = p3.pivot_table(
            index="target_well", columns="method", values="macro_f1", aggfunc="last"
        )
        pivot.to_csv(RESULTS_DIR / "protocol3_lowo_macro_f1.csv")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("macro F1")
        ax.set_title("Protocol-3 leave-one-well-out")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "protocol3_lowo.png", dpi=150)
        plt.close(fig)

    # class distribution (always)
    class_path = ROOT / "data" / "processed" / "class_counts.csv"
    if class_path.exists():
        cc = pd.read_csv(class_path)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(cc["lithology_name"], cc["count"], color="#2f6f8f")
        ax.set_ylabel("Count")
        ax.set_title("FORCE 2020 lithofacies class distribution")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=150)
        plt.close(fig)

    print(f"Wrote figures to {FIGURES_DIR}")
    print(f"summary_all rows: {len(df)}")
    print(df.groupby("protocol")["method"].count() if "protocol" in df.columns else "")


if __name__ == "__main__":
    main()
