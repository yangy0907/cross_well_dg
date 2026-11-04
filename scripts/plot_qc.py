"""Quick QC / result plots for the paper."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    class_counts = pd.read_csv(PROCESSED_DIR / "class_counts.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(class_counts["lithology_name"], class_counts["count"], color="#2f6f8f")
    ax.set_ylabel("Count")
    ax.set_title("FORCE 2020 lithofacies class distribution")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)

    summary_path = RESULTS_DIR / "summary_all.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        p1 = df[df["protocol"] == "protocol1_random_wells"].copy()
        if not p1.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(p1["method"], p1["worst_well_f1"], color="#c4733b", label="worst-well F1")
            ax.plot(p1["method"], p1["macro_f1"], "o-", color="#1f3a5f", label="macro F1")
            ax.set_ylabel("Score")
            ax.set_title("Protocol-1 method comparison")
            ax.legend()
            plt.xticks(rotation=30, ha="right")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "protocol1_methods.png", dpi=150)
            plt.close(fig)

    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
