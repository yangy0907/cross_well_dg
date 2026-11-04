"""Make a simple graphical abstract for GSE submission."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Embed TrueType (Type 42) fonts for publisher PDF compliance.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "gse" / "figs"
PAPER = ROOT / "results" / "paper_figures"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))

    # Left: leakage vs cross-well
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Evaluation gap on FORCE 2020", fontsize=11, pad=8)
    for i, (y, label, score, color) in enumerate(
        [
            (4.2, "Depth-point split\n(optimistic protocol)", "Macro-F1 ≈ 0.737", "#4c6a80"),
            (
                1.6,
                "Well hold-out\n(cross-well / DG)",
                "Macro-F1 ≈ 0.34 Global\n≈ 0.37–0.38 Hybrid\n(18 closed multi-rep)",
                "#b85c38",
            ),
        ]
    ):
        box = FancyBboxPatch(
            (0.6, y - 0.9),
            5.2,
            1.8,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            facecolor=color,
            edgecolor="none",
            alpha=0.9,
        )
        ax.add_patch(box)
        ax.text(3.2, y + 0.35, label, ha="center", va="center", color="white", fontsize=9)
        ax.text(3.2, y - 0.40, score, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.annotate(
        "",
        xy=(3.2, 2.7),
        xytext=(3.2, 3.2),
        arrowprops=dict(arrowstyle="->", color="#333", lw=1.4),
    )
    ax.text(8.2, 3.0, "Same LightGBM\nfamily", ha="center", va="center", fontsize=9, color="#333")
    ax.add_patch(
        FancyBboxPatch(
            (6.4, 2.2),
            3.4,
            1.6,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            facecolor="#f2f2f2",
            edgecolor="#888",
        )
    )

    # Right: protocols + hybrid
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("This paper: protocols + CPU hybrid baseline", fontsize=11, pad=8)
    protocols = [
        "P1 random wells (primary)",
        "P2 quadrant hold-out",
        "P3 six-well stress",
    ]
    for i, p in enumerate(protocols):
        y = 4.6 - i * 1.15
        ax.add_patch(
            FancyBboxPatch(
                (0.5, y - 0.4),
                4.0,
                0.85,
                boxstyle="round,pad=0.04,rounding_size=0.1",
                facecolor="#2a6f6f",
                edgecolor="none",
                alpha=0.9,
            )
        )
        ax.text(2.5, y, p, ha="center", va="center", color="white", fontsize=9)
    ax.add_patch(
        FancyBboxPatch(
            (5.3, 1.2),
            4.2,
            3.6,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            facecolor="#f7f4ef",
            edgecolor="#2a6f6f",
            lw=1.2,
        )
    )
    ax.text(7.4, 4.3, "Hybrid LightGBM", ha="center", fontsize=10, fontweight="bold", color="#2a6f6f")
    ax.text(
        7.4,
        2.9,
        "global + well-wise z\n+ masks + optional\nrelative depth / Smooth\n\nDG vs UDA-lite\nreported separately",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#333",
    )
    ax.annotate(
        "",
        xy=(5.3, 3.0),
        xytext=(4.6, 3.0),
        arrowprops=dict(arrowstyle="->", color="#2a6f6f", lw=1.5),
    )

    fig.tight_layout()
    for folder in (OUT, PAPER):
        fig.savefig(folder / "graphical_abstract.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved graphical_abstract.png")


if __name__ == "__main__":
    main()
