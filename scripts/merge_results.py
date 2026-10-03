"""Merge all summary_*.csv into summary_all.csv."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    files = sorted(RESULTS.glob("summary_*.csv"))
    files = [f for f in files if f.name != "summary_all.csv"]
    if not files:
        print("No summary files found.")
        return
    merged = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    merged.to_csv(RESULTS / "summary_all.csv", index=False)
    print(f"Merged {len(files)} files -> {RESULTS / 'summary_all.csv'} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
