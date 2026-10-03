"""Patch CSV n_estimators_budget to match actual train.py schedules (no retrain)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import RESULTS_DIR

BUDGET = {
    "rf_global": 200,
    "rf_global_full": 100,
    "xgb_global": 180,
    "lgbm_groupdro": 120,
}


def patch_file(path: Path) -> bool:
    df = pd.read_csv(path)
    if "method" not in df.columns or "n_estimators_budget" not in df.columns:
        return False
    before = df["n_estimators_budget"].astype(str).tolist()
    df["n_estimators_budget"] = [
        BUDGET.get(str(m), 180) for m in df["method"].astype(str)
    ]
    after = df["n_estimators_budget"].astype(str).tolist()
    if before == after:
        return False
    df.to_csv(path, index=False)
    return True


def main() -> None:
    n = 0
    for path in sorted(RESULTS_DIR.glob("summary_*.csv")):
        if patch_file(path):
            print(f"patched {path.name}")
            n += 1
    sa = RESULTS_DIR / "summary_all.csv"
    if sa.exists() and patch_file(sa):
        print(f"patched {sa.name}")
        n += 1
    print(f"done; {n} files updated")


if __name__ == "__main__":
    main()
