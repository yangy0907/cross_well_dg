"""Rebuild Protocol-1 seed42 per-class F1 CSV from r3 details JSON (no retrain).

Also filters sklearn aggregate keys (micro/macro/weighted avg) that can leak into
per_class on some seeds. Prefer ``scripts/sync_r3_ccommon_and_wellf1.py`` for
closed multi-seed C_common Macro + per-class + well-F1 figure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import LITHOLOGY_MAP, RESULTS_DIR

OUT = RESULTS_DIR / "followup"
DETAILS = RESULTS_DIR / "details_protocol1_random_wells_seed42.json"
METHODS = {
    "lgbm_global": "Global",
    "lgbm_hybrid": "Hybrid",
}
_SKIP = {"accuracy", "macro avg", "weighted avg", "micro avg", "samples avg"}


def main() -> None:
    rows_all = json.loads(DETAILS.read_text(encoding="utf-8"))
    out_rows = []
    for rec in rows_all:
        name = METHODS.get(rec["method"])
        if name is None:
            continue
        if not str(rec.get("pipeline_version", "")).startswith("r3_"):
            raise SystemExit(
                f"{rec['method']} pipeline_version={rec.get('pipeline_version')} is not r3"
            )
        macro = float(rec["overall"]["macro_f1"])
        for cls_name, stats in rec["per_class"].items():
            if cls_name in _SKIP:
                continue
            if not isinstance(stats, dict) or "f1" not in stats:
                continue
            code = next(
                (c for c, n in LITHOLOGY_MAP.items() if n == cls_name),
                -1,
            )
            out_rows.append(
                {
                    "method": name,
                    "code": code,
                    "class": cls_name,
                    "support": int(stats["support"]),
                    "f1": float(stats["f1"]),
                    "macro_f1": macro,
                }
            )
    tab = pd.DataFrame(out_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "per_class_f1_protocol1_seed42.csv"
    tab.to_csv(path, index=False)
    print("wrote", path)
    print(tab.groupby("method")["macro_f1"].first().to_string())


if __name__ == "__main__":
    main()
