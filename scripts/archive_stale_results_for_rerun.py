"""Move active summary/details CSVs/JSON into a timestamped archive before a clean re-run.

Does NOT delete archives. After this, commit the clean tree, then run ONLY:
  py -3 run_experiment.py run --config configs/paper_r3_release.yaml
Then diagnostics / figures / check_active_results_r3.py (see RELEASE_CHECKLIST.md).
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import RESULTS_DIR
from src.train import PIPELINE_VERSION


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = RESULTS_DIR / "followup" / f"archive_pre_{PIPELINE_VERSION}_{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    moved = []
    patterns = [
        "summary_*.csv",
        "details_*.json",
        "summary_all.csv",
        "summary_naive_depth_split.csv",
    ]
    for pat in patterns:
        for p in RESULTS_DIR.glob(pat):
            if not p.is_file():
                continue
            target = dest / p.name
            shutil.move(str(p), str(target))
            moved.append(p.name)
    readme = dest / "README.md"
    readme.write_text(
        f"# Archived before clean re-run\n\n"
        f"- UTC: {stamp}\n"
        f"- Target pipeline_version: `{PIPELINE_VERSION}`\n"
        f"- Files moved: {len(moved)}\n"
        f"- Reason: enforce single config/split/data provenance; resume now requires matching hashes.\n",
        encoding="utf-8",
    )
    print(f"archived {len(moved)} files -> {dest}")
    for name in moved[:20]:
        print(f"  {name}")
    if len(moved) > 20:
        print(f"  ... +{len(moved) - 20} more")


if __name__ == "__main__":
    main()
