"""After a clean-commit embargo re-run: sync paper numbers if needed, rebuild PDFs + manifest + gate.

Usage (from repo root, after embargo_fixed_block_study.py finishes):
  py -3 -u scripts/finish_embargo_paper_sync.py
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import PIPELINE_VERSION

FOLLOWUP = ROOT / "results" / "followup"
SUMMARY = FOLLOWUP / "embargo_fixed_block_summary.csv"
STUDY = FOLLOWUP / "embargo_fixed_block_study.csv"


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("RUN", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(cwd or ROOT))
    if r.returncode != 0:
        raise SystemExit(f"failed ({r.returncode}): {' '.join(cmd)}")


def main() -> int:
    if not STUDY.exists() or not SUMMARY.exists():
        raise SystemExit("missing embargo study/summary CSV")
    study = pd.read_csv(STUDY)
    summary = pd.read_csv(SUMMARY)
    if len(study) != 50:
        raise SystemExit(f"embargo study has {len(study)} rows, expected 50")
    if "pipeline_version" not in study.columns or (
        study["pipeline_version"].astype(str) != PIPELINE_VERSION
    ).any():
        raise SystemExit("embargo study pipeline_version mismatch")
    dirty = study["git_dirty"].astype(str).str.lower().isin({"true", "1", "yes"})
    if dirty.any():
        raise SystemExit(
            f"embargo still git_dirty on {int(dirty.sum())}/50 rows — "
            "re-run on a clean source commit before finishing"
        )
    print(summary.to_string(index=False), flush=True)
    print(
        f"[{datetime.now(timezone.utc).isoformat()}] embargo clean; "
        "rebuilding figures/manifest/gate/PDFs",
        flush=True,
    )
    # Paper tex already updated to current metrics; recompile + gate.
    run([sys.executable, "-u", "scripts/build_release_manifest.py"])
    run([sys.executable, "-u", "scripts/check_active_results_r3.py"])
    paper = ROOT / "paper" / "gse"
    run(["latexmk", "-pdf", "-interaction=nonstopmode", "supplementary.tex"], cwd=paper)
    run(["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"], cwd=paper)
    print("DONE finish_embargo_paper_sync", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
