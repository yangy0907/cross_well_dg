"""After paper_r3_full finishes: merge summaries, follow-ups, figures.

Usage (from cross_well_dg/):
  py -u scripts/finish_r3_pipeline.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    py = sys.executable
    # Ensure no competing lock from a dead process is required; finish assumes main run done.
    run([py, "-u", "scripts/revision_r3_openset_bootstrap.py"])
    run([py, "-u", "scripts/while_coldstart_study.py"])
    run([py, "-u", "scripts/embargo_fixed_block_study.py"])
    run([py, "-u", "scripts/make_paper_figures.py"])
    print("Done follow-ups + figures. Update P2/P3 numbers in main.tex from summary CSVs.")


if __name__ == "__main__":
    main()
