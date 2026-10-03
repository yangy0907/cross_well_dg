"""After embargo finishes (or if already current), run remaining r3 follow-ups + gate/PDF.

Safe to relaunch: skips embargo if CSV already on current PIPELINE_VERSION.
Detached-friendly: logs to results/followup/resume_after_pause_*.log
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import PIPELINE_VERSION  # noqa: E402

FOLLOWUP = ROOT / "results" / "followup"
LOG = FOLLOWUP / f"resume_after_pause_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}Z] {msg}"
    print(line, flush=True)
    FOLLOWUP.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    log("RUN " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(cwd or ROOT))
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(cmd)}")


def pipeline_of(csv_name: str) -> str | None:
    p = FOLLOWUP / csv_name
    if not p.exists():
        return None
    try:
        import pandas as pd

        df = pd.read_csv(p, usecols=["pipeline_version"])
        vers = sorted(df["pipeline_version"].astype(str).unique().tolist())
        return vers[0] if len(vers) == 1 else ",".join(vers)
    except Exception as exc:  # noqa: BLE001
        log(f"warn read {csv_name}: {exc}")
        return None


EXPECTED_EMBARGO_ROWS = 50  # 5 embargo distances × 10 subsample seeds


def embargo_complete() -> bool:
    """True when study has full grid on current pipeline and summary exists."""
    study = FOLLOWUP / "embargo_fixed_block_study.csv"
    summary = FOLLOWUP / "embargo_fixed_block_summary.csv"
    if not study.exists() or not summary.exists():
        return False
    try:
        import pandas as pd

        df = pd.read_csv(study)
        if df.empty or "pipeline_version" not in df.columns:
            return False
        if sorted(df["pipeline_version"].astype(str).unique().tolist()) != [PIPELINE_VERSION]:
            return False
        if len(df) < EXPECTED_EMBARGO_ROWS:
            return False
        # summary may lack pipeline_version on older files; prefer column if present
        sm = pd.read_csv(summary)
        if "pipeline_version" in sm.columns:
            if sorted(sm["pipeline_version"].astype(str).unique().tolist()) != [PIPELINE_VERSION]:
                return False
        return True
    except Exception as exc:  # noqa: BLE001
        log(f"warn embargo_complete: {exc}")
        return False


def _embargo_proc_count() -> int:
    """Count live embargo study workers (exclude the powershell probe itself)."""
    alive = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "Get-CimInstance Win32_Process | "
                "Where-Object { "
                "$_.Name -match '^(python|py)(\\.exe)?$' -and "
                "$_.CommandLine -match 'scripts[/\\\\]embargo_fixed_block_study\\.py' "
                "} | Measure-Object | Select-Object -ExpandProperty Count"
            ),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    try:
        return int((alive.stdout or "").strip() or "0")
    except ValueError:
        return 0


def wait_embargo_current(timeout_s: int = 6 * 3600) -> None:
    """Wait until embargo study+summary are complete on current pipeline."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if embargo_complete():
            log("embargo complete on current pipeline")
            return
        count = _embargo_proc_count()
        n = 0
        study = FOLLOWUP / "embargo_fixed_block_study.csv"
        if study.exists():
            try:
                import pandas as pd

                n = len(pd.read_csv(study))
            except Exception:  # noqa: BLE001
                n = -1
        log(f"waiting for embargo… rows={n}/{EXPECTED_EMBARGO_ROWS} procs={count}")
        if count == 0 and not embargo_complete():
            log("embargo process gone but incomplete; restarting")
            run([sys.executable, "-u", "scripts/embargo_fixed_block_study.py"])
            if embargo_complete():
                return
            continue
        time.sleep(60)
    raise SystemExit("timeout waiting for embargo on current pipeline")


def main() -> int:
    log(f"resume pipeline={PIPELINE_VERSION}")
    if embargo_complete():
        log("skip embargo (already complete)")
    else:
        count = _embargo_proc_count()
        if count == 0:
            log("starting embargo_fixed_block_study.py")
            run([sys.executable, "-u", "scripts/embargo_fixed_block_study.py"])
        else:
            log(f"embargo already running (count={count}); waiting")
            wait_embargo_current()
        if not embargo_complete():
            # Final attempt (e.g. hung peer killed externally)
            log("embargo still incomplete; running locally")
            run([sys.executable, "-u", "scripts/embargo_fixed_block_study.py"])

    for script in (
        "scripts/while_coldstart_study.py",
        "scripts/parent_well_sensitivity.py",
        "scripts/while_sparse_prefix_study.py",
        "scripts/revision_r3_openset_bootstrap.py",
        "scripts/build_acceptance_record.py",
        "scripts/sync_r3_paper_tables.py",
        "scripts/make_paper_figures.py",
        "scripts/build_release_manifest.py",
        "scripts/check_active_results_r3.py",
    ):
        run([sys.executable, "-u", script])

    paper = ROOT / "paper" / "gse"
    run(["latexmk", "-pdf", "-interaction=nonstopmode", "supplementary.tex"], cwd=paper)
    run(["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"], cwd=paper)
    log("DONE resume_after_pause")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
