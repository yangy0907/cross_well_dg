"""Pause the active paper_r3_release run after 14:00, resume after 16:20 (local time).

Usage:
  py -3 -u scripts/schedule_pause_resume_r3.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LOCK = RESULTS / ".run_experiment.lock"
LOG = RESULTS / "run_r3_release_20260930.log"
PAUSE_H, PAUSE_M = 14, 0
RESUME_H, RESUME_M = 16, 20


def _log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _next_today(h: int, m: int) -> datetime:
    now = datetime.now()
    target = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def _sleep_until(target: datetime) -> None:
    while True:
        now = datetime.now()
        rem = (target - now).total_seconds()
        if rem <= 0:
            return
        time.sleep(min(rem, 30.0))


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes

            SYNCHRONIZE = 0x00100000
            handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_lock_pid() -> int:
    if not LOCK.exists():
        return 0
    try:
        raw = json.loads(LOCK.read_text(encoding="utf-8"))
        return int(raw.get("pid", 0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0


def stop_run() -> None:
    pid = _read_lock_pid()
    _log(f"pause: lock pid={pid}")
    if pid and _pid_alive(pid):
        if os.name == "nt":
            # Kill the python process holding the lock (and its console tree).
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                capture_output=True,
                text=True,
            )
        else:
            try:
                os.kill(pid, 15)
            except OSError:
                pass
        time.sleep(2)
    # Clear stale lock so resume can acquire it.
    try:
        if LOCK.exists():
            LOCK.unlink()
            _log("pause: removed stale lock")
    except OSError as exc:
        _log(f"pause: lock unlink failed: {exc}")
    _log("pause: stopped (completed methods remain in summary_*.csv for resume)")


def resume_run() -> None:
    _log("resume: starting paper_r3_release.yaml")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Append to the same log; resume skips finished provenance-matched methods.
    with LOG.open("a", encoding="utf-8") as logf:
        logf.write("\n===== RESUME after scheduled pause =====\n")
        logf.flush()
        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(ROOT / "run_experiment.py"),
                "run",
                "--config",
                "configs/paper_r3_release.yaml",
            ],
            cwd=str(ROOT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    _log(f"resume: spawned pid={proc.pid} (detached to log)")
    # Do not wait — scheduler exits; experiment continues independently.


def main() -> int:
    pause_at = _next_today(PAUSE_H, PAUSE_M)
    resume_at = _next_today(RESUME_H, RESUME_M)
    if resume_at <= pause_at:
        resume_at += timedelta(days=1)
    _log(f"scheduler armed: pause>={pause_at.isoformat(timespec='seconds')} "
         f"resume>={resume_at.isoformat(timespec='seconds')}")
    _sleep_until(pause_at)
    stop_run()
    _log(f"waiting until resume {resume_at.isoformat(timespec='seconds')}")
    _sleep_until(resume_at)
    resume_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
