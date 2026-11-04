"""Build RELEASE_MANIFEST.json: code, configs, splits, CSV/JSON, parquet hashes."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import PROCESSED_DIR, RESULTS_DIR, SPLITS_DIR
from src.provenance import _GIT_DIRTY_EXCLUDES
from src.train import PIPELINE_VERSION


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_meta() -> dict:
    """Distinguish source-tree dirty vs full artifact-tree dirty.

    - ``source_dirty`` / ``source_git_dirty_at_run_start`` semantics:
      matches CSV ``git_dirty`` (excludes results/, processed data, latex aux).
    - ``artifact_dirty`` / ``artifact_tree_dirty_at_manifest_creation``:
      full ``git status --porcelain`` including generated results.
    - ``dirty`` is kept as an alias of ``artifact_dirty`` for backward compatibility.
    """

    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=ROOT, text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    artifact_dirty = bool(run(["git", "status", "--porcelain"]))
    source_cmd = ["git", "status", "--porcelain", *_GIT_DIRTY_EXCLUDES]
    source_dirty = bool(run(source_cmd))
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "commit_short": run(["git", "rev-parse", "--short=12", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "source_dirty": source_dirty,
        "artifact_dirty": artifact_dirty,
        "dirty": artifact_dirty,  # legacy alias = artifact tree
        "describe": run(["git", "describe", "--always", "--dirty", "--tags"]),
        "semantics": {
            "source_dirty": (
                "source_git_dirty_at_run_start — same ignore set as CSV git_dirty "
                "(excludes results/, data/processed|raw, logs, latex aux)"
            ),
            "artifact_dirty": (
                "artifact_tree_dirty_at_manifest_creation — full working tree "
                "including generated results CSVs/PDFs"
            ),
            "dirty": "legacy alias of artifact_dirty",
        },
    }


def collect(pattern_root: Path, glob: str) -> list[dict]:
    rows = []
    for p in sorted(pattern_root.glob(glob)):
        if not p.is_file():
            continue
        rows.append(
            {
                "path": str(p.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            }
        )
    return rows


def main() -> None:
    # Side-car hashes for parquet (even when parquet itself is gitignored).
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("curves_all.parquet", "labeled.parquet"):
        pq = PROCESSED_DIR / name
        if pq.exists():
            (PROCESSED_DIR / f"{name}.sha256").write_text(
                f"{sha256_file(pq)}  {name}\n", encoding="utf-8"
            )

    git = git_meta()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "git": git,
        "configs": collect(ROOT / "configs", "*.yaml"),
        "splits": collect(SPLITS_DIR, "protocol*.json"),
        "summary_csv": collect(RESULTS_DIR, "summary_*.csv"),
        "details_json": collect(RESULTS_DIR, "details_*.json"),
        "followup": collect(RESULTS_DIR / "followup", "*.csv")
        + collect(RESULTS_DIR / "followup", "*.json"),
        "parquet_sha256_files": collect(PROCESSED_DIR, "*.sha256"),
        "notes": [
            "Regenerate parquet with: py run_experiment.py prepare",
            "CSV git_dirty == source_dirty (ignore results/); manifest dirty == artifact_dirty.",
            "Prefer source_dirty=false and artifact_dirty=false before tagging a release.",
            "Re-run build_release_manifest.py after the clean commit that freezes the tag.",
            "Primary Protocol-1 multi-rep means use the locked 20-seed list "
            "(18 closed / 2 open); historical 42/43/45 is SI contrast only.",
        ],
    }
    out = ROOT / "RELEASE_MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"wrote {out} "
        f"(source_dirty={git['source_dirty']}, artifact_dirty={git['artifact_dirty']})"
    )
    if git["artifact_dirty"]:
        print(
            "WARNING: artifact tree is dirty; re-run after commit for a tag-ready manifest."
        )
    if git["source_dirty"]:
        print(
            "WARNING: source tree is dirty; CSV git_dirty may also be True on new runs."
        )


if __name__ == "__main__":
    main()
