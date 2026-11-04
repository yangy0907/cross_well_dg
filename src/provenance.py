"""Run provenance: git / config / split / dataset hashes for result rows."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .constants import PROCESSED_DIR, PROJECT_ROOT

# Frozen once per experiment process so mid-run writes to tracked results/
# (and the run lock file) cannot flip git_dirty mid-schedule.
_FROZEN_GIT: dict[str, Any] | None = None

# Paths that must not mark the *source* tree dirty for release provenance.
# Results and locks are outputs of the run being fingerprinting.
_GIT_DIRTY_EXCLUDES: tuple[str, ...] = (
    ":(exclude)results/",
    ":(exclude)data/processed/",
    ":(exclude)data/raw/",
    ":(exclude)*.log",
    ":(exclude)paper/**/*.aux",
    ":(exclude)paper/**/*.log",
    ":(exclude)paper/**/*.out",
    ":(exclude)paper/**/*.fls",
    ":(exclude)paper/**/*.fdb_latexmk",
    ":(exclude)paper/**/*.synctex.gz",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_provenance(
    repo: Path | None = None,
    *,
    ignore_outputs: bool = True,
    use_frozen: bool = True,
) -> dict[str, Any]:
    """Return git commit / dirty / describe.

    When ``use_frozen`` and a process snapshot exists, reuse it.
    When ``ignore_outputs``, dirty checks exclude results/ and other run outputs
    so writing summary CSVs cannot poison later methods' provenance.
    """
    global _FROZEN_GIT
    if use_frozen and _FROZEN_GIT is not None:
        return dict(_FROZEN_GIT)

    root = repo or PROJECT_ROOT
    out: dict[str, Any] = {
        "git_commit": "",
        "git_dirty": True,
        "git_describe": "",
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status_cmd = ["git", "status", "--porcelain"]
        if ignore_outputs:
            status_cmd.extend(_GIT_DIRTY_EXCLUDES)
        dirty = (
            subprocess.check_output(
                status_cmd,
                cwd=root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            != ""
        )
        try:
            describe = subprocess.check_output(
                ["git", "describe", "--always", "--dirty"],
                cwd=root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            # describe --dirty still sees results/; prefer our ignore_outputs flag.
            if ignore_outputs and dirty is False and describe.endswith("-dirty"):
                describe = describe[: -len("-dirty")]
            elif ignore_outputs and dirty and not describe.endswith("-dirty"):
                describe = describe + "-dirty"
        except subprocess.CalledProcessError:
            describe = commit[:12] + ("-dirty" if dirty else "")
        out["git_commit"] = commit
        out["git_dirty"] = bool(dirty)
        out["git_describe"] = describe
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    return out


def freeze_git_provenance(
    repo: Path | None = None, *, ignore_outputs: bool = True
) -> dict[str, Any]:
    """Capture source-tree git provenance once (call before any result writes)."""
    global _FROZEN_GIT
    _FROZEN_GIT = git_provenance(
        repo, ignore_outputs=ignore_outputs, use_frozen=False
    )
    return dict(_FROZEN_GIT)


def clear_frozen_git_provenance() -> None:
    global _FROZEN_GIT
    _FROZEN_GIT = None


def frozen_git_provenance() -> dict[str, Any] | None:
    return dict(_FROZEN_GIT) if _FROZEN_GIT is not None else None


def hash_obj(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False).encode(
        "utf-8"
    )
    return _sha256_bytes(payload)


def dataset_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("curves_all.parquet", "labeled.parquet", "dataset_meta.json"):
        p = PROCESSED_DIR / name
        if p.exists():
            out[f"hash_{name.replace('.', '_')}"] = _sha256_file(p)
        else:
            out[f"hash_{name.replace('.', '_')}"] = ""
    return out


def feature_param_hash(
    feature_names: list[str],
    *,
    feature_mode: str,
    use_relative_depth: bool,
    well_z_mode: str,
    model_params: dict[str, Any] | None = None,
) -> str:
    return hash_obj(
        {
            "feature_names": feature_names,
            "feature_mode": feature_mode,
            "use_relative_depth": use_relative_depth,
            "well_z_mode": well_z_mode,
            "model_params": model_params or {},
        }
    )


def collect_run_provenance(
    *,
    split: dict[str, Any] | None = None,
    config_path: Path | None = None,
    feature_names: list[str] | None = None,
    feature_mode: str = "",
    use_relative_depth: bool = False,
    well_z_mode: str = "",
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prov = git_provenance(use_frozen=True, ignore_outputs=True)
    prov.update(dataset_hashes())
    if split is not None:
        # Hash identity fields only (stable across path noise).
        split_key = {
            k: split.get(k)
            for k in (
                "protocol",
                "seed",
                "train_wells",
                "val_wells",
                "test_wells",
                "holdout_blocks",
                "target_well",
                "selection",
                # naive / block depth splits
                "test_frac",
                "val_frac",
                "row_index_hash",
                "match_n_train",
                "subsample_seed",
                "wells_hash",
                "selection_params_hash",
            )
            if k in split
        }
        prov["split_hash"] = hash_obj(split_key)
    else:
        prov["split_hash"] = ""
    if config_path is not None and Path(config_path).exists():
        prov["config_hash"] = _sha256_file(Path(config_path))
        prov["config_path"] = str(config_path)
    else:
        prov["config_hash"] = ""
        prov["config_path"] = ""
    if feature_names is not None:
        prov["feature_param_hash"] = feature_param_hash(
            feature_names,
            feature_mode=feature_mode,
            use_relative_depth=use_relative_depth,
            well_z_mode=well_z_mode,
            model_params=model_params,
        )
    else:
        prov["feature_param_hash"] = ""
    return prov
