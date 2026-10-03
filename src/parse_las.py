"""Parse FORCE 2020 LAS files into a unified long table."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import LAS_DIR, LITHOLOGY_MAP, NULL_VALUE


def _extract_quadrant(well_id: str, file_stem: str) -> str:
    """NPD quadrant from well name like '15/9-13' or filename '15_9-13'.

    Norwegian Official well names are QUADRANT/BLOCK-WELL (e.g. 15/9-13:
    quadrant 15, block 9). We store only the leading quadrant token. The
    dataframe column is still named ``block`` for backward compatibility with
    existing parquet/CSV splits; treat it as a quadrant id, not an NPD block.
    """
    m = re.match(r"(\d+)\s*/", well_id)
    if m:
        return m.group(1)
    m = re.match(r"(\d+)_", file_stem)
    if m:
        return m.group(1)
    return "unknown"


# Backward-compatible alias
_extract_block = _extract_quadrant


def _parse_curve_names(header_lines: list[str]) -> list[str]:
    names: list[str] = []
    for line in header_lines:
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("~"):
            continue
        # LAS curve: NAME .unit : description
        token = s.split(".", 1)[0].strip()
        if token:
            names.append(token)
    return names


def parse_las_file(path: Path) -> pd.DataFrame:
    """Parse one LAS 2.0 file into a DataFrame."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    well_id = path.stem
    for line in lines:
        if line.strip().upper().startswith("WELL."):
            # WELL.  15/9-13 Sleipner ... : WELL
            body = line.split(":", 1)[0]
            well_id = body.split(".", 1)[1].strip()
            # keep short UWI-like id when possible
            break
    for line in lines:
        if line.strip().upper().startswith("UWI."):
            body = line.split(":", 1)[0]
            uwi = body.split(".", 1)[1].strip()
            if uwi:
                well_id = uwi
            break

    # Locate sections
    curve_start = ascii_start = None
    for i, line in enumerate(lines):
        u = line.strip().upper()
        if u.startswith("~C"):
            curve_start = i + 1
        if u.startswith("~A"):
            ascii_start = i + 1
            break
    if curve_start is None or ascii_start is None:
        raise ValueError(f"Missing ~Curve/~Ascii in {path.name}")

    curve_lines = lines[curve_start : ascii_start - 1]
    # trim trailing parameter junk before ~A (already handled)
    curve_names = _parse_curve_names(curve_lines)
    if not curve_names:
        raise ValueError(f"No curves in {path.name}")

    rows: list[list[float]] = []
    for line in lines[ascii_start:]:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            continue
        # pad / truncate to curve count
        if len(vals) < len(curve_names):
            vals = vals + [NULL_VALUE] * (len(curve_names) - len(vals))
        elif len(vals) > len(curve_names):
            vals = vals[: len(curve_names)]
        rows.append(vals)

    if not rows:
        raise ValueError(f"No ascii rows in {path.name}")

    df = pd.DataFrame(rows, columns=curve_names)
    df = df.replace(NULL_VALUE, np.nan)

    # Normalize column names
    rename = {}
    lower_map = {c.lower(): c for c in df.columns}
    if "dept" in lower_map:
        rename[lower_map["dept"]] = "depth"
    if "depth_md" in lower_map:
        rename[lower_map["depth_md"]] = "depth_md"
    for key, std in [
        ("force_2020_lithofacies_lithology", "lithology"),
        ("force_2020_lithofacies_confidence", "confidence"),
        ("x_loc", "x_loc"),
        ("y_loc", "y_loc"),
        ("z_loc", "z_loc"),
    ]:
        if key in lower_map:
            rename[lower_map[key]] = std
    df = df.rename(columns=rename)

    if "depth" not in df.columns:
        df["depth"] = np.arange(len(df), dtype=float)

    df["well_id"] = well_id
    df["file_name"] = path.name
    df["block"] = _extract_block(well_id, path.stem)

    if "lithology" in df.columns:
        df["lithology_name"] = df["lithology"].map(LITHOLOGY_MAP)
    else:
        df["lithology"] = np.nan
        df["lithology_name"] = np.nan

    if "confidence" not in df.columns:
        df["confidence"] = np.nan

    return df


def iter_las_files(las_dir: Path | None = None) -> Iterable[Path]:
    las_dir = Path(las_dir or LAS_DIR)
    return sorted(las_dir.glob("*.las"))


def parse_all_wells(las_dir: Path | None = None, max_wells: int | None = None) -> pd.DataFrame:
    files = list(iter_las_files(las_dir))
    if max_wells is not None:
        files = files[:max_wells]
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for path in files:
        try:
            frames.append(parse_las_file(path))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path.name}: {exc}")
    if not frames:
        raise RuntimeError("No LAS files parsed.\n" + "\n".join(errors))
    df = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    df.attrs["parse_errors"] = errors
    return df
