"""Build processed dataset and QC summary."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .constants import (
    CONF_COL,
    CORE_CURVES,
    DATA_DIR,
    DEPTH_COL,
    FIGURES_DIR,
    LABEL_COL,
    LITHOLOGY_MAP,
    PROCESSED_DIR,
    RAW_DIR,
    WELL_COL,
)
from .parse_las import parse_all_wells

CURVES_ALL_NAME = "curves_all.parquet"
LABELED_NAME = "labeled.parquet"


def _ensure_dirs() -> None:
    for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _standardize_curve_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in CORE_CURVES:
        if c not in out.columns:
            out[c] = pd.NA
    ordered = (
        [WELL_COL, "file_name", "block", DEPTH_COL, "depth_md", "x_loc", "y_loc", "z_loc"]
        + CORE_CURVES
        + [LABEL_COL, "lithology_name", CONF_COL]
    )
    for col in ordered:
        if col not in out.columns:
            if col == "lithology_name":
                out[col] = pd.NA
            elif col == CONF_COL:
                out[col] = pd.NA
            elif col == LABEL_COL:
                out[col] = pd.NA
            else:
                out[col] = pd.NA
    extras = [c for c in out.columns if c not in ordered]
    return out[ordered + extras]


def prepare_dataset(max_wells: int | None = None, save: bool = True) -> pd.DataFrame:
    """Parse LAS, save full curves + labeled view, return labeled rows for supervised runs."""
    _ensure_dirs()
    raw = parse_all_wells(max_wells=max_wells)

    curves = _standardize_curve_columns(raw)
    # Preserve lithology_name mapping where codes are known; leave unlabeled as NA.
    if LABEL_COL in curves.columns:
        known = curves[LABEL_COL].notna()
        curves.loc[known, LABEL_COL] = curves.loc[known, LABEL_COL].astype(int)
        in_map = known & curves[LABEL_COL].isin(LITHOLOGY_MAP)
        curves.loc[in_map, "lithology_name"] = curves.loc[in_map, LABEL_COL].map(LITHOLOGY_MAP)
        # Drop invalid codes (present but not in FORCE map) from labeled pool later.
        bad = known & ~curves[LABEL_COL].isin(LITHOLOGY_MAP)
        curves.loc[bad, LABEL_COL] = pd.NA
        curves.loc[bad, "lithology_name"] = pd.NA

    labeled = curves.dropna(subset=[LABEL_COL]).copy()
    labeled[LABEL_COL] = labeled[LABEL_COL].astype(int)
    labeled["lithology_name"] = labeled[LABEL_COL].map(LITHOLOGY_MAP)

    if labeled[CONF_COL].notna().any():
        med = float(labeled[CONF_COL].median())
        labeled[CONF_COL] = labeled[CONF_COL].fillna(med)
        curves.loc[curves[CONF_COL].isna(), CONF_COL] = med
    else:
        labeled[CONF_COL] = 1.0
        curves[CONF_COL] = curves[CONF_COL].fillna(1.0)

    if save:
        curves_path = PROCESSED_DIR / CURVES_ALL_NAME
        out_path = PROCESSED_DIR / LABELED_NAME
        curves.to_parquet(curves_path, index=False)
        labeled.to_parquet(out_path, index=False)

        # Per-well label coverage on full curves (independent of supervised filter).
        cov_rows = []
        for wid, g in curves.groupby(WELL_COL, sort=False):
            n_all = int(len(g))
            n_lab = int(g[LABEL_COL].notna().sum())
            cov_rows.append(
                {
                    WELL_COL: wid,
                    "block": g["block"].iloc[0],
                    "n_curve_rows": n_all,
                    "n_labeled_rows": n_lab,
                    "label_frac": float(n_lab / n_all) if n_all else 0.0,
                    "depth_min_curve": float(g[DEPTH_COL].min()),
                    "depth_max_curve": float(g[DEPTH_COL].max()),
                    "depth_min_labeled": float(g.loc[g[LABEL_COL].notna(), DEPTH_COL].min())
                    if n_lab
                    else float("nan"),
                    "depth_max_labeled": float(g.loc[g[LABEL_COL].notna(), DEPTH_COL].max())
                    if n_lab
                    else float("nan"),
                    "n_classes": int(g.loc[g[LABEL_COL].notna(), LABEL_COL].nunique()) if n_lab else 0,
                }
            )
        cov_df = pd.DataFrame(cov_rows).sort_values(WELL_COL)
        cov_df.to_csv(PROCESSED_DIR / "label_coverage_by_well.csv", index=False)

        well_summary = (
            labeled.groupby(WELL_COL)
            .agg(
                block=("block", "first"),
                n_rows=(LABEL_COL, "size"),
                depth_min=(DEPTH_COL, "min"),
                depth_max=(DEPTH_COL, "max"),
                n_classes=(LABEL_COL, "nunique"),
                conf_mean=(CONF_COL, "mean"),
            )
            .reset_index()
        )
        well_summary = well_summary.merge(
            cov_df[[WELL_COL, "n_curve_rows", "label_frac"]],
            on=WELL_COL,
            how="left",
        )
        well_summary.to_csv(PROCESSED_DIR / "well_summary.csv", index=False)

        class_counts = (
            labeled.groupby(["lithology", "lithology_name"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        class_counts.to_csv(PROCESSED_DIR / "class_counts.csv", index=False)

        curve_cov = []
        for c in CORE_CURVES:
            curve_cov.append(
                {
                    "curve": c,
                    "non_null_curves_all": int(curves[c].notna().sum()),
                    "non_null_frac_curves_all": float(curves[c].notna().mean()),
                    "non_null_labeled": int(labeled[c].notna().sum()),
                    "non_null_frac_labeled": float(labeled[c].notna().mean()),
                    "wells_with_curve": int(
                        curves.groupby(WELL_COL)[c].apply(lambda s: s.notna().any()).sum()
                    ),
                }
            )
        pd.DataFrame(curve_cov).to_csv(PROCESSED_DIR / "curve_coverage.csv", index=False)

        n_curve = int(len(curves))
        n_lab = int(len(labeled))
        meta = {
            "n_rows_curves_all": n_curve,
            "n_rows_labeled": n_lab,
            "label_frac_corpus": float(n_lab / n_curve) if n_curve else 0.0,
            "n_wells": int(labeled[WELL_COL].nunique()),
            "n_classes": int(labeled[LABEL_COL].nunique()),
            "label_frac_well_mean": float(cov_df["label_frac"].mean()) if len(cov_df) else 0.0,
            "label_frac_well_min": float(cov_df["label_frac"].min()) if len(cov_df) else 0.0,
            "label_frac_well_max": float(cov_df["label_frac"].max()) if len(cov_df) else 0.0,
            "wells_label_frac_lt_1": int((cov_df["label_frac"] < 1.0).sum()) if len(cov_df) else 0,
            "blocks": sorted(labeled["block"].astype(str).unique().tolist()),
            "parse_errors": raw.attrs.get("parse_errors", []),
            "stats_support": "full_curve",
            "paths": {
                "curves_all": str(curves_path),
                "labeled": str(out_path),
                "label_coverage_by_well": str(PROCESSED_DIR / "label_coverage_by_well.csv"),
                "well_summary": str(PROCESSED_DIR / "well_summary.csv"),
            },
        }
        (PROCESSED_DIR / "dataset_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return labeled


def load_labeled(path: Path | None = None) -> pd.DataFrame:
    path = path or (PROCESSED_DIR / LABELED_NAME)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run prepare_dataset first.")
    return pd.read_parquet(path)


def load_curves_all(path: Path | None = None) -> pd.DataFrame:
    """Full wireline table (labeled + unlabeled depths) for well aggregates."""
    path = path or (PROCESSED_DIR / CURVES_ALL_NAME)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Re-run prepare_dataset to build full-curve stats support."
        )
    return pd.read_parquet(path)
