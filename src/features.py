"""Feature engineering: global / well-norm / hybrid representations."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .constants import CONF_COL, CORE_CURVES, DEPTH_COL, LABEL_COL, WELL_COL

# High-coverage curves preferred for alignment / similarity
ALIGN_CURVES = ["GR", "RHOB", "NPHI", "DTC", "RDEP", "RMED", "CALI", "PEF"]


@dataclass
class FeatureBundle:
    X: np.ndarray
    y: np.ndarray
    conf: np.ndarray
    wells: np.ndarray
    feature_names: list[str]
    depths: np.ndarray


@dataclass
class FeaturePipeline:
    """Fit on train wells only; transform any split.

    When ``curve_context`` is provided (full wireline table), whole-well μ/σ,
    causal prefix z-scores and relative depth use all observable curve depths
    for that well, independent of label availability. Features are still emitted
    only for rows in ``df`` (typically labeled depths used for train/score).
    """

    mode: str = "wellnorm"  # global | wellnorm | hybrid
    use_missing_mask: bool = True
    use_relative_depth: bool = True
    # full = post-drilling whole-well μ/σ; causal = expanding prefix along depth.
    well_z_mode: str = "full"  # full | causal
    use_local_stats: bool = False
    curves: list[str] | None = None
    fill_value: float = 0.0
    # Set before fit/transform to enable full-curve well aggregates.
    curve_context: pd.DataFrame | None = field(default=None, repr=False)

    _imputer: SimpleImputer | None = None
    _feature_names: list[str] | None = None
    _global_median: pd.Series | None = None
    _global_std: pd.Series | None = None

    def _active_curves(self, df: pd.DataFrame) -> list[str]:
        curves = self.curves or CORE_CURVES
        return [c for c in curves if c in df.columns]

    def _context_for_wells(self, wells: np.ndarray) -> pd.DataFrame:
        """Return curve rows for the given wells (full curves if context set)."""
        well_set = set(map(str, pd.unique(wells)))
        if self.curve_context is None:
            return pd.DataFrame()
        ctx = self.curve_context
        mask = ctx[WELL_COL].astype(str).isin(well_set)
        return ctx.loc[mask]

    def _well_mu_sd(self, ctx_well: pd.DataFrame, col: str) -> tuple[float, float]:
        s = ctx_well[col].to_numpy(dtype=np.float64)
        valid = ~np.isnan(s)
        if not valid.any():
            return 0.0, 1.0
        mu = float(np.nanmean(s))
        sd = float(np.nanstd(s, ddof=0))
        if sd < 1e-12 or np.isnan(sd):
            sd = 1.0
        return mu, sd

    def _well_zscore_from_context(self, out: pd.DataFrame, col: str) -> pd.Series:
        """Z-score labeled/request rows using μ/σ from full-curve context when available."""
        z = np.zeros(len(out), dtype=np.float64)
        ctx = self._context_for_wells(out[WELL_COL].to_numpy())
        if ctx.empty:
            # Fallback: labeled-only moments (legacy path).
            return out.groupby(WELL_COL)[col].transform(
                lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-6)
                if s.std(ddof=0) and s.std(ddof=0) > 0
                else s * 0.0
            )
        ctx_by_well = {str(w): g for w, g in ctx.groupby(WELL_COL, sort=False)}
        for w, idx in out.groupby(WELL_COL, sort=False).groups.items():
            g_ctx = ctx_by_well.get(str(w))
            if g_ctx is None or g_ctx.empty:
                s = out.loc[idx, col]
                mu = float(s.mean()) if len(s) else 0.0
                sd = float(s.std(ddof=0)) if len(s) else 1.0
                if sd < 1e-12 or np.isnan(sd):
                    sd = 1.0
            else:
                mu, sd = self._well_mu_sd(g_ctx, col)
            vals = out.loc[idx, col].to_numpy(dtype=np.float64)
            z_w = np.where(np.isnan(vals), 0.0, (vals - mu) / (sd + 1e-6))
            # map index positions
            pos = out.index.get_indexer(idx)
            z[pos] = z_w
        return pd.Series(z, index=out.index)

    def _causal_z_lookup(
        self, depths_query: np.ndarray, depths_full: np.ndarray, values_full: np.ndarray
    ) -> np.ndarray:
        """Expanding μ/σ on full depth-ordered curve; lookup z at query depths."""
        order = np.argsort(depths_full, kind="mergesort")
        d_ord = depths_full[order]
        s_ord = values_full[order]
        valid = ~np.isnan(s_ord)
        cnt = np.cumsum(valid.astype(np.float64))
        s0 = np.where(valid, s_ord, 0.0)
        csum = np.cumsum(s0)
        csum2 = np.cumsum(s0 * s0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.where(cnt > 0, csum / np.maximum(cnt, 1.0), 0.0)
            var = np.where(cnt > 0, csum2 / np.maximum(cnt, 1.0) - mu * mu, 0.0)
        sd = np.sqrt(np.maximum(var, 0.0))
        sd = np.where(sd < 1e-12, 1.0, sd)
        z_ord = np.where(valid, (s_ord - mu) / (sd + 1e-6), 0.0)

        # Per-query searchsorted so unordered / descending LAS depths stay causal.
        # side='right' on (query + eps) yields the last full-curve index with depth <= query.
        q = np.asarray(depths_query, dtype=np.float64)
        idx = np.searchsorted(d_ord, q + 1e-9, side="right") - 1
        z_out = np.zeros(len(q), dtype=np.float64)
        valid_q = idx >= 0
        if valid_q.any():
            z_out[valid_q] = z_ord[idx[valid_q]]
        return z_out

    def _causal_well_zscore_from_context(self, out: pd.DataFrame, col: str) -> pd.Series:
        z = np.zeros(len(out), dtype=np.float64)
        ctx = self._context_for_wells(out[WELL_COL].to_numpy())
        ctx_by_well = (
            {str(w): g for w, g in ctx.groupby(WELL_COL, sort=False)} if not ctx.empty else {}
        )
        for w, idx in out.groupby(WELL_COL, sort=False).groups.items():
            g_out = out.loc[idx]
            g_ctx = ctx_by_well.get(str(w))
            if g_ctx is None or g_ctx.empty:
                # Fallback: causal on request rows only.
                order = np.argsort(g_out[DEPTH_COL].to_numpy(), kind="mergesort")
                s = g_out[col].to_numpy(dtype=np.float64)
                s_ord = s[order]
                valid = ~np.isnan(s_ord)
                cnt = np.cumsum(valid.astype(np.float64))
                s0 = np.where(valid, s_ord, 0.0)
                csum = np.cumsum(s0)
                csum2 = np.cumsum(s0 * s0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    mu = np.where(cnt > 0, csum / np.maximum(cnt, 1.0), 0.0)
                    var = np.where(cnt > 0, csum2 / np.maximum(cnt, 1.0) - mu * mu, 0.0)
                sd = np.sqrt(np.maximum(var, 0.0))
                sd = np.where(sd < 1e-12, 1.0, sd)
                z_ord = np.where(valid, (s_ord - mu) / (sd + 1e-6), 0.0)
                z_w = np.empty_like(z_ord)
                z_w[order] = z_ord
            else:
                z_w = self._causal_z_lookup(
                    g_out[DEPTH_COL].to_numpy(dtype=np.float64),
                    g_ctx[DEPTH_COL].to_numpy(dtype=np.float64),
                    g_ctx[col].to_numpy(dtype=np.float64),
                )
            pos = out.index.get_indexer(idx)
            z[pos] = z_w
        return pd.Series(z, index=out.index)

    def _rel_depth_from_context(self, out: pd.DataFrame) -> pd.Series:
        ctx = self._context_for_wells(out[WELL_COL].to_numpy())
        if ctx.empty:
            dmin = out.groupby(WELL_COL)[DEPTH_COL].transform("min")
            dmax = out.groupby(WELL_COL)[DEPTH_COL].transform("max")
            return (out[DEPTH_COL] - dmin) / (dmax - dmin + 1e-6)
        stats = (
            ctx.groupby(WELL_COL)[DEPTH_COL]
            .agg(dmin="min", dmax="max")
            .reset_index()
        )
        merged = out[[WELL_COL, DEPTH_COL]].merge(stats, on=WELL_COL, how="left")
        dmin = merged["dmin"].to_numpy(dtype=np.float64)
        dmax = merged["dmax"].to_numpy(dtype=np.float64)
        # If a well is missing from context, fall back to request-row min/max.
        miss = np.isnan(dmin) | np.isnan(dmax)
        if miss.any():
            dmin_fb = out.groupby(WELL_COL)[DEPTH_COL].transform("min").to_numpy()
            dmax_fb = out.groupby(WELL_COL)[DEPTH_COL].transform("max").to_numpy()
            dmin = np.where(miss, dmin_fb, dmin)
            dmax = np.where(miss, dmax_fb, dmax)
        depth = out[DEPTH_COL].to_numpy(dtype=np.float64)
        return pd.Series((depth - dmin) / (dmax - dmin + 1e-6), index=out.index)

    def _ensure_meta(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if LABEL_COL not in out.columns:
            out[LABEL_COL] = np.nan
        if CONF_COL not in out.columns:
            out[CONF_COL] = 1.0
        else:
            out[CONF_COL] = out[CONF_COL].fillna(1.0)
        return out

    def _engineer(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = self._ensure_meta(df)
        curves = self._active_curves(df)
        base_cols = [WELL_COL, DEPTH_COL, LABEL_COL, CONF_COL] + curves
        out = df[base_cols].copy()

        feat_cols: list[str] = []

        # --- raw / globally standardized curves ---
        if self.mode in {"global", "hybrid"}:
            raw = out[curves].copy()
            if fit:
                self._global_median = raw.median(numeric_only=True)
                self._global_std = raw.std(ddof=0, numeric_only=True).replace(0, np.nan).fillna(1.0)
            assert self._global_median is not None and self._global_std is not None
            for c in curves:
                med = float(self._global_median.get(c, 0.0))
                std = float(self._global_std.get(c, 1.0)) + 1e-6
                name = c if self.mode == "global" else f"{c}_raw"
                out[name] = (raw[c] - med) / std
                feat_cols.append(name)

        # --- well-wise z-score (full-well or causal prefix on curve_context) ---
        if self.mode in {"wellnorm", "hybrid"}:
            for c in curves:
                name = c if self.mode == "wellnorm" else f"{c}_wn"
                if self.well_z_mode == "causal":
                    out[name] = self._causal_well_zscore_from_context(out, c)
                else:
                    out[name] = self._well_zscore_from_context(out, c)
                feat_cols.append(name)

        # relative depth from full logged interval when context available
        if self.use_relative_depth:
            out["rel_depth"] = self._rel_depth_from_context(out)
            feat_cols.append("rel_depth")

        if self.use_local_stats:
            src = [f"{c}_wn" for c in curves] if self.mode == "hybrid" else curves
            for c, src_c in zip(curves, src):
                if src_c not in out.columns:
                    continue
                name = f"{c}_roll_mean"
                out[name] = (
                    out.groupby(WELL_COL)[src_c]
                    .transform(lambda s: s.rolling(5, min_periods=1, center=True).mean())
                )
                feat_cols.append(name)

        if self.use_missing_mask:
            for c in curves:
                m = f"{c}_isna"
                out[m] = out[c].isna().astype(np.float32)
                feat_cols.append(m)

        mat = out[feat_cols].copy()
        # Keep column count stable: all-NaN features → 0 so imputer does not drop them.
        for c in feat_cols:
            if mat[c].isna().all():
                mat[c] = 0.0
        if fit:
            self._imputer = SimpleImputer(
                strategy="median", fill_value=self.fill_value, keep_empty_features=True
            )
            arr = self._imputer.fit_transform(mat)
            self._feature_names = feat_cols
        else:
            if self._imputer is None or self._feature_names is None:
                raise RuntimeError("Pipeline not fitted")
            for c in self._feature_names:
                if c not in out.columns:
                    out[c] = np.nan
            mat2 = out[self._feature_names].copy()
            for c in self._feature_names:
                if mat2[c].isna().all():
                    mat2[c] = 0.0
            arr = self._imputer.transform(mat2)

        feat = pd.DataFrame(arr, columns=self._feature_names, index=out.index)
        feat[WELL_COL] = out[WELL_COL].values
        feat[DEPTH_COL] = out[DEPTH_COL].values
        feat[LABEL_COL] = out[LABEL_COL].values
        feat[CONF_COL] = out[CONF_COL].values
        return feat

    def fit_transform(
        self, df: pd.DataFrame, curve_context: pd.DataFrame | None = None
    ) -> FeatureBundle:
        if curve_context is not None:
            self.curve_context = curve_context
        return self._to_bundle(self._engineer(df, fit=True))

    def transform(
        self, df: pd.DataFrame, curve_context: pd.DataFrame | None = None
    ) -> FeatureBundle:
        if curve_context is not None:
            self.curve_context = curve_context
        return self._to_bundle(self._engineer(df, fit=False))

    def _to_bundle(self, feat: pd.DataFrame) -> FeatureBundle:
        assert self._feature_names is not None
        y_raw = feat[LABEL_COL].to_numpy()
        # Unlabeled rows (curve-only transforms) get sentinel -1
        y = np.array(
            [int(v) if pd.notna(v) else -1 for v in y_raw],
            dtype=np.int32,
        )
        return FeatureBundle(
            X=feat[self._feature_names].to_numpy(dtype=np.float32),
            y=y,
            conf=feat[CONF_COL].to_numpy(dtype=np.float32),
            wells=feat[WELL_COL].to_numpy(),
            feature_names=list(self._feature_names),
            depths=feat[DEPTH_COL].to_numpy(dtype=np.float32),
        )


def encode_labels(y_train: np.ndarray, y_other: np.ndarray | None = None):
    """Map lithology codes to 0..K-1 based on train labels."""
    classes = np.array(sorted(pd.unique(y_train)))
    mapping = {int(c): i for i, c in enumerate(classes)}
    inv = {i: int(c) for c, i in mapping.items()}

    def _map(y: np.ndarray) -> np.ndarray:
        return np.array([mapping[int(v)] for v in y], dtype=np.int32)

    yt = _map(y_train)
    yo = _map(y_other) if y_other is not None else None
    return yt, yo, mapping, inv
