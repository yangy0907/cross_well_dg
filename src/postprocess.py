"""Post-processing and domain-similarity utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _majority(x: np.ndarray) -> float:
    vals, counts = np.unique(x, return_counts=True)
    return float(vals[np.argmax(counts)])


def _segment_indices(depths_sorted: np.ndarray, gap_factor: float = 1.5) -> list[np.ndarray]:
    """Split a depth-ordered well into contiguous segments; break on large gaps."""
    n = len(depths_sorted)
    if n == 0:
        return []
    if n == 1:
        return [np.array([0], dtype=int)]
    dd = np.diff(depths_sorted.astype(np.float64))
    pos = dd[dd > 0]
    med = float(np.median(pos)) if len(pos) else 0.0
    if med <= 0:
        # Constant / irregular: treat as one segment.
        return [np.arange(n, dtype=int)]
    thresh = gap_factor * med
    breaks = np.where(dd > thresh)[0]  # break after these positions
    segs: list[np.ndarray] = []
    start = 0
    for b in breaks:
        segs.append(np.arange(start, b + 1, dtype=int))
        start = b + 1
    segs.append(np.arange(start, n, dtype=int))
    return segs


def smooth_predictions_by_well(
    pred: np.ndarray,
    wells: np.ndarray,
    depths: np.ndarray,
    window: int = 7,
    *,
    gap_factor: float = 1.5,
) -> np.ndarray:
    """Majority-vote smoothing within each well along depth order.

    Votes never cross wells. Within a well, rolling windows are restricted to
    contiguous depth segments: a gap larger than ``gap_factor`` × median positive
    Δdepth starts a new segment (no vote across logging breaks / unlabeled holes
    that leave large depth jumps in the scored support).
    """
    if window < 3:
        return pred
    out = pred.copy()
    wells = np.asarray(wells)
    depths = np.asarray(depths)
    k = window if window % 2 == 1 else window + 1
    for w in pd.unique(wells):
        idx = np.where(wells == w)[0]
        if len(idx) < 3:
            continue
        order = idx[np.argsort(depths[idx], kind="mergesort")]
        d_ord = depths[order]
        seq = out[order].astype(np.float64)
        for seg in _segment_indices(d_ord, gap_factor=gap_factor):
            if len(seg) < 3:
                continue
            smoothed = (
                pd.Series(seq[seg])
                .rolling(k, center=True, min_periods=1)
                .apply(_majority, raw=True)
                .to_numpy()
            )
            out[order[seg]] = smoothed.astype(out.dtype, copy=False)
    return out


def well_mean_vectors(
    X: np.ndarray, wells: np.ndarray, feature_idx: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Mean feature vector per well (optionally on a feature subset)."""
    if feature_idx is not None:
        X = X[:, feature_idx]
    out: dict[str, np.ndarray] = {}
    for w in pd.unique(wells):
        m = wells == w
        out[str(w)] = X[m].mean(axis=0)
    return out


def similarity_sample_weights(
    X_train: np.ndarray,
    wells_train: np.ndarray,
    X_target: np.ndarray,
    wells_target: np.ndarray,
    feature_names: list[str] | None = None,
    temperature: float = 1.0,
    *,
    normalize_by_mean_distance: bool = True,
    X_source_means: np.ndarray | None = None,
    wells_source_means: np.ndarray | None = None,
) -> np.ndarray:
    """
    Weight source samples by how similar their well is to the unlabeled target domain.

    sim(w) = exp(-||mu_w - mu_tgt|| / (T * scale)), where scale is the mean source-well
    distance to mu_tgt when normalize_by_mean_distance is True (stabilises T across splits),
    else scale=1. Weights are finally rescaled to unit mean.

    Prefer full-curve (label-independent) arrays for both ``X_target`` and
    ``X_source_means`` so mu_w and mu_t share the same support. When
    ``X_source_means`` is omitted, source means fall back to labelled ``X_train``.
    Returned weights are always aligned to ``wells_train`` (supervised rows).
    """
    idx = None
    if feature_names is not None:
        prefer = []
        for i, n in enumerate(feature_names):
            if n.endswith("_wn"):
                continue
            base = n.replace("_raw", "")
            if base in {"GR", "RHOB", "NPHI", "DTC", "RDEP", "RMED", "CALI", "PEF"}:
                prefer.append(i)
        if len(prefer) >= 4:
            idx = np.asarray(prefer, dtype=int)

    X_mu = X_train if X_source_means is None else X_source_means
    w_mu = wells_train if wells_source_means is None else wells_source_means
    src_mu = well_mean_vectors(X_mu, w_mu, idx)
    tgt_mu = well_mean_vectors(X_target, wells_target, idx)
    if not tgt_mu:
        return np.ones(len(wells_train), dtype=np.float64)
    tgt_center = np.mean(np.stack(list(tgt_mu.values()), axis=0), axis=0)

    dists = {w: float(np.linalg.norm(mu - tgt_center)) for w, mu in src_mu.items()}
    vals = np.asarray(list(dists.values()), dtype=np.float64)
    scale = float(vals.mean() + 1e-6) if normalize_by_mean_distance else 1.0
    sim = {w: float(np.exp(-d / (temperature * scale))) for w, d in dists.items()}
    w = np.array(
        [sim.get(str(ww), 1.0) for ww in wells_train], dtype=np.float64
    )
    return w / (w.mean() + 1e-12)
