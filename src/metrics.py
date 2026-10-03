"""Evaluation metrics for cross-well lithofacies prediction."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def overall_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int] | None = None,
) -> dict[str, float]:
    """Pooled metrics. If labels is set, Macro-F1 averages over that fixed set."""
    kw: dict[str, Any] = {"average": "macro", "zero_division": 0}
    if labels is not None:
        kw["labels"] = list(labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, **kw)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
        ),
    }


def well_wise_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    wells: np.ndarray,
    *,
    label_mode: str = "well_true",
    global_labels: Sequence[int] | None = None,
) -> pd.DataFrame:
    """
    Per-well Macro-F1.

    label_mode:
      - well_true: average over classes present in that well's y_true (same denominator across methods)
      - fixed: use global_labels for every well
    """
    rows = []
    for w in pd.unique(wells):
        m = wells == w
        if m.sum() == 0:
            continue
        yt, yp = y_true[m], y_pred[m]
        if label_mode == "fixed" and global_labels is not None:
            labels = list(global_labels)
        else:
            labels = sorted(np.unique(yt).tolist())
        rows.append(
            {
                "well_id": w,
                "n": int(m.sum()),
                "n_classes_true": int(len(np.unique(yt))),
                "macro_f1": float(
                    f1_score(yt, yp, average="macro", zero_division=0, labels=labels)
                ),
                "accuracy": float(accuracy_score(yt, yp)),
            }
        )
    return pd.DataFrame(rows).sort_values("macro_f1")


def summarize_well_f1(well_df: pd.DataFrame, ddof: int = 1) -> dict[str, float]:
    """Aggregate per-well Macro-F1.

    W10 (worst10pct_well_f1) = mean of the worst ``ceil(0.1 * n_wells)`` wells.
    With Protocol-1 ``n_test_wells=17``, that is ``ceil(1.7)=2`` wells.
    """
    if well_df.empty:
        return {
            "well_macro_f1_mean": 0.0,
            "well_macro_f1_median": 0.0,
            "well_macro_f1_std": 0.0,
            "worst10pct_well_f1": 0.0,
            "worst_well_f1": 0.0,
        }
    s = well_df["macro_f1"]
    k = max(1, int(np.ceil(0.1 * len(s))))
    worst10 = float(s.nsmallest(k).mean())
    return {
        "well_macro_f1_mean": float(s.mean()),
        "well_macro_f1_median": float(s.median()),
        "well_macro_f1_std": float(s.std(ddof=ddof)) if len(s) > 1 else 0.0,
        "worst10pct_well_f1": worst10,
        "worst_well_f1": float(s.min()),
    }


def evaluate_split(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    wells: np.ndarray,
    label_names: dict[int, str] | None = None,
    *,
    pooled_labels: Sequence[int] | None = None,
    fixed_well_labels: Sequence[int] | None = None,
) -> dict[str, Any]:
    """
    pooled_labels: label set for pooled Macro-F1 (default: classes in y_true).

    Primary per-well Macro-F1 uses classes present in that well's ground truth
    (``well_true``). Companion ``well_summary_fixed`` / ``well_table_fixed`` use
    ``fixed_well_labels`` on every well (default: same as ``pooled_labels``).

    Canonical release definition (train.py): ``fixed_well_labels`` = train-learnable
    class indices only (excludes open-set unk), so worst-domain risk shares one
    denominator across wells within a split.
    """
    if pooled_labels is None:
        pooled_labels = sorted(np.unique(y_true).tolist())
    if fixed_well_labels is None:
        fixed_well_labels = list(pooled_labels)
    overall = overall_metrics(y_true, y_pred, labels=pooled_labels)
    well_df = well_wise_f1(y_true, y_pred, wells, label_mode="well_true")
    summary = summarize_well_f1(well_df, ddof=1)
    well_df_fixed = well_wise_f1(
        y_true,
        y_pred,
        wells,
        label_mode="fixed",
        global_labels=fixed_well_labels,
    )
    summary_fixed = summarize_well_f1(well_df_fixed, ddof=1)
    if not well_df.empty:
        # Consistency: worst-well F1 must equal min of per-well Macro-F1
        wmin = float(well_df["macro_f1"].min())
        if abs(summary["worst_well_f1"] - wmin) > 1e-9:
            raise AssertionError(
                f"worst_well_f1={summary['worst_well_f1']} != min(per_well)={wmin}"
            )
    report = classification_report(
        y_true, y_pred, zero_division=0, output_dict=True, labels=list(pooled_labels)
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(pooled_labels))
    return {
        "overall": overall,
        "well_summary": summary,
        "well_table": well_df,
        "well_summary_fixed": summary_fixed,
        "well_table_fixed": well_df_fixed,
        "classification_report": report,
        "confusion_matrix": cm,
        "label_names": label_names or {},
        "pooled_labels": list(pooled_labels),
        "fixed_well_labels": list(fixed_well_labels),
    }


def evaluate_known_class_only(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    wells: np.ndarray,
    *,
    unk_id: int,
    label_names: dict[int, str] | None = None,
) -> dict[str, Any] | None:
    """Macro-F1 / worst-well on rows whose true label is not ``unk_id``.

    Unseen-class rows are excluded (not scored as a forced-zero class). Returns
    None when no known-class rows remain.
    """
    mask = np.asarray(y_true) != int(unk_id)
    if not mask.any():
        return None
    yt = np.asarray(y_true)[mask]
    yp = np.asarray(y_pred)[mask]
    ww = np.asarray(wells)[mask]
    labels = sorted(np.unique(yt).tolist())
    return evaluate_split(yt, yp, ww, label_names=label_names, pooled_labels=labels)
