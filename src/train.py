"""Train / evaluate experiment runners."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import LITHOLOGY_MAP, RESULTS_DIR, WELL_COL
from .features import FeaturePipeline
from .metrics import evaluate_known_class_only, evaluate_split
from .models import LGBMConfig, LGBMLithologyModel, MLPCoralModel, RFLithologyModel, XGBLithologyModel
from .provenance import collect_run_provenance
from .splits import apply_split, naive_depth_split

# Bump when feature routing, metrics, curve-support, CORAL, or resume provenance change.
PIPELINE_VERSION = "r3_fullcurve_stats_20260930"

# Inductive DG: train-fit stats only (no target-well aggregates such as rel_depth / well z).
INDUCTIVE_DG_METHODS = {
    "lgbm_global",
    "lgbm_global_eq120k",
    "rf_global",
    "rf_global_full",
    "xgb_global",
    "mlp_source_only",
    "mlp_coral_uda",
    "mlp_coral_strong",
}

# Module-level curve context set by run_protocol_experiment / CLI.
_CURVE_CONTEXT: pd.DataFrame | None = None
_ACTIVE_SPLIT: dict[str, Any] | None = None
_CONFIG_PATH: Path | None = None


def set_curve_context(df: pd.DataFrame | None) -> None:
    global _CURVE_CONTEXT
    _CURVE_CONTEXT = df


def set_run_meta(*, split: dict[str, Any] | None = None, config_path: Path | None = None) -> None:
    global _ACTIVE_SPLIT, _CONFIG_PATH
    if split is not None:
        _ACTIVE_SPLIT = split
    if config_path is not None:
        _CONFIG_PATH = config_path


def _remap_bundle_y(bundle_y: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    return np.array([mapping[int(v)] for v in bundle_y], dtype=np.int32)


def _subsample_rows(
    X: np.ndarray,
    y: np.ndarray,
    *arrays: np.ndarray,
    max_rows: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    if len(y) <= max_rows:
        return (X, y, *arrays)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=max_rows, replace=False)
    out = [X[idx], y[idx]]
    for a in arrays:
        out.append(a[idx])
    return tuple(out)


def _feature_routing(method: str) -> tuple[str, bool, str]:
    """Return (feat_mode, use_relative_depth, well_z_mode)."""
    if method in INDUCTIVE_DG_METHODS:
        return "global", False, "full"
    if method == "lgbm_hybrid_while":
        return "hybrid", False, "causal"
    if method == "lgbm_hybrid_norel":
        return "hybrid", False, "full"
    if method == "lgbm_wellnorm_norel":
        # Completes the 2x2: well-z ± relative depth × ± global levels.
        return "wellnorm", False, "full"
    if method.startswith("lgbm_hybrid") or method in {
        "lgbm_hybrid",
        "lgbm_hybrid_sim",
        "lgbm_hybrid_smooth",
        "lgbm_hybrid_plus",
        "lgbm_hybrid_rareboost",
        "lgbm_hybrid_sim_perwell",
        "lgbm_hybrid_plus_perwell",
    }:
        return "hybrid", True, "full"
    if method in {"lgbm_wellnorm", "lgbm_wellnorm_rel"}:
        return "wellnorm", True, "full"
    return "wellnorm", True, "full"


def _curve_rows_for_wells(wells: np.ndarray) -> pd.DataFrame | None:
    if _CURVE_CONTEXT is None:
        return None
    well_set = set(map(str, pd.unique(wells)))
    return _CURVE_CONTEXT[_CURVE_CONTEXT[WELL_COL].astype(str).isin(well_set)].copy()


def run_method_on_split(
    parts: dict[str, pd.DataFrame],
    method: str,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Methods:
      - lgbm_global / lgbm_global_eq120k / rf_global / rf_global_full / xgb_global / mlp_*: inductive DG
      - lgbm_wellnorm / lgbm_hybrid*: test-domain norm on full-curve stats
      - lgbm_hybrid_while: causal prefix well-z on full curves
    """
    from .postprocess import similarity_sample_weights, smooth_predictions_by_well

    t0 = time.time()
    train_df, val_df, test_df = parts["train"], parts["val"], parts["test"]

    feat_mode, use_rel_depth, well_z_mode = _feature_routing(method)

    use_conf = method in {
        "lgbm_conf",
        "lgbm_full",
        "lgbm_groupdro",
        "lgbm_hybrid_plus",
        "lgbm_hybrid_plus_perwell",
    }
    use_balance = method == "lgbm_full"
    groupdro = 2 if method == "lgbm_groupdro" else 0
    use_sim = method in {
        "lgbm_hybrid_sim",
        "lgbm_hybrid_plus",
        "lgbm_hybrid_sim_perwell",
        "lgbm_hybrid_plus_perwell",
    }
    use_smooth = method in {
        "lgbm_hybrid_smooth",
        "lgbm_hybrid_plus",
        "lgbm_hybrid_plus_perwell",
    }
    use_rareboost = method == "lgbm_hybrid_rareboost"
    per_well_sim = method in {"lgbm_hybrid_sim_perwell", "lgbm_hybrid_plus_perwell"}
    eq120k = method in {"lgbm_global_eq120k", "rf_global"}  # subsampled schedules only

    pipe = FeaturePipeline(
        mode=feat_mode,
        use_missing_mask=True,
        use_relative_depth=use_rel_depth,
        well_z_mode=well_z_mode,
        use_local_stats=False,
        curve_context=_CURVE_CONTEXT,
    )

    tr = pipe.fit_transform(train_df)
    va = pipe.transform(val_df)
    te = pipe.transform(test_df)

    # Full-curve features for similarity means (label-independent support on
    # both source and target wells; supervised fit still uses labelled Xtr).
    te_sim_X, te_sim_wells = te.X, te.wells
    tr_sim_X, tr_sim_wells = tr.X, tr.wells
    if use_sim and _CURVE_CONTEXT is not None:
        tgt_curves = _curve_rows_for_wells(te.wells)
        if tgt_curves is not None and len(tgt_curves) > 0:
            te_full = pipe.transform(tgt_curves)
            te_sim_X, te_sim_wells = te_full.X, te_full.wells
        src_curves = _curve_rows_for_wells(tr.wells)
        if src_curves is not None and len(src_curves) > 0:
            tr_full = pipe.transform(src_curves)
            tr_sim_X, tr_sim_wells = tr_full.X, tr_full.wells

    classes = sorted(pd.unique(tr.y).tolist())
    classes = [c for c in classes if int(c) >= 0]
    mapping = {int(c): i for i, c in enumerate(classes)}
    n_cls = len(mapping)
    unk_id = n_cls

    def _pack_train_val(bundle):
        mask = np.array([int(v) in mapping for v in bundle.y])
        return (
            bundle.X[mask],
            _remap_bundle_y(bundle.y[mask], mapping),
            bundle.conf[mask],
            bundle.wells[mask],
            bundle.depths[mask],
            int((~mask).sum()),
            int(mask.sum()),
        )

    def _pack_test(bundle):
        y = np.array(
            [mapping[int(v)] if int(v) in mapping else unk_id for v in bundle.y],
            dtype=np.int32,
        )
        n_unseen = int((y == unk_id).sum())
        return bundle.X, y, bundle.conf, bundle.wells, bundle.depths, n_unseen, len(y)

    Xtr, ytr, ctr, wtr, dtr, n_drop_tr, n_keep_tr = _pack_train_val(tr)
    Xva, yva, cva, wva, dva, n_drop_va, n_keep_va = _pack_train_val(va)
    Xte, yte, cte, wte, dte, n_unseen_te, n_keep_te = _pack_test(te)

    pred: np.ndarray
    sim_mode = "none"
    smoothing_scope = "none"
    model_params: dict[str, Any] = {}
    # n_train = eligible labelled train rows after label filter; n_train_fit = rows
    # actually passed to fit after any schedule subsample (MLP/RF/eq120k).
    n_train_fit = int(n_keep_tr)
    early_stopping_metric = "none"
    n_estimators_budget: int | None = None

    if method == "mlp_source_only":
        max_tr = 120_000
        Xtr_s, ytr_s = _subsample_rows(Xtr, ytr, max_rows=max_tr, seed=seed)
        n_train_fit = int(len(ytr_s))
        early_stopping_metric = "sklearn_accuracy_holdout_10pct"
        model = MLPCoralModel(use_coral=False, random_state=seed, max_iter=35, hidden=(96, 48))
        model.fit(Xtr_s, ytr_s, X_target_unlabeled=None)
        pred = model.predict(Xte)
        model_params = {"max_rows": max_tr, "hidden": (96, 48)}
    elif method == "mlp_coral_uda":
        max_tr, max_tgt = 120_000, 40_000
        Xtr_s, ytr_s = _subsample_rows(Xtr, ytr, max_rows=max_tr, seed=seed)
        Xte_coral, _ = _subsample_rows(Xte, yte, max_rows=max_tgt, seed=seed + 1)
        n_train_fit = int(len(ytr_s))
        early_stopping_metric = "sklearn_accuracy_holdout_10pct"
        model = MLPCoralModel(use_coral=True, random_state=seed, max_iter=35, hidden=(96, 48))
        model.fit(Xtr_s, ytr_s, X_target_unlabeled=Xte_coral)
        pred = model.predict(Xte)
        # Labelled-score positions with labels hidden — not full-curve target UDA.
        sim_mode = "transductive_score_support"
        model_params = {"max_rows": max_tr, "max_tgt": max_tgt}
    elif method == "mlp_coral_strong":
        max_tr, max_tgt = 200_000, 80_000
        Xtr_s, ytr_s = _subsample_rows(Xtr, ytr, max_rows=max_tr, seed=seed)
        Xte_coral, _ = _subsample_rows(Xte, yte, max_rows=max_tgt, seed=seed + 1)
        n_train_fit = int(len(ytr_s))
        early_stopping_metric = "sklearn_accuracy_holdout_10pct"
        model = MLPCoralModel(
            use_coral=True,
            random_state=seed,
            max_iter=60,
            hidden=(256, 128),
        )
        model.fit(Xtr_s, ytr_s, X_target_unlabeled=Xte_coral)
        pred = model.predict(Xte)
        sim_mode = "transductive_score_support"
        model_params = {"max_rows": max_tr, "max_tgt": max_tgt, "hidden": (256, 128)}
    elif method == "rf_global":
        model = RFLithologyModel(n_estimators=200, random_state=seed, max_rows=120_000)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        n_train_fit = int(min(len(ytr), 120_000))
        early_stopping_metric = "none"
        n_estimators_budget = 200
        model_params = {"n_estimators": 200, "max_rows": 120_000}
    elif method == "rf_global_full":
        # Equal-row inductive RF vs full-split LightGBM (no 120k ceiling).
        model = RFLithologyModel(
            n_estimators=100,
            random_state=seed,
            max_rows=None,
            max_depth=20,
            min_samples_leaf=50,
        )
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        n_train_fit = int(len(ytr))
        early_stopping_metric = "none"
        n_estimators_budget = 100
        model_params = {
            "n_estimators": 100,
            "max_rows": None,
            "max_depth": 20,
            "min_samples_leaf": 50,
            "equal_row_budget": True,
        }
    elif method == "xgb_global":
        model = XGBLithologyModel(
            n_estimators=180,
            random_state=seed,
            max_rows=None,
            learning_rate=0.05,
            max_depth=8,
        )
        model.fit(Xtr, ytr, X_val=Xva, y_val=yva)
        pred = model.predict(Xte)
        n_train_fit = int(len(ytr))
        early_stopping_metric = "well_heldout_mlogloss"
        n_estimators_budget = 180
        model_params = {
            "n_estimators": 180,
            "max_rows": None,
            "max_depth": 8,
            "equal_row_budget": True,
        }
    elif method == "lgbm_global_eq120k":
        Xtr_s, ytr_s, ctr_s, wtr_s = _subsample_rows(
            Xtr, ytr, ctr, wtr, max_rows=120_000, seed=seed
        )
        Xva_s, yva_s = Xva, yva
        model = LGBMLithologyModel(
            config=LGBMConfig(random_state=seed, n_estimators=180),
            use_confidence=False,
            use_well_balance=False,
            groupdro_rounds=0,
        )
        model.fit(Xtr_s, ytr_s, ctr_s, wtr_s, X_val=Xva_s, y_val=yva_s)
        pred = model.predict(Xte)
        n_train_fit = int(len(ytr_s))
        early_stopping_metric = "well_heldout_multi_logloss"
        n_estimators_budget = 180
        model_params = {"n_estimators": 180, "max_rows": 120_000}
    elif per_well_sim:
        use_conf_pw = method == "lgbm_hybrid_plus_perwell"
        use_smooth_pw = method == "lgbm_hybrid_plus_perwell"
        feat_names = list(tr.feature_names)
        pred = np.empty(len(yte), dtype=np.int32)
        for tw in pd.unique(wte):
            mte = wte == tw
            Xte_w = Xte[mte]
            m_sim = te_sim_wells.astype(str) == str(tw)
            X_tgt = te_sim_X[m_sim] if m_sim.any() else Xte_w
            w_tgt = te_sim_wells[m_sim] if m_sim.any() else wte[mte]
            ext_w = similarity_sample_weights(
                Xtr,
                wtr,
                X_tgt,
                w_tgt,
                feature_names=feat_names,
                temperature=1.0,
                X_source_means=tr_sim_X,
                wells_source_means=tr_sim_wells,
            )
            model = LGBMLithologyModel(
                config=LGBMConfig(random_state=seed, n_estimators=180),
                use_confidence=use_conf_pw,
                use_well_balance=False,
                groupdro_rounds=0,
                external_weights=ext_w,
            )
            model.fit(Xtr, ytr, ctr, wtr, X_val=Xva, y_val=yva)
            pred_w = model.predict(Xte_w)
            if use_smooth_pw:
                pred_w = smooth_predictions_by_well(pred_w, wte[mte], dte[mte], window=7)
            pred[mte] = pred_w
        sim_mode = "per_well"
        n_train_fit = int(len(ytr))
        early_stopping_metric = "well_heldout_multi_logloss"
        n_estimators_budget = 180
        model_params = {"n_estimators": 180, "sim": "per_well"}
    else:
        ext_w = None
        if use_sim:
            ext_w = similarity_sample_weights(
                Xtr,
                wtr,
                te_sim_X,
                te_sim_wells,
                feature_names=list(tr.feature_names),
                temperature=1.0,
                X_source_means=tr_sim_X,
                wells_source_means=tr_sim_wells,
            )
            sim_mode = "batch_target"
        if use_rareboost:
            rare = {74000, 70032, 99000, 90000, 93000, 86000}
            inv = {v: k for k, v in mapping.items()}
            rw = np.ones(len(ytr), dtype=np.float64)
            for i, yi in enumerate(ytr):
                if inv.get(int(yi)) in rare:
                    rw[i] = 3.0
            ext_w = rw if ext_w is None else ext_w * rw
        n_est = 180 if method != "lgbm_groupdro" else 120
        Xfit, yfit, cfit, wfit = Xtr, ytr, ctr, wtr
        if eq120k and method == "lgbm_global":
            pass  # full LGBM
        model = LGBMLithologyModel(
            config=LGBMConfig(random_state=seed, n_estimators=n_est),
            use_confidence=use_conf,
            use_well_balance=use_balance,
            groupdro_rounds=groupdro,
            external_weights=ext_w,
        )
        if groupdro > 0:
            model.fit(Xfit, yfit, cfit, wfit)
            early_stopping_metric = "none"
        else:
            model.fit(Xfit, yfit, cfit, wfit, X_val=Xva, y_val=yva)
            early_stopping_metric = "well_heldout_multi_logloss"
        pred = model.predict(Xte)
        if use_smooth:
            # Score-support only: smooth on labelled test depths (not full-curve MD grid).
            pred = smooth_predictions_by_well(pred, wte, dte, window=7)
            smoothing_scope = "labeled_score_support"
        n_train_fit = int(len(yfit))
        n_estimators_budget = int(n_est)
        model_params = {"n_estimators": n_est}

    label_names = {mapping[k]: LITHOLOGY_MAP.get(k, str(k)) for k in mapping}
    if n_unseen_te > 0:
        label_names[unk_id] = "UNSEEN_IN_TRAIN"
    # Pooled Macro-F1: classes present in this test y_true (may include unk).
    pooled_labels = sorted(np.unique(yte).tolist())
    # Fixed per-well Macro-F1: train-learnable classes only (shared across wells;
    # excludes never-predicted unk). Matches scripts/revision_round3_fixed_taxonomy.py.
    fixed_well_labels = sorted(mapping.values())
    ev = evaluate_split(
        yte,
        pred,
        wte,
        label_names=label_names,
        pooled_labels=pooled_labels,
        fixed_well_labels=fixed_well_labels,
    )
    well_table = ev["well_table"]
    well_table_fixed = ev["well_table_fixed"]
    # sklearn report can include aggregate keys; never treat them as facies.
    _SKIP_REPORT_KEYS = {
        "accuracy",
        "macro avg",
        "weighted avg",
        "micro avg",
        "samples avg",
    }
    per_class = {}
    report = ev.get("classification_report") or {}
    for key, val in report.items():
        if key in _SKIP_REPORT_KEYS:
            continue
        if isinstance(val, dict) and "f1-score" in val:
            name = label_names.get(int(key), str(key)) if str(key).isdigit() else str(key)
            if name in _SKIP_REPORT_KEYS:
                continue
            per_class[name] = {
                "f1": float(val["f1-score"]),
                "precision": float(val["precision"]),
                "recall": float(val["recall"]),
                "support": int(val["support"]),
            }

    known_diag: dict[str, Any] = {}
    if n_unseen_te > 0:
        kn = evaluate_known_class_only(
            yte, pred, wte, unk_id=unk_id, label_names=label_names
        )
        if kn is not None:
            known_diag = {
                "known_macro_f1": kn["overall"]["macro_f1"],
                "known_worst_well_f1": kn["well_summary"]["worst_well_f1"],
                "known_worst10pct_well_f1": kn["well_summary"]["worst10pct_well_f1"],
                "known_n_test": int(len(yte) - n_unseen_te),
            }
            # Wells that contain any unseen labels
            unseen_wells = sorted(
                {str(w) for w, y in zip(wte, yte) if int(y) == unk_id}
            )
            known_diag["unseen_wells"] = unseen_wells
            known_diag["n_unseen_wells"] = len(unseen_wells)

    # Optional val-open diagnostic: score ALL val rows (unseen → unk), without
    # changing early-stopping (which still uses known-class-only yva).
    # Skipped for per-well similarity (no single model) — report-only elsewhere.
    val_open_diag: dict[str, Any] = {}
    max_proba_diag: dict[str, Any] = {}
    if per_well_sim:
        val_open_diag = {"skipped": "per_well_sim"}
        max_proba_diag = {"skipped": "per_well_sim"}
    else:
        try:
            yva_open = np.array(
                [mapping[int(v)] if int(v) in mapping else unk_id for v in va.y],
                dtype=np.int32,
            )
            n_val_unseen_kept = int((yva_open == unk_id).sum())
            pred_va_open = model.predict(va.X)
            if use_smooth:
                from .postprocess import smooth_predictions_by_well as _smooth_va

                pred_va_open = _smooth_va(pred_va_open, va.wells, va.depths, window=7)
            pooled_va = sorted(np.unique(yva_open).tolist())
            ev_va = evaluate_split(
                yva_open,
                pred_va_open,
                va.wells,
                label_names=label_names,
                pooled_labels=pooled_va,
                fixed_well_labels=fixed_well_labels,
            )
            val_open_diag = {
                "n_val_total": int(len(yva_open)),
                "n_val_unseen_label_kept": n_val_unseen_kept,
                "n_val_dropped_unseen_label": int(n_drop_va),
                "macro_f1_including_unseen": float(ev_va["overall"]["macro_f1"]),
                "worst_well_f1_fixed": float(
                    (ev_va.get("well_summary_fixed") or {}).get("worst_well_f1", 0.0)
                ),
                "note": "report-only; early stopping still uses known-class val rows",
            }
            if n_val_unseen_kept > 0:
                kn_va = evaluate_known_class_only(
                    yva_open,
                    pred_va_open,
                    va.wells,
                    unk_id=unk_id,
                    label_names=label_names,
                )
                if kn_va is not None:
                    val_open_diag["known_macro_f1"] = float(kn_va["overall"]["macro_f1"])
        except Exception as exc:  # noqa: BLE001 — diagnostic must never fail the run
            val_open_diag = {"error": str(exc)}

        # Descriptive max-proba histogram / coverage–risk buckets (test set).
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(Xte)
                max_p = np.max(np.asarray(proba, dtype=np.float64), axis=1)
                edges = [0.0, 0.5, 0.7, 0.9, 1.0000001]
                hist, _ = np.histogram(max_p, bins=edges)
                max_proba_diag = {
                    "n": int(len(max_p)),
                    "mean": float(max_p.mean()),
                    "median": float(np.median(max_p)),
                    "bins": ["[0,0.5)", "[0.5,0.7)", "[0.7,0.9)", "[0.9,1]"],
                    "counts": [int(x) for x in hist],
                    "fracs": [float(x) / max(1, len(max_p)) for x in hist],
                }
        except Exception as exc:  # noqa: BLE001
            max_proba_diag = {"error": str(exc)}

    # A priori closed/open bucket for this split (test unseen depths).
    label_regime = "closed_set" if int(n_unseen_te) == 0 else "open_set"

    prov = collect_run_provenance(
        split=_ACTIVE_SPLIT,
        config_path=_CONFIG_PATH,
        feature_names=list(tr.feature_names),
        feature_mode=feat_mode,
        use_relative_depth=bool(use_rel_depth),
        well_z_mode=well_z_mode,
        model_params=model_params,
    )

    result = {
        "method": method,
        "seed": seed,
        "pipeline_version": PIPELINE_VERSION,
        "feature_mode": feat_mode,
        "use_relative_depth": bool(use_rel_depth),
        "well_z_mode": well_z_mode,
        "stats_support": "full_curve" if _CURVE_CONTEXT is not None else "labeled_only",
        # Eligible labelled train rows after label filter (pre-subsample).
        "n_train": int(n_keep_tr),
        "n_train_eligible": int(n_keep_tr),
        # Rows actually used in fit after schedule subsample (MLP/RF/eq120k).
        "n_train_fit": int(n_train_fit),
        "n_val": int(n_keep_va),
        "n_test": int(n_keep_te),
        "n_train_dropped_unseen_label": int(n_drop_tr),
        "n_val_dropped_unseen_label": int(n_drop_va),
        "n_val_dropped_unseen": int(n_drop_va),  # clear alias for summaries
        "n_test_unseen_label_kept": int(n_unseen_te),
        "label_regime": label_regime,
        "n_features": int(Xtr.shape[1]),
        "seconds": round(time.time() - t0, 2),
        "overall": ev["overall"],
        "well_summary": ev["well_summary"],
        "well_table": well_table.to_dict(orient="records"),
        "well_table_fixed": well_table_fixed.to_dict(orient="records"),
        "label_mapping": {str(k): v for k, v in mapping.items()},
        "train_class_names": [
            LITHOLOGY_MAP.get(int(k), str(k)) for k in sorted(mapping.keys())
        ],
        "per_class": per_class,
        "uda_target_mode": sim_mode,
        "smoothing_scope": smoothing_scope,
        "early_stopping_metric": early_stopping_metric,
        # Tree schedules only; MLP omits (None → blank in CSV).
        "n_estimators_budget": n_estimators_budget,
        "known_class_diagnostics": known_diag,
        "val_open_diagnostics": val_open_diag,
        "max_proba_diagnostics": max_proba_diag,
        "well_summary_fixed": ev.get("well_summary_fixed", {}),
        "fixed_label_mode": "train_learnable",
        "n_fixed_labels": int(len(fixed_well_labels)),
        "fixed_well_labels": list(fixed_well_labels),
        **prov,
    }
    return result


def run_protocol_experiment(
    df: pd.DataFrame,
    split: dict[str, Any],
    methods: list[str],
    out_dir: Path | None = None,
    *,
    curves_all: pd.DataFrame | None = None,
    config_path: Path | None = None,
) -> pd.DataFrame:
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if curves_all is not None:
        set_curve_context(curves_all)
    set_run_meta(split=split, config_path=config_path)
    parts = apply_split(df, split)
    tag = split.get("protocol", "split")
    seed = split.get("seed", 0)
    extra = ""
    if split.get("holdout_blocks"):
        extra = "_blk" + "-".join(split["holdout_blocks"])
    if split.get("target_well"):
        sel = split.get("selection") or "spaced"
        prefix = "" if sel == "spaced" else f"_{sel}"
        extra = prefix + "_well" + str(split["target_well"]).replace("/", "_").replace(" ", "")
    stem = f"{tag}{extra}_seed{seed}"
    summary_path = out_dir / f"summary_{stem}.csv"
    details_path = out_dir / f"details_{stem}.json"

    rows: list[dict] = []
    details: list[dict] = []
    done_methods: set[str] = set()
    expected_train = len(parts["train"])
    expected_test = len(parts["test"])
    # Shared provenance fingerprint for this split/config/data (not per-method).
    expected_prov = collect_run_provenance(split=split, config_path=config_path)
    expected_shared = {
        "pipeline_version": PIPELINE_VERSION,
        "config_hash": str(expected_prov.get("config_hash", "")),
        "split_hash": str(expected_prov.get("split_hash", "")),
        "hash_curves_all_parquet": str(expected_prov.get("hash_curves_all_parquet", "")),
        "hash_labeled_parquet": str(expected_prov.get("hash_labeled_parquet", "")),
        "hash_dataset_meta_json": str(expected_prov.get("hash_dataset_meta_json", "")),
        "n_train": expected_train,
        "n_test": expected_test,
    }

    def _row_matches_shared(row: dict[str, Any]) -> bool:
        for key, exp in expected_shared.items():
            if key in {"n_train", "n_test"}:
                try:
                    if int(row.get(key, -1)) != int(exp):
                        return False
                except (TypeError, ValueError):
                    return False
            else:
                got = str(row.get(key, "") or "")
                if not got or got != str(exp):
                    return False
        # Require non-empty feature_param_hash so pre-provenance rows never resume.
        if not str(row.get("feature_param_hash", "") or ""):
            return False
        return True

    if summary_path.exists():
        prev = pd.read_csv(summary_path)
        if not prev.empty and "method" in prev.columns:
            keep_mask = prev.apply(lambda r: _row_matches_shared(r.to_dict()), axis=1)
            n_drop = int((~keep_mask).sum())
            if n_drop:
                print(
                    f"[resume] drop {n_drop}/{len(prev)} stale row(s) in {summary_path.name} "
                    f"(pipeline/config/split/data hash or n_train/n_test mismatch)"
                )
            prev = prev.loc[keep_mask].copy()
            if prev.empty:
                print(
                    f"[resume] ignore {summary_path.name}: no rows match current provenance "
                    f"(config_hash={expected_shared['config_hash'][:12]}…, "
                    f"split_hash={expected_shared['split_hash'][:12]}…, "
                    f"ver={PIPELINE_VERSION})"
                )
            else:
                done_methods = set(prev["method"].astype(str))
                rows = prev.to_dict(orient="records")
                if details_path.exists():
                    try:
                        details = json.loads(details_path.read_text(encoding="utf-8"))
                        if not isinstance(details, list):
                            details = []
                        details = [
                            d
                            for d in details
                            if isinstance(d, dict) and _row_matches_shared(d)
                        ]
                    except json.JSONDecodeError:
                        details = []
        else:
            done_methods = set()
            rows = []
            details = []
    elif details_path.exists():
        try:
            details = json.loads(details_path.read_text(encoding="utf-8"))
            if not isinstance(details, list):
                details = []
            details = [
                d for d in details if isinstance(d, dict) and _row_matches_shared(d)
            ]
        except json.JSONDecodeError:
            details = []

    for m in methods:
        if m in done_methods:
            print(f"[skip] {split.get('protocol')} | method={m} (provenance-matched in {summary_path.name})")
            continue
        print(f"[run] {split.get('protocol')} | method={m}")
        res = run_method_on_split(parts, m, seed=int(split.get("seed", 42)))
        details.append(res)
        row = {
            "protocol": split.get("protocol"),
            "seed": split.get("seed"),
            "holdout_blocks": ",".join(split.get("holdout_blocks", [])),
            # Compatible alias: Protocol-2 values are NPD quadrant ids (legacy column name retained).
            "holdout_quadrants": ",".join(split.get("holdout_blocks", [])),
            "target_well": split.get("target_well", ""),
            "method": m,
            **res["overall"],
            **res["well_summary"],
            "seconds": res["seconds"],
            "n_train": res["n_train"],
            "n_train_eligible": res.get("n_train_eligible", res["n_train"]),
            "n_train_fit": res.get("n_train_fit", res["n_train"]),
            "n_test": res["n_test"],
            "n_test_unseen_label_kept": res.get("n_test_unseen_label_kept", 0),
            "n_val_dropped_unseen_label": res.get("n_val_dropped_unseen_label", 0),
            "n_val_dropped_unseen": res.get(
                "n_val_dropped_unseen", res.get("n_val_dropped_unseen_label", 0)
            ),
            "label_regime": res.get("label_regime", ""),
            "uda_target_mode": res.get("uda_target_mode", "none"),
            "early_stopping_metric": res.get("early_stopping_metric", ""),
            "n_estimators_budget": (
                "" if res.get("n_estimators_budget") is None else res.get("n_estimators_budget", "")
            ),
            "pipeline_version": res.get("pipeline_version", PIPELINE_VERSION),
            "feature_mode": res.get("feature_mode", ""),
            "use_relative_depth": res.get("use_relative_depth", ""),
            "well_z_mode": res.get("well_z_mode", ""),
            "stats_support": res.get("stats_support", ""),
            "n_features": res.get("n_features", ""),
            "git_commit": res.get("git_commit", ""),
            "git_dirty": res.get("git_dirty", ""),
            "config_hash": res.get("config_hash", ""),
            "split_hash": res.get("split_hash", ""),
            "hash_curves_all_parquet": res.get("hash_curves_all_parquet", ""),
            "hash_labeled_parquet": res.get("hash_labeled_parquet", ""),
            "hash_dataset_meta_json": res.get("hash_dataset_meta_json", ""),
            "feature_param_hash": res.get("feature_param_hash", ""),
            "fixed_label_mode": res.get("fixed_label_mode", ""),
            "n_fixed_labels": res.get("n_fixed_labels", ""),
            "known_macro_f1": (res.get("known_class_diagnostics") or {}).get("known_macro_f1", ""),
            "known_worst_well_f1": (res.get("known_class_diagnostics") or {}).get(
                "known_worst_well_f1", ""
            ),
            "known_worst10pct_well_f1": (res.get("known_class_diagnostics") or {}).get(
                "known_worst10pct_well_f1", ""
            ),
            "worst_well_f1_fixed": (res.get("well_summary_fixed") or {}).get(
                "worst_well_f1", ""
            ),
            "worst10pct_well_f1_fixed": (res.get("well_summary_fixed") or {}).get(
                "worst10pct_well_f1", ""
            ),
            "smoothing_scope": res.get("smoothing_scope", "none"),
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        details_path.write_text(
            json.dumps(details, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    return pd.DataFrame(rows)


def run_naive_leakage_baseline(
    df: pd.DataFrame, seed: int = 42, *, config_path: Path | None = None
) -> dict[str, Any]:
    """Negative control: random depth split with the same Global-LGBM family as Protocol-1."""
    parts = naive_depth_split(df, seed=seed)
    split_meta = {
        k: parts[k]
        for k in (
            "protocol",
            "seed",
            "test_frac",
            "val_frac",
            "row_index_hash",
        )
        if k in parts
    }
    set_run_meta(split=split_meta, config_path=config_path)
    train_parts = {"train": parts["train"], "val": parts["val"], "test": parts["test"]}
    res = run_method_on_split(train_parts, "lgbm_global", seed=seed)
    res["method"] = "naive_depth_split_lgbm"
    res["protocol"] = "naive_depth_split"
    res["row_index_hash"] = parts.get("row_index_hash", "")
    res["test_frac"] = parts.get("test_frac", "")
    res["val_frac"] = parts.get("val_frac", "")
    return res
