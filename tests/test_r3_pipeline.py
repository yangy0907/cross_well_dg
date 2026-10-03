"""Unit tests for r3 full-curve stats, open-set mapping, smoothing, splits."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import CONF_COL, DEPTH_COL, LABEL_COL, WELL_COL
from src.features import FeaturePipeline
from src.metrics import evaluate_known_class_only, evaluate_split
from src.postprocess import _segment_indices, smooth_predictions_by_well
from src.splits import protocol1_random_wells
from src.train import PIPELINE_VERSION, _feature_routing


def _toy_well(well: str, depths, curves, labels) -> pd.DataFrame:
    n = len(depths)
    return pd.DataFrame(
        {
            WELL_COL: [well] * n,
            DEPTH_COL: depths,
            "GR": curves,
            "RHOB": np.asarray(curves, dtype=float) * 0.01 + 2.0,
            "NPHI": np.linspace(0.1, 0.3, n),
            "DTC": np.linspace(80, 100, n),
            LABEL_COL: labels,
            CONF_COL: [1.0] * n,
        }
    )


def test_pipeline_version_is_r3():
    assert PIPELINE_VERSION.startswith("r3_")


def test_release_config_includes_equal_row_and_wellnorm_norel():
    import yaml

    cfg = yaml.safe_load(
        (ROOT / "configs" / "paper_r3_release.yaml").read_text(encoding="utf-8")
    )
    methods = set(cfg["methods"])
    assert "lgbm_wellnorm_norel" in methods
    assert "rf_global_full" in methods
    assert "xgb_global" in methods
    assert "lgbm_hybrid" in methods


def test_multirep_config_matches_locked_constants():
    import yaml

    from src.constants import PROTOCOL1_CORE_METHODS, PROTOCOL1_MULTIREP_SEEDS

    cfg = yaml.safe_load(
        (ROOT / "configs" / "paper_r3_multirep.yaml").read_text(encoding="utf-8")
    )
    assert list(cfg["seeds"]) == list(PROTOCOL1_MULTIREP_SEEDS)
    assert list(cfg["methods"]) == list(PROTOCOL1_CORE_METHODS)
    assert "well_id" in cfg["protocol1"]["split_modes"]
    assert "parent_grouped" in cfg["protocol1"]["split_modes"]
    assert cfg.get("closed_set_rule")


def test_git_provenance_freeze_ignores_results(tmp_path, monkeypatch):
    from src import provenance as prov

    prov.clear_frozen_git_provenance()
    snap = prov.freeze_git_provenance(ignore_outputs=True)
    assert "git_commit" in snap
    assert "git_dirty" in snap
    # Reuse frozen snapshot even if we ask again
    again = prov.git_provenance(use_frozen=True)
    assert again["git_commit"] == snap["git_commit"]
    assert again["git_dirty"] == snap["git_dirty"]
    prov.clear_frozen_git_provenance()


def test_naive_depth_split_exposes_row_index_hash():
    from src.splits import naive_depth_split

    df = pd.DataFrame(
        {
            "well_id": ["W"] * 20,
            "depth": list(range(20)),
            "lithology": [65000] * 20,
        }
    )
    parts = naive_depth_split(df, seed=42)
    assert "train" in parts and "val" in parts and "test" in parts
    assert parts["protocol"] == "naive_depth_split"
    assert isinstance(parts["row_index_hash"], str) and len(parts["row_index_hash"]) == 64
    # Deterministic
    parts2 = naive_depth_split(df, seed=42)
    assert parts["row_index_hash"] == parts2["row_index_hash"]


def test_full_curve_well_z_differs_from_labeled_only():
    depths = np.arange(0.0, 100.0, 1.0)
    gr = np.where(depths < 70, 20.0, 120.0)
    labels = np.where(depths < 70, np.nan, 65000)
    full = _toy_well("W1", depths, gr, labels)
    labeled = full.dropna(subset=[LABEL_COL]).copy()
    labeled[LABEL_COL] = labeled[LABEL_COL].astype(int)

    curves = ["GR", "RHOB", "NPHI", "DTC"]
    pipe_full = FeaturePipeline(
        mode="wellnorm", use_relative_depth=True, well_z_mode="full", curves=curves
    )
    z_full = pipe_full.fit_transform(labeled, curve_context=full)

    pipe_lab = FeaturePipeline(
        mode="wellnorm", use_relative_depth=True, well_z_mode="full", curves=curves
    )
    z_lab = pipe_lab.fit_transform(labeled, curve_context=None)

    assert "GR" in z_full.feature_names
    i = z_full.feature_names.index("GR")
    assert not np.allclose(z_full.X[:, i], z_lab.X[:, i], atol=1e-5)


def test_causal_no_future_leakage():
    depths = np.array([10.0, 20.0, 30.0, 40.0])
    gr = np.array([10.0, 10.0, 100.0, 100.0])
    labels = [65000, 65000, 65000, 65000]
    df = _toy_well("W1", depths, gr, labels)
    ctx = df.copy()
    extra = _toy_well("W1", [50.0], [1000.0], [np.nan])
    ctx = pd.concat([ctx, extra], ignore_index=True)

    curves = ["GR", "RHOB", "NPHI", "DTC"]
    pipe = FeaturePipeline(
        mode="hybrid", use_relative_depth=False, well_z_mode="causal", curves=curves
    )
    bun = pipe.fit_transform(df, curve_context=ctx)
    i = bun.feature_names.index("GR_wn")
    assert abs(bun.X[0, i]) < 1e-6
    assert abs(bun.X[1, i]) < 1e-5


def test_smooth_does_not_cross_depth_gap():
    pred = np.array([0, 0, 0, 1, 1, 1])
    wells = np.array(["W"] * 6)
    # Two clusters separated by a large gap
    depths = np.array([100.0, 100.5, 101.0, 200.0, 200.5, 201.0])
    out = smooth_predictions_by_well(pred, wells, depths, window=5, gap_factor=1.5)
    # Left cluster stays 0; right stays 1 (no cross-gap majority of the other class)
    assert set(out[:3].tolist()) == {0}
    assert set(out[3:].tolist()) == {1}


def test_segment_indices_breaks_on_gap():
    d = np.array([1.0, 2.0, 3.0, 20.0, 21.0])
    segs = _segment_indices(d, gap_factor=1.5)
    assert len(segs) == 2
    assert list(segs[0]) == [0, 1, 2]
    assert list(segs[1]) == [3, 4]


def test_open_set_known_class_only():
    y_true = np.array([0, 0, 1, 2, 2])  # 2 = unk
    y_pred = np.array([0, 1, 1, 0, 1])
    wells = np.array(["A", "A", "B", "B", "B"])
    kn = evaluate_known_class_only(y_true, y_pred, wells, unk_id=2)
    assert kn is not None
    assert kn["overall"]["macro_f1"] >= 0.0
    # Only known rows scored: 3 rows
    assert kn["well_table"]["n"].sum() == 3


def test_fixed_well_labels_use_train_learnable_not_pooled():
    """fixed_well_labels may omit unk even when pooled_labels include it."""
    y_true = np.array([0, 0, 1, 2])  # 2 = unk present in test
    y_pred = np.array([0, 0, 1, 0])
    wells = np.array(["A", "A", "B", "B"])
    ev = evaluate_split(
        y_true,
        y_pred,
        wells,
        pooled_labels=[0, 1, 2],
        fixed_well_labels=[0, 1],
    )
    assert ev["fixed_well_labels"] == [0, 1]
    assert ev["pooled_labels"] == [0, 1, 2]
    assert "worst_well_f1" in ev["well_summary_fixed"]


def test_protocol1_split_disjoint(tmp_path, monkeypatch):
    from src import splits as splits_mod

    monkeypatch.setattr(splits_mod, "SPLITS_DIR", tmp_path)
    wells = [f"W{i}" for i in range(40)]
    rows = []
    for w in wells:
        rows.append(
            {
                WELL_COL: w,
                DEPTH_COL: 100.0,
                LABEL_COL: 65000,
                CONF_COL: 1.0,
                "block": "15",
                "GR": 50.0,
            }
        )
    df = pd.DataFrame(rows)
    split = protocol1_random_wells(
        df, seed=42, train_frac=0.7, val_frac=0.15, save=False
    )
    tr, va, te = set(split["train_wells"]), set(split["val_wells"]), set(split["test_wells"])
    assert not (tr & va)
    assert not (va & te)
    assert not (tr & te)
    assert tr | va | te == set(wells)
    # save=False must not write protocol1_seed42.json even if SPLITS_DIR is redirected
    assert not (tmp_path / "protocol1_seed42.json").exists()


def test_hybrid_while_routing():
    mode, rel, wz = _feature_routing("lgbm_hybrid_while")
    assert mode == "hybrid" and rel is False and wz == "causal"


def test_wellnorm_norel_routing():
    mode, rel, wz = _feature_routing("lgbm_wellnorm_norel")
    assert mode == "wellnorm" and rel is False and wz == "full"
    mode2, rel2, wz2 = _feature_routing("lgbm_wellnorm")
    assert mode2 == "wellnorm" and rel2 is True and wz2 == "full"
    mode3, rel3, _ = _feature_routing("lgbm_hybrid_norel")
    assert mode3 == "hybrid" and rel3 is False


def test_causal_z_lookup_handles_descending_queries():
    """Descending query depths must not leak deeper prefix stats into shallower rows."""
    pipe = FeaturePipeline(
        mode="hybrid", use_relative_depth=False, well_z_mode="causal", curves=["GR"]
    )
    depths_full = np.array([10.0, 20.0, 30.0, 40.0])
    values = np.array([10.0, 10.0, 100.0, 100.0])
    # Ascending reference
    z_asc = pipe._causal_z_lookup(
        np.array([10.0, 20.0, 30.0, 40.0]), depths_full, values
    )
    # Same depths, descending order — values must permute accordingly
    z_desc = pipe._causal_z_lookup(
        np.array([40.0, 30.0, 20.0, 10.0]), depths_full, values
    )
    np.testing.assert_allclose(z_desc[::-1], z_asc, atol=1e-9)
    # Shallow query after a deep query must still use only the shallow prefix
    z_mixed = pipe._causal_z_lookup(
        np.array([40.0, 10.0]), depths_full, values
    )
    assert abs(z_mixed[1]) < 1e-6


def test_protocol1_closed_seeds_exclude_openset_44():
    from src.constants import (
        PROTOCOL1_ALL_SEEDS,
        PROTOCOL1_CLOSED_SEEDS,
        PROTOCOL1_OPENSET_SEEDS,
    )

    assert PROTOCOL1_CLOSED_SEEDS == (42, 43, 45)
    assert PROTOCOL1_OPENSET_SEEDS == (44,)
    assert 44 not in PROTOCOL1_CLOSED_SEEDS
    assert set(PROTOCOL1_CLOSED_SEEDS) | set(PROTOCOL1_OPENSET_SEEDS) == set(
        PROTOCOL1_ALL_SEEDS
    )


def test_fig_multiseed_primary_aggregation_excludes_seed_44(tmp_path, monkeypatch):
    """Primary multi-seed table/figure must not pool open-set seed 44."""
    import scripts.make_paper_figures as mpf

    fig_tmp = tmp_path / "paper_figures"
    gse_tmp = tmp_path / "gse_figs"
    fig_tmp.mkdir()
    gse_tmp.mkdir()
    monkeypatch.setattr(mpf, "PAPER_DIR", fig_tmp)
    monkeypatch.setattr(mpf, "FIGURES_DIR", fig_tmp)
    # Never write production paper/gse/figs from unit tests.
    monkeypatch.setattr(mpf, "PAPER_GSE_FIGS", None)
    monkeypatch.setattr(mpf, "ROOT", ROOT)

    rows = []
    for seed in (42, 43, 44, 45):
        for method, base in [("lgbm_global", 0.40), ("lgbm_hybrid", 0.42)]:
            # Seed 44 deliberately lower so pooling would pull the mean down.
            macro = base - 0.10 if seed == 44 else base
            rows.append(
                {
                    "protocol": "protocol1_random_wells",
                    "seed": seed,
                    "method": method,
                    "macro_f1": macro,
                    "worst_well_f1": 0.2,
                    "worst_well_f1_fixed": 0.1,
                }
            )
    df = pd.DataFrame(rows)
    mpf.fig_multiseed(df)
    out = fig_tmp / "table_protocol1_multiseed.csv"
    assert out.exists()
    # Production gse tree must remain untouched by this test.
    prod_gse = ROOT / "paper" / "gse" / "figs" / "table_protocol1_multiseed.csv"
    assert mpf.PAPER_GSE_FIGS is None
    g = pd.read_csv(out)
    assert not g["seeds"].astype(str).str.contains("44").any()
    assert (g["n_seeds"] == 3).all()
    # Closed-set mean for lgbm_global must be 0.40, not (0.40*3+0.30)/4
    g_global = g.set_index("method").loc["lgbm_global"]
    assert abs(float(g_global["macro_mean"]) - 0.40) < 1e-9
    # Primary worst column prefers fixed taxonomy when present.
    assert abs(float(g_global["worst_mean"]) - 0.1) < 1e-9
    assert abs(float(g_global["worst_true_mean"]) - 0.2) < 1e-9
    # Sanity: production path was not selected as the write target.
    del prod_gse


def test_blank_mask_detects_nan_split_hash():
    """Gate must treat real NaN as blank (pandas 3 astype(str) alone does not)."""
    from scripts.check_active_results_r3 import _blank_mask

    s = pd.Series(["abc", np.nan, "", "nan", None])
    bad = _blank_mask(s)
    assert bad.tolist() == [False, True, True, True, True]


def test_acceptance_requires_complete_multirep_summary(tmp_path, monkeypatch):
    import scripts.build_acceptance_record as bar

    monkeypatch.setattr(bar, "OUT", tmp_path)
    monkeypatch.setattr(bar, "MULTIREP", tmp_path / "multirep")
    monkeypatch.setattr(bar, "RESULTS_DIR", tmp_path)

    # Incomplete closed summary (wrong n_closed_reps) → hard fail
    rows = []
    for method in bar.PRIMARY_METHODS:
        rows.append(
            {
                "split_mode": "well_id",
                "method": method,
                "method_label": method,
                "n_closed_reps": 17,  # must be 18
                "macro_legacy_mean": 0.3,
                "macro_legacy_std": 0.01,
                "worst_fixed_mean": 0.1,
                "worst_fixed_std": 0.01,
                "w10_fixed_mean": 0.11,
                "w10_fixed_std": 0.01,
            }
        )
    pd.DataFrame(rows).to_csv(
        tmp_path / "protocol1_multirep_well_id_closed_summary.csv", index=False
    )
    assert bar.main() == 1


def test_protocol3_rejects_stale_cache(tmp_path, monkeypatch):
    from src import splits as sp

    monkeypatch.setattr(sp, "SPLITS_DIR", tmp_path)
    df = pd.DataFrame(
        {
            WELL_COL: ["15/9-1", "15/9-2", "15/9-3", "17/1-1"] * 5,
            "depth": list(range(20)),
            "block": ["15", "15", "15", "17"] * 5,
            LABEL_COL: [65000] * 20,
        }
    )
    # First call writes cache
    a = sp.protocol3_leave_one_well_out(df, max_targets=2, seed=42, selection="spaced", save=True)
    assert a and "wells_hash" in a[0]
    # Corrupt wells_hash → must raise, not silent reuse
    path = next(tmp_path.glob("protocol3_*.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["wells_hash"] = "deadbeef"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale or incomplete"):
        sp.protocol3_leave_one_well_out(df, max_targets=2, seed=42, selection="spaced", save=True)
