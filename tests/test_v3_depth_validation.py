"""Protect physical scoring support and isolation of the relative-depth stress."""
import numpy as np
import pandas as pd
import pytest
from src.features import FeaturePipeline
from scripts.v3_depth_validation import boundary_coordinates, change_relative_column
from src.models import LGBMConfig, LGBMLithologyModel
import lightgbm as lgb


def test_crop_uses_each_wells_physical_interval_and_common_support():
    # Two differently positioned and scaled wells; 20% top crop has known metres.
    depth = np.array([100., 120., 150., 200., 1000., 1200., 1500., 2000.])
    lo = np.array([100.] * 4 + [1000.] * 4)
    hi = np.array([200.] * 4 + [2000.] * 4)
    keep, rel = boundary_coordinates(depth, lo, hi, .2, 0.)
    np.testing.assert_array_equal(keep, [False, True, True, True, False, True, True, True])
    np.testing.assert_allclose(rel[keep], [0., .375, 1., 0., .375, 1.], atol=2e-8)


def test_baseline_coordinates_match_full_context_not_labelled_support():
    context = pd.DataFrame({'well_id': ['A'] * 5, 'depth': [100., 125., 150., 175., 200.],
                            'GR': [10., 20., 30., 40., 50.]})
    labelled = context.iloc[[1, 3]].copy()
    labelled['lithology'] = [30000, 65000]
    pipe = FeaturePipeline(mode='hybrid', curves=['GR'], raw_missingness_mask=True,
                           use_relative_depth=True, curve_context=context)
    b = pipe.fit_transform(labelled)
    keep, rel = boundary_coordinates(b.depths, [100.] * 2, [200.] * 2, 0., 0.)
    assert keep.all()
    np.testing.assert_array_equal(rel, b.X[:, b.feature_names.index('rel_depth')])
    np.testing.assert_allclose(rel, [.25, .75], atol=1e-8)
    # Target labels affect scoring only, never these feature coordinates.
    labelled['lithology'] = [93000, 93000]
    np.testing.assert_array_equal(pipe.transform(labelled).X, b.X)


def test_boundary_stress_changes_only_relative_coordinate_and_keeps_original():
    original = np.array([[10., .25, 0., -1.], [20., .75, 1., 2.]], dtype=np.float32)
    saved = original.copy()
    changed = change_relative_column(original, ['GR_raw', 'rel_depth', 'GR_isna', 'GR_wn'], [0., 1.])
    np.testing.assert_array_equal(original, saved)
    np.testing.assert_array_equal(changed[:, [0, 2, 3]], original[:, [0, 2, 3]])
    np.testing.assert_array_equal(changed[:, 1], [0., 1.])


@pytest.mark.parametrize('top,bottom', [(-.1, 0.), (0., -.1), (.5, .5), (.8, .3)])
def test_impossible_intervals_are_rejected(top, bottom):
    with pytest.raises(ValueError):
        boundary_coordinates([50.], [0.], [100.], top, bottom)


def test_trained_model_roundtrips_through_unicode_path(tmp_path):
    X = np.arange(40, dtype=np.float32).reshape(20, 2)
    y = np.array([0] * 10 + [1] * 10)
    model = LGBMLithologyModel(LGBMConfig(n_estimators=3, num_leaves=3,
                                        min_child_samples=1, n_jobs=1))
    model.fit(X, y, np.ones(20), np.array(['A'] * 20))
    path = tmp_path / '测井模型.txt'
    path.write_text(model.model.booster_.model_to_string(), encoding='utf-8')
    reloaded = lgb.Booster(model_str=path.read_text(encoding='utf-8'))
    np.testing.assert_allclose(reloaded.predict(X), model.predict_proba(X)[:, 1], atol=1e-12)
