"""Boundary controls must use original physical depth precision."""
import numpy as np
import pandas as pd
from src.features import FeaturePipeline
from scripts.v3_depth_validation import boundary_coordinates


def test_relative_depth_reconstruction_uses_original_dataframe_depths():
    df = pd.DataFrame({'well_id': ['A'] * 4,
        'depth': [2000.1234567, 2001.2765432, 2100.3141592, 2300.9876543],
        'GR': [10., 20., 30., 40.], 'lithology': [30000, 30000, 65000, 65000]})
    p = FeaturePipeline(mode='hybrid', curves=['GR'], curve_context=df,
                         raw_missingness_mask=True, use_relative_depth=True)
    b = p.fit_transform(df)
    lo, hi = np.repeat(df.depth.min(), 4), np.repeat(df.depth.max(), 4)
    _, rel_original = boundary_coordinates(df.depth.to_numpy(), lo, hi, 0., 0.)
    _, rel_exported = boundary_coordinates(b.depths, lo, hi, 0., 0.)
    np.testing.assert_array_equal(rel_original, b.X[:, b.feature_names.index('rel_depth')])
    assert not np.array_equal(rel_original, rel_exported)
