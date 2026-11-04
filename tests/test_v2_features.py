"""Protect v1 behavior and validate v2 input fairness under actual missing values."""
import numpy as np
import pandas as pd
import pytest
from src.features import FeaturePipeline

def toy():
    return pd.DataFrame({'well_id':['A']*4, 'depth':[1.,2.,3.,4.],
        'lithology':[65000,65000,30000,30000], 'confidence':[1.]*4,
        'GR':[20.,np.nan,80.,120.], 'RHOB':[2.,2.2,np.nan,2.5],
        'NPHI':[.1,.2,.3,np.nan], 'DTC':[80.,90.,100.,110.]})

@pytest.mark.parametrize('relative',[False,True])
def test_v2_wellnorm_preserves_raw_missingness(relative):
    df=toy()
    p=FeaturePipeline(mode='wellnorm',curves=['GR','RHOB','NPHI','DTC'],
        use_relative_depth=relative,raw_missingness_mask=True,curve_context=df)
    b=p.fit_transform(df)
    for c in p.curves:
        np.testing.assert_array_equal(b.X[:,b.feature_names.index(c+'_isna')], df[c].isna().astype(float))
    # Missingness at inference belongs to the requested rows, not training/context labels.
    test=df.copy(); test['GR']=[np.nan,50.,60.,70.]
    bt=p.transform(test)
    np.testing.assert_array_equal(bt.X[:,bt.feature_names.index('GR_isna')], [1.,0.,0.,0.])

@pytest.mark.parametrize('mode,relative',[('global',False),('hybrid',False),('hybrid',True)])
def test_unaffected_models_have_identical_legacy_and_v2_features(mode,relative):
    df=toy()
    kw=dict(mode=mode,use_relative_depth=relative,curves=['GR','RHOB','NPHI','DTC'],curve_context=df)
    old=FeaturePipeline(**kw); new=FeaturePipeline(**kw,raw_missingness_mask=True)
    a=old.fit_transform(df); b=new.fit_transform(df)
    assert a.feature_names==b.feature_names
    np.testing.assert_array_equal(a.X,b.X)
    changed=df.copy();changed.loc[0,'GR']=np.nan
    np.testing.assert_array_equal(old.transform(changed).X,new.transform(changed).X)

def test_v1_wellnorm_default_is_preserved_for_reproduction():
    df=toy();p=FeaturePipeline(mode='wellnorm',curves=['GR'],use_relative_depth=False,curve_context=df)
    b=p.fit_transform(df)
    np.testing.assert_array_equal(b.X[:,b.feature_names.index('GR_isna')], [0.,0.,0.,0.])

def test_curve_subset_retains_identical_row_identity_and_labels():
    df=toy();kw=dict(mode='hybrid',use_relative_depth=False,raw_missingness_mask=True,curve_context=df)
    a=FeaturePipeline(**kw,curves=['GR']).fit_transform(df)
    b=FeaturePipeline(**kw,curves=['GR','RHOB','NPHI','DTC']).fit_transform(df)
    np.testing.assert_array_equal(a.y,b.y)
    np.testing.assert_array_equal(a.wells,b.wells)
    np.testing.assert_array_equal(a.depths,b.depths)
    assert a.X.shape[0]==b.X.shape[0]==len(df)
