"""Label-dependent decomposition of observed log shifts; no causal attribution or refit."""
from pathlib import Path
import json
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.constants import PROTOCOL1_MULTIREP_SEEDS
OUT=ROOT/'results/v2/existing_well_analysis'
CURVES=['GR','RHOB','NPHI','DTC']

def main():
    df=pd.read_parquet(ROOT/'data/processed/labeled.parquet',columns=['well_id','lithology']+CURVES)
    means=df.groupby(['well_id','lithology'])[CURVES].mean()
    counts=df.groupby(['well_id','lithology'])[CURVES].count()
    records=[]
    for mode in ['well_id','parent_grouped']:
        for seed in PROTOCOL1_MULTIREP_SEEDS:
            ref=json.loads((ROOT/f'results/multirep/details/{mode}/seed{seed}_lgbm_global.json').read_text(encoding='utf-8'))
            if ref['n_test_unseen_label_kept']:continue
            prefix='protocol1_seed' if mode=='well_id' else 'protocol1_parent_grouped_seed'
            split=json.loads((ROOT/f'data/splits/{prefix}{seed}.json').read_text(encoding='utf-8'))
            tr=df[df.well_id.isin(split['train_wells'])]
            sm=tr.groupby('lithology')[CURVES].mean()
            sn=tr.groupby('lithology')[CURVES].count()
            sd=tr[CURVES].std(ddof=0)
            for well in split['test_wells']:
                wm=means.loc[well].reindex(sm.index)
                wn=counts.loc[well].reindex(sm.index,fill_value=0)
                for c in CURVES:
                    qw=wn[c]/wn[c].sum(); qs=sn[c]/sn[c].sum()
                    if wn[c].sum()==0 or not np.isfinite(sd[c]) or sd[c]==0:continue
                    if ((qw>0)&sm[c].isna()).any():continue
                    composition=float(((qw-qs)*sm[c]).sum())
                    within=float((qw*(wm[c]-sm[c])).sum())
                    observed=float((qw*wm[c]).sum()-(qs*sm[c]).sum())
                    error=abs(observed-composition-within)/sd[c]
                    if error>1e-9:raise AssertionError((mode,seed,well,c,error))
                    records.append({'mode':mode,'seed':seed,'well_id':well,'curve':c,
                        'composition_z':composition/sd[c],'within_class_z':within/sd[c],
                        'observed_z':observed/sd[c],'identity_error_z':error})
    signed=pd.DataFrame(records)
    signed.to_csv(OUT/'shift_decomposition_by_curve.csv',index=False)
    signed['composition_abs_z']=signed.composition_z.abs()
    signed['within_class_abs_z']=signed.within_class_z.abs()
    group=signed.groupby(['mode','seed','well_id'])
    agg=group[['composition_abs_z','within_class_abs_z']].mean()
    agg['n_valid_curves']=group.size()
    agg['composition_fraction']=agg.composition_abs_z/(agg.composition_abs_z+agg.within_class_abs_z).replace(0,np.nan)
    agg=agg.reset_index()
    effects=pd.read_csv(OUT/'paired_well_split_effects.csv')
    effects=effects[effects.contrast=='HN-Global']
    merged=effects.merge(agg,on=['mode','seed','well_id'],validate='one_to_one')
    merged.to_csv(OUT/'effects_with_shift_decomposition.csv',index=False)
    cols=['delta_present_f1','composition_abs_z','within_class_abs_z','composition_fraction','n_valid_curves']
    perwell=merged.groupby(['mode','parent_well','well_id'])[cols].mean().reset_index()
    parent=perwell.groupby(['mode','parent_well'])[cols].mean().reset_index()
    parent.to_csv(OUT/'shift_decomposition_per_parent.csv',index=False)
    associations=[]
    for mode,g in parent.groupby('mode'):
        for c in cols[1:4]:
            v=g[[c,'delta_present_f1']].dropna()
            associations.append({'mode':mode,'descriptor':c,'n_parents':len(v),
                'rho':v[c].corr(v.delta_present_f1,method='spearman'),
                'meaning':'Post-hoc true-label-dependent association; not an instrument/geology causal decomposition or deployment predictor.'})
    pd.DataFrame(associations).to_csv(OUT/'shift_decomposition_associations.csv',index=False)
    plt.rcParams.update({'pdf.fonttype':42,'font.family':'DejaVu Sans','font.size':9})
    fig,axs=plt.subplots(1,3,figsize=(11,3.7))
    g=parent[parent['mode']=='well_id']
    for ax,c,title in zip(axs,cols[1:4],['Composition component','Within-class component','Relative composition magnitude']):
        v=g[[c,'delta_present_f1']].dropna()
        ax.scatter(v[c],v.delta_present_f1,color='#4C78A8',s=20,alpha=.75)
        ax.axhline(0,color='black',lw=.7);ax.set(xlabel=title,ylabel='HN - Global present-class F1')
    fig.suptitle('Observed-log mean-shift decomposition on labelled support\nExploratory parent averages; components are not causal effects',fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT/'fig_v2_shift_decomposition.png',dpi=240)
    fig.savefig(OUT/'fig_v2_shift_decomposition.pdf')
    plt.close(fig)
    print(pd.DataFrame(associations).to_string(index=False),flush=True)
    note={'max_identity_error_z':float(signed.identity_error_z.max()),'n_curve_records':len(signed),
          'support':'Non-missing labelled samples; each curve uses its own observed-class proportions.',
          'identity':'target mean - source mean = sum((q_target-q_source)*mu_source) + sum(q_target*(mu_target-mu_source))',
          'limitations':'Uses target ground-truth labels post-hoc. Within-class shifts include fluid, geology, measurement, sampling and label differences. Not a causal measurement-drift estimate.'}
    (OUT/'shift_decomposition_notes.json').write_text(json.dumps(note,indent=2),encoding='utf-8')

if __name__=='__main__':main()
