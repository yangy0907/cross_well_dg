"""Exploratory well-level effect heterogeneity from existing v1 predictions; no refits."""
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
from src.constants import CORE_CURVES, PROTOCOL1_MULTIREP_SEEDS

OUT=ROOT/'results/v2/existing_well_analysis'
CORE4=['GR','RHOB','NPHI','DTC']
CONTRASTS={'HN-Global':('lgbm_hybrid_norel','lgbm_global'),
           'Hybrid-HN':('lgbm_hybrid','lgbm_hybrid_norel'),
           'Hybrid-WellNorm-legacy':('lgbm_hybrid','lgbm_wellnorm')}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    gate=json.loads((ROOT/'results/v2/reproduction/gate.json').read_text(encoding='utf-8'))
    if not gate['passed']:raise RuntimeError('Reproduction gate failed')
    full=pd.read_parquet(ROOT/'data/processed/curves_all.parquet',columns=['well_id','depth']+CORE_CURVES)
    lab=pd.read_parquet(ROOT/'data/processed/labeled.parquet',columns=['well_id','lithology']+CORE4)
    grp=full.groupby('well_id',sort=True)
    traits=pd.DataFrame(index=grp.size().index)
    traits['missing4']=full[CORE4].isna().mean(axis=1).groupby(full.well_id).mean()
    traits['missing16']=full[CORE_CURVES].isna().mean(axis=1).groupby(full.well_id).mean()
    traits['depth_span']=grp.depth.max()-grp.depth.min()
    traits['abs_cali_minus_bs']=abs(full.CALI-full.BS).groupby(full.well_id).median()
    traits['abs_drho']=abs(full.DRHO).groupby(full.well_id).median()
    means=grp[CORE4].mean()
    counts=lab.groupby(['well_id','lithology']).size().unstack(fill_value=0)
    proportions=counts.div(counts.sum(axis=1),axis=0)
    traits['true_n_classes']=(counts>0).sum(axis=1)
    traits['true_dominant_share']=proportions.max(axis=1)
    traits['true_class_entropy']=-(proportions*np.log(proportions.replace(0,np.nan))).sum(axis=1)
    traits['true_shale_fraction']=proportions.get(65000,0)
    traits.index.name='well_id'
    traits.to_csv(OUT/'well_descriptors.csv')
    rows=[]
    for mode in ['well_id','parent_grouped']:
        for seed in PROTOCOL1_MULTIREP_SEEDS:
            detail_dir=ROOT/f'results/multirep/details/{mode}'
            ref=json.loads((detail_dir/f'seed{seed}_lgbm_global.json').read_text(encoding='utf-8'))
            if ref['n_test_unseen_label_kept']!=0:continue
            prefix='protocol1_seed' if mode=='well_id' else 'protocol1_parent_grouped_seed'
            split=json.loads((ROOT/f'data/splits/{prefix}{seed}.json').read_text(encoding='utf-8'))
            train=lab[lab.well_id.isin(split['train_wells'])]
            center=train[CORE4].median()
            scale=train[CORE4].std(ddof=0).replace(0,np.nan)
            drift=means.sub(center).div(scale).abs().mean(axis=1)
            source_prior=train.lithology.value_counts(normalize=True).reindex(counts.columns,fill_value=0)
            tv=proportions.sub(source_prior).abs().sum(axis=1)/2
            cache={}
            for method in {m for pair in CONTRASTS.values() for m in pair}:
                d=json.loads((detail_dir/f'seed{seed}_{method}.json').read_text(encoding='utf-8'))
                cache[method]={k:{w['well_id']:w for w in d[k]} for k in ['well_table','well_table_fixed']}
            for contrast,(left,right) in CONTRASTS.items():
                for well in split['test_wells']:
                    a=cache[left]['well_table'][well];b=cache[right]['well_table'][well]
                    row={'mode':mode,'seed':seed,'well_id':well,'contrast':contrast,
                         'parent_well':re.sub(r'\s+[A-Za-z0-9]+$','',well.strip()),
                         'delta_present_f1':a['macro_f1']-b['macro_f1'],
                         'delta_fixed_f1':cache[left]['well_table_fixed'][well]['macro_f1']-cache[right]['well_table_fixed'][well]['macro_f1'],
                         'delta_accuracy':a['accuracy']-b['accuracy'],
                         'left_present_f1':a['macro_f1'],'right_present_f1':b['macro_f1'],
                         'curve_location_drift4':float(drift.loc[well]),'true_class_prior_tv':float(tv.loc[well]),
                         **traits.loc[well].to_dict()}
                    rows.append(row)
    long=pd.DataFrame(rows)
    long.to_csv(OUT/'paired_well_split_effects.csv',index=False)
    keys=['mode','contrast','well_id','parent_well']
    numeric=long.select_dtypes(include='number').columns.drop('seed')
    by_well=long.groupby(keys)[list(numeric)].mean().reset_index()
    appearances=long.groupby(keys).size().rename('n_test_appearances').reset_index()
    by_well=by_well.merge(appearances,on=keys,validate='one_to_one')
    by_well.to_csv(OUT/'effects_averaged_per_well.csv',index=False)
    # One parent-level point after averaging the overlapping split appearances.
    parent=by_well.groupby(['mode','contrast','parent_well'])[list(numeric)].mean().reset_index()
    parent.to_csv(OUT/'effects_averaged_per_parent.csv',index=False)
    descriptors=['curve_location_drift4','missing4','missing16','abs_cali_minus_bs','abs_drho',
                 'true_class_prior_tv','true_n_classes','true_dominant_share']
    correlations=[]; summaries=[]
    for (mode,contrast),g in parent.groupby(['mode','contrast']):
        summaries.append({'mode':mode,'contrast':contrast,'n_parents':len(g),
            'n_positive_parents':int((g.delta_present_f1>0).sum()),
            'mean_delta_present':g.delta_present_f1.mean(),'median_delta_present':g.delta_present_f1.median(),
            'p10_delta_present':g.delta_present_f1.quantile(.1),'p90_delta_present':g.delta_present_f1.quantile(.9)})
        for descriptor in descriptors:
            valid=g[[descriptor,'delta_present_f1']].dropna()
            rho=valid[descriptor].corr(valid.delta_present_f1,method='spearman') if len(valid)>2 else np.nan
            correlations.append({'mode':mode,'contrast':contrast,'descriptor':descriptor,
                'n_parents_with_data':len(valid),'spearman_rho':rho,
                'label_dependent':descriptor.startswith('true_'),
                'inference':'exploratory descriptive rank association; no p-value or causal claim'})
    pd.DataFrame(correlations).to_csv(OUT/'exploratory_rank_associations.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'parent_effect_summary.csv',index=False)
    eligible=by_well[(by_well['mode']=='well_id')&(by_well.contrast=='HN-Global')&(by_well.n_test_appearances>=2)]
    examples=pd.concat([eligible.nsmallest(3,'delta_present_f1').assign(selection='largest_losses'),
                        eligible.nlargest(3,'delta_present_f1').assign(selection='largest_gains')])
    examples.to_csv(OUT/'posthoc_case_wells.csv',index=False)
    plt.rcParams.update({'pdf.fonttype':42,'font.family':'DejaVu Sans','font.size':9})
    g=parent[(parent['mode']=='well_id')&(parent.contrast=='HN-Global')]
    fig,axs=plt.subplots(2,2,figsize=(10,7))
    ax=axs[0,0]
    ax.hist(g.delta_present_f1,bins=15,color='#4C78A8',edgecolor='white')
    ax.axvline(0,color='black',lw=.8);ax.set(xlabel='HN - Global present-class F1',ylabel='Parent wells',title='A. Mean paired effect per parent')
    for ax,x,title in [(axs[0,1],'curve_location_drift4','B. Observable curve-location difference'),
                        (axs[1,0],'missing4','C. Observable missing-curve fraction'),
                        (axs[1,1],'true_class_prior_tv','D. True class-prior shift (post-hoc only)')]:
        v=g[[x,'delta_present_f1']].dropna()
        ax.scatter(v[x],v.delta_present_f1,s=23,alpha=.75,color='#4C78A8')
        ax.axhline(0,color='black',lw=.8)
        ax.set(xlabel=x.replace('_',' '),ylabel='HN - Global present-class F1',title=title)
    fig.suptitle('Existing full16 results: overlapping splits averaged within parent\nExploratory associations, not a validated deployment rule',fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT/'fig_v2_well_heterogeneity.png',dpi=240)
    fig.savefig(OUT/'fig_v2_well_heterogeneity.pdf')
    plt.close(fig)
    note={'scope':'Existing v1 closed splits only; no refitting', 'split_well_rows':len(long),
          'parent_effect_summary':summaries,
          'unit':'Average appearances per well, then average sibling wells per parent; models and source wells remain shared.',
          'limitations':['No population confidence intervals or p-values.','Class-composition descriptors require true labels and are not deployment predictors.',
                         'Drift descriptors also reflect geology and sampling; they do not identify instrument effects.',
                         'Hybrid-WellNorm-legacy includes the old missing-mask asymmetry; not a clean causal contrast.',
                         'Case wells selected post-hoc for illustration; n_test_appearances is reported.']}
    (OUT/'analysis_notes.json').write_text(json.dumps(note,indent=2,ensure_ascii=False),encoding='utf-8')
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)

if __name__=='__main__':main()
