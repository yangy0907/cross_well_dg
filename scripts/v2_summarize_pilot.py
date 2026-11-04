"""Summarize the bounded pilot, explicitly distinguishing new fits and reused v1 references."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.constants import LITHOLOGY_MAP
OUT=ROOT/'results/v2/pilot_common_core'
LABELS={'lgbm_global':'Global','lgbm_wellnorm':'WellNorm (raw mask)',
        'lgbm_hybrid_norel':'Hybrid-norel','lgbm_hybrid':'Hybrid'}

def main():
    cfg=yaml.safe_load((ROOT/'configs/v2_pilot_common_core.yaml').read_text(encoding='utf-8'))
    complete=json.loads((OUT/'COMPLETE.json').read_text(encoding='utf-8'))
    manifest=json.loads((OUT/'execution_manifest.json').read_text(encoding='utf-8'))
    if not complete['complete'] or complete['new_fit_count']!=20:raise RuntimeError('Pilot incomplete')
    for rel,expected in manifest['source_hashes'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Execution source/data/split changed: {rel}')
    refs=OUT/'references';refs.mkdir(exist_ok=True)
    all_results=[]
    for seed in cfg['seeds']:
        for curves in ['core4','full16']:
            for method in cfg['methods']:
                p=OUT/'details'/f'{curves}_seed{seed}_{method}.json'
                if p.exists():
                    d=json.loads(p.read_text(encoding='utf-8'))
                    if d['execution_fingerprint']!=manifest['execution_fingerprint']:raise RuntimeError('Fingerprint mismatch')
                else:
                    if curves!='full16' or method not in cfg['full16_reference_methods']:raise RuntimeError(f'Missing new fit {p}')
                    p=ROOT/f'results/multirep/details/well_id/seed{seed}_{method}.json'
                    d=json.loads(p.read_text(encoding='utf-8'))
                    # Preserve original provenance instead of relabelling an old row as a new fit.
                    d.update({'curve_set':'full16','evidence_origin':'v1_saved_feature_equivalent',
                        'reference_file':str(p.relative_to(ROOT)),
                        'reference_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                        'reference_note':'Raw masks are algebraically unchanged for Global and Hybrid; tests protect equivalence. Reference n_jobs=-1, new fits n_jobs=8.'})
                    (refs/f'seed{seed}_{method}.json').write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
                all_results.append(d)
    common=set.intersection(*(set(d['train_class_names']) for d in all_results if d['label_regime']=='closed_set'))
    rows=[];class_rows=[]
    for d in all_results:
        ccommon=np.mean([d['per_class'].get(c,{}).get('f1',0.) for c in sorted(common)])
        row={'seed':d['seed'],'curve_set':d['curve_set'],'method':d['method'],
             'origin':d['evidence_origin'],'regime':d['label_regime'],
             'macro':d['overall']['macro_f1'],'macro_common':ccommon,'accuracy':d['overall']['accuracy'],
             'worst_fixed':d['well_summary_fixed']['worst_well_f1'],
             'w10_fixed':d['well_summary_fixed']['worst10pct_well_f1'],
             'n_train':d['n_train'],'n_train_fit':d['n_train_fit'],'n_val':d['n_val'],
             'n_test':d['n_test'],'n_unseen':d['n_test_unseen_label_kept'],'n_features':d['n_features']}
        rows.append(row)
        for name,pc in d['per_class'].items():
            class_rows.append({'seed':d['seed'],'curve_set':d['curve_set'],'method':d['method'],
                'regime':d['label_regime'],'class':name,**pc})
    long=pd.DataFrame(rows)
    for seed,g in long.groupby('seed'):
        for key in ['n_train','n_train_fit','n_val','n_test','n_unseen']:
            if g[key].nunique()!=1:raise AssertionError((seed,key))
    long.to_csv(OUT/'matched_metrics_long.csv',index=False)
    pc=pd.DataFrame(class_rows);pc.to_csv(OUT/'per_class_long.csv',index=False)
    closed=long[long.regime=='closed_set']
    summaries=[]
    for (curves,method),g in closed.groupby(['curve_set','method'],sort=False):
        r={'curve_set':curves,'method':method,'n_closed_splits':len(g)}
        for c in ['macro','macro_common','worst_fixed','w10_fixed']:
            r[c+'_mean']=g[c].mean();r[c+'_std']=g[c].std(ddof=1)
        summaries.append(r)
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'closed_summary.csv',index=False)
    long[long.regime=='open_set'].to_csv(OUT/'open_stress.csv',index=False)
    paired=[]
    for method,g in closed.groupby('method'):
        for metric in ['macro','macro_common','worst_fixed','w10_fixed']:
            pivot=g.pivot(index='seed',columns='curve_set',values=metric)
            delta=pivot.core4-pivot.full16
            paired.append({'method':method,'metric':metric,'contrast':'core4-full16',
                'n':len(delta),'delta_mean':delta.mean(),'delta_std':delta.std(ddof=1),
                'n_positive':int((delta>0).sum())})
    pd.DataFrame(paired).to_csv(OUT/'paired_curve_set_changes.csv',index=False)
    interactions=[]
    for seed,g in closed.groupby('seed'):
        p=g.pivot(index='method',columns='curve_set',values='macro')
        a=float(p.loc['lgbm_hybrid','core4']-p.loc['lgbm_hybrid_norel','core4'])
        b=float(p.loc['lgbm_hybrid','full16']-p.loc['lgbm_hybrid_norel','full16'])
        interactions.append({'seed':int(seed),'relative_depth_delta_core4':a,
            'relative_depth_delta_full16':b,'difference_of_differences':a-b})
    pd.DataFrame(interactions).to_csv(OUT/'relative_depth_interaction.csv',index=False)
    corrections=[]
    for seed in cfg['seeds']:
        new=next(d for d in all_results if d['seed']==seed and d['curve_set']=='full16' and d['method']=='lgbm_wellnorm')
        old=json.loads((ROOT/f'results/multirep/details/well_id/seed{seed}_lgbm_wellnorm.json').read_text(encoding='utf-8'))
        corrections.append({'seed':seed,'regime':new['label_regime'],
            'legacy_macro':old['overall']['macro_f1'],'raw_mask_macro':new['overall']['macro_f1'],
            'delta_macro':new['overall']['macro_f1']-old['overall']['macro_f1'],
            'legacy_worst_fixed':old['well_summary_fixed']['worst_well_f1'],
            'raw_mask_worst_fixed':new['well_summary_fixed']['worst_well_f1']})
    pd.DataFrame(corrections).to_csv(OUT/'wellnorm_mask_control.csv',index=False)
    # Publication figures show SD across overlapping splits, never population CIs.
    plt.rcParams.update({'pdf.fonttype':42,'font.family':'DejaVu Sans','font.size':9})
    fig,axs=plt.subplots(1,2,figsize=(10,4.3))
    x=np.arange(len(cfg['methods']))
    for ax,metric,title in [(axs[0],'macro','Pooled Macro-F1'),(axs[1],'worst_fixed','Coverage-aware Worst-fixed')]:
        for j,(curves,color) in enumerate([('full16','#4C78A8'),('core4','#F58518')]):
            s=summary[summary.curve_set==curves].set_index('method').loc[cfg['methods']]
            ax.bar(x+(j-.5)*.35,s[metric+'_mean'],width=.35,yerr=s[metric+'_std'],
                color=color,label=curves,capsize=3)
        ax.set_xticks(x,[LABELS[m] for m in cfg['methods']],rotation=20,ha='right')
        ax.set(ylabel=title,title=title);ax.legend(frameon=False)
    fig.suptitle('Matched-row pilot: 3 closed splits; bars show mean ± split SD\nFull16 Global/Hybrid references reused; all core4 and corrected WellNorm rows newly fitted',fontsize=10)
    fig.tight_layout();fig.savefig(OUT/'fig_v2_common_core.png',dpi=240);fig.savefig(OUT/'fig_v2_common_core.pdf');plt.close(fig)
    # Compact TeX table, preserving all methods and both curve sets.
    tex=[r'\begin{tabular}{@{}llcccc@{}}',r'\toprule',
         r'Curves & Method & Macro & $C_{\mathrm{common}}$ & Worst-fixed & W10-fixed \\',r'\midrule']
    for curves in ['full16','core4']:
        for method in cfg['methods']:
            s=summary[(summary.curve_set==curves)&(summary.method==method)].iloc[0]
            vals=[f"${s[k+'_mean']:.3f}\\pm{s[k+'_std']:.3f}$" for k in ['macro','macro_common','worst_fixed','w10_fixed']]
            tex.append(f'{curves} & {LABELS[method]} & '+' & '.join(vals)+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}']
    (OUT/'table_v2_common_core.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    notes={'new_fit_count':20,'reused_reference_count':12,'total_matched_rows':len(long),
        'closed_splits':sorted(int(s) for s in closed.seed.unique()),
        'open_splits':sorted(int(s) for s in long[long.regime=='open_set'].seed.unique()),
        'common_classes':sorted(common),'matched_training_test_rows':True,
        'pilot':'Exploratory; three overlapping closed splits, not population inference or a full replication of Carvalho preprocessing.',
        'raw_masks':'New fits consistently use original observed missingness. Old WellNorm references are shown only in the explicit mask control.',
        'reference_compute':'v1 references used n_jobs=-1; new pilot fits use 8 threads. Same learner hyperparameters/early stopping/rows/seeds; runtime is not compared.'}
    (OUT/'summary_notes.json').write_text(json.dumps(notes,indent=2,ensure_ascii=False),encoding='utf-8')
    print(summary.to_string(index=False),flush=True)
    print('Mask control\n'+pd.DataFrame(corrections).to_string(index=False),flush=True)

if __name__=='__main__':main()
