"""Summarize new-partition consistency and matched-support boundary sensitivity."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.constants import LITHOLOGY_MAP
OUT = ROOT / 'results/v3/relative_depth_validation_run2'


def scalar_metrics(result):
    return {'macro': result['overall']['macro_f1'],
            'accuracy': result['overall']['accuracy'],
            'worst_fixed': result['well_summary_fixed']['worst_well_f1'],
            'w10_fixed': result['well_summary_fixed']['worst10pct_well_f1']}


def tex_table(headers, rows, columns):
    return '\n'.join([r'\begin{tabular}{@{}' + columns + r'@{}}', r'\toprule',
          ' & '.join(headers) + r' \\', r'\midrule'] +
          [' & '.join(str(v) for v in row) + r' \\' for row in rows] +
          [r'\bottomrule', r'\end{tabular}']) + '\n'


def main():
    complete = json.loads((OUT / 'COMPLETE.json').read_text(encoding='utf-8'))
    manifest = json.loads((OUT / 'execution_manifest.json').read_text(encoding='utf-8'))
    if not complete['complete'] or complete['execution_fingerprint'] != manifest['execution_fingerprint']:
        raise RuntimeError('Numerical phase incomplete')
    for rel, expected in manifest['source_hashes'].items():
        h = hashlib.sha256()
        with (ROOT / rel).open('rb') as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
                h.update(chunk)
        if h.hexdigest() != expected:
            raise RuntimeError(f'Frozen input changed: {rel}')
    cfg = manifest['config']
    baseline, stress, well_effects = [], [], []
    details = []
    for seed in cfg['seeds']:
        for curves in cfg['curve_sets']:
            pair = {}
            for tag in ['norel', 'hybrid']:
                p = OUT / 'details' / f'{curves}_seed{seed}_{tag}.json'
                d = json.loads(p.read_text(encoding='utf-8'))
                if d['execution_fingerprint'] != manifest['execution_fingerprint']:
                    raise AssertionError('Wrong case execution')
                details.append(d)
                pair[tag] = d
                baseline.append({'seed': seed, 'curve_set': curves, 'method': d['method'],
                    'regime': d['label_regime'], 'n_train': d['n_train'], 'n_val': d['n_val'],
                    'n_test': d['n_test'], 'n_unseen': d['n_test_unseen_label_kept'],
                    **scalar_metrics(d)})
            records = json.loads((OUT / 'details' / f'{curves}_seed{seed}_stress.json').read_text(encoding='utf-8'))
            if records['execution_fingerprint'] != manifest['execution_fingerprint']:
                raise AssertionError('Wrong stress execution')
            for record in records['records']:
                h, b, n = [record[k] for k in ['hybrid_changed', 'hybrid_same_rows', 'norel_same_rows']]
                for key in ['pooled_labels', 'fixed_well_labels']:
                    if not h[key] == b[key] == n[key]:
                        raise AssertionError('Stress taxonomy mismatch')
                for key in ['well_table', 'well_table_fixed']:
                    sizes = [{r['well_id']: r['n'] for r in obj[key]} for obj in [h, b, n]]
                    if not sizes[0] == sizes[1] == sizes[2]:
                        raise AssertionError('Stress scoring support mismatch')
                r = {'seed': seed, 'curve_set': curves, 'regime': pair['hybrid']['label_regime'],
                     'scenario': record['scenario'], 'n_scored': record['n_scored'],
                     'fraction_rows_retained': record['n_scored'] / pair['hybrid']['n_test'],
                     'n_scored_wells': record['n_scored_wells'],
                     'n_true_classes': len(h['pooled_labels']), 'n_fixed_classes': len(h['fixed_well_labels']),
                     'n_unseen_retained': record['n_unseen_retained']}
                sm, bm, nm = [scalar_metrics(v) for v in [h, b, n]]
                for metric in sm:
                    r[metric + '_changed'] = sm[metric]
                    r[metric + '_baseline_same_rows'] = bm[metric]
                    r[metric + '_norel_same_rows'] = nm[metric]
                    r[metric + '_stress_delta'] = sm[metric] - bm[metric]
                    r[metric + '_relative_gain_original'] = bm[metric] - nm[metric]
                    r[metric + '_relative_gain_stressed'] = sm[metric] - nm[metric]
                stress.append(r)
                bd = {w['well_id']: w for w in b['well_table']}
                nd = {w['well_id']: w for w in n['well_table']}
                for w in h['well_table']:
                    well_effects.append({'seed': seed, 'curve_set': curves, 'scenario': record['scenario'],
                        'well_id': w['well_id'], 'n_scored': w['n'],
                        'present_f1_stress_delta': w['macro_f1'] - bd[w['well_id']]['macro_f1'],
                        'relative_gain_original': bd[w['well_id']]['macro_f1'] - nd[w['well_id']]['macro_f1'],
                        'relative_gain_stressed': w['macro_f1'] - nd[w['well_id']]['macro_f1']})
    common = set.intersection(*(set(d['train_class_names']) for d in details if d['label_regime'] == 'closed_set'))
    for row, d in zip(baseline, details):
        names = {name: str(d['label_mapping'][str(code)]) for code, name in LITHOLOGY_MAP.items() if str(code) in d['label_mapping']}
        row['macro_common'] = np.mean([d['classification_report'].get(names[c], {}).get('f1-score', 0.) for c in sorted(common)])
    base = pd.DataFrame(baseline)
    st = pd.DataFrame(stress)
    base.to_csv(OUT / 'baseline_metrics.csv', index=False)
    st.to_csv(OUT / 'boundary_metrics.csv', index=False)
    pd.DataFrame(well_effects).to_csv(OUT / 'boundary_well_effects.csv', index=False)
    closed = base[base.regime == 'closed_set']
    if closed.empty:
        raise RuntimeError('No closed partition: keep open results; do not create a closed summary')
    summary = closed.groupby(['curve_set', 'method']).agg(
        n_splits=('seed', 'size'), macro_mean=('macro', 'mean'), macro_std=('macro', 'std'),
        macro_common_mean=('macro_common', 'mean'), macro_common_std=('macro_common', 'std'),
        worst_mean=('worst_fixed', 'mean'), worst_std=('worst_fixed', 'std'),
        w10_mean=('w10_fixed', 'mean'), w10_std=('w10_fixed', 'std')).reset_index()
    summary.to_csv(OUT / 'closed_summary.csv', index=False)
    interactions = []
    for seed, g in closed.groupby('seed'):
        pivot = g.pivot(index='curve_set', columns='method', values='macro')
        gains = pivot.lgbm_hybrid - pivot.lgbm_hybrid_norel
        interactions.append({'seed': seed, 'relative_depth_delta_core4': gains.core4,
              'relative_depth_delta_full16': gains.full16, 'difference_of_differences': gains.core4 - gains.full16})
    inter = pd.DataFrame(interactions)
    inter.to_csv(OUT / 'new_split_interactions.csv', index=False)
    discovery = pd.read_csv(ROOT / 'results/v2/pilot_common_core/relative_depth_interaction.csv')
    pd.concat([discovery.assign(phase='v2_well_id_discovery'),
               inter.assign(phase='v3_new_parent_grouped')]).to_csv(OUT / 'discovery_and_followup.csv', index=False)
    closed_st = st[st.regime == 'closed_set']
    stress_summary = closed_st.groupby(['curve_set', 'scenario'], sort=False).agg(
        n_splits=('seed', 'size'), retained_fraction_mean=('fraction_rows_retained', 'mean'),
        macro_stress_delta_mean=('macro_stress_delta', 'mean'),
        macro_stress_delta_std=('macro_stress_delta', 'std'),
        macro_stress_delta_min=('macro_stress_delta', 'min'),
        macro_stress_delta_max=('macro_stress_delta', 'max'),
        original_relative_gain_mean=('macro_relative_gain_original', 'mean'),
        stressed_relative_gain_mean=('macro_relative_gain_stressed', 'mean'),
        worst_stress_delta_mean=('worst_fixed_stress_delta', 'mean'),
        w10_stress_delta_mean=('w10_fixed_stress_delta', 'mean')).reset_index()
    stress_summary.to_csv(OUT / 'boundary_summary.csv', index=False)
    # Figures distinguish phases and show paired changes without population CIs.
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4), sharey=True)
    for ax, data, title in [(axes[0], discovery, 'V2 discovery: well-id splits'),
                             (axes[1], inter, 'New parent-grouped splits')]:
        for _, r in data.iterrows():
            ax.plot([0, 1], [r.relative_depth_delta_full16, r.relative_depth_delta_core4],
                    marker='o', alpha=.75, label=str(int(r.seed)))
        ax.axhline(0, color='black', lw=.7)
        ax.set(xticks=[0, 1], xticklabels=['16 curves', '4 curves'],
               ylabel='Hybrid - Hybrid-norel Macro-F1', title=title)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Relative-depth benefit: new partitions of the same FORCE corpus', fontsize=10)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(OUT / f'fig_v3_new_splits.{ext}', dpi=240)
    plt.close(fig)
    scenarios = [c['name'] for c in cfg['boundary_crops'] if c['name'] != 'baseline']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, curves in zip(axes, ['core4', 'full16']):
        for seed, g in closed_st[(closed_st.curve_set == curves) & (closed_st.scenario != 'baseline')].groupby('seed'):
            g = g.set_index('scenario').loc[scenarios]
            ax.plot(np.arange(len(scenarios)), g.macro_stress_delta, marker='o', alpha=.5,
                     label=str(int(seed)))
        g = stress_summary[(stress_summary.curve_set == curves) & (stress_summary.scenario != 'baseline')].set_index('scenario').loc[scenarios]
        ax.plot(np.arange(len(scenarios)), g.macro_stress_delta_mean, color='black', lw=2,
                marker='D', label='Mean over splits')
        ax.axhline(0, color='black', lw=.7)
        ax.set(xticks=np.arange(len(scenarios)), xticklabels=scenarios, title=curves,
               ylabel='Changed coordinate - baseline Macro-F1\n(on identical retained rows)')
        ax.tick_params(axis='x', rotation=35)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle('Boundary-coordinate sensitivity; other log features and well-z moments frozen', fontsize=10)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(OUT / f'fig_v3_boundary.{ext}', dpi=240)
    plt.close(fig)
    rows = []
    for curves in ['full16', 'core4']:
        for method, label in [('lgbm_hybrid_norel', 'Hybrid-norel'), ('lgbm_hybrid', 'Hybrid')]:
            r = summary[(summary.curve_set == curves) & (summary.method == method)].iloc[0]
            rows.append([curves, label, f'${r.macro_mean:.3f}\\pm{r.macro_std:.3f}$',
                         f'${r.macro_common_mean:.3f}\\pm{r.macro_common_std:.3f}$',
                         f'${r.worst_mean:.3f}\\pm{r.worst_std:.3f}$',
                         f'${r.w10_mean:.3f}\\pm{r.w10_std:.3f}$'])
    (OUT / 'table_v3_new_splits.tex').write_text(tex_table(
        ['Curves', 'Method', 'Macro', r'$C_{\mathrm{common}}$', 'Worst-fixed', 'W10-fixed'], rows, 'llcccc'), encoding='utf-8')
    notes = {'new_split_fit_count': 12, 'reproduction_gate_fit_count': 1,
             'boundary_additional_fits': 0, 'successful_fit_count': 13,
             'prior_failed_gate_fit_count': cfg['prior_failed_gate_fit_count'],
             'total_fit_attempts': 13 + cfg['prior_failed_gate_fit_count'],
             'n_closed_splits': len(inter), 'n_open_splits': int((base.groupby('seed').regime.first() == 'open_set').sum()),
             'new_split_seeds': cfg['seeds'], 'grouping': 'parent_grouped',
             'common_classes': sorted(common),
             'mean_depth_gain_core4': float(inter.relative_depth_delta_core4.mean()),
             'mean_depth_gain_full16': float(inter.relative_depth_delta_full16.mean()),
             'mean_difference_of_differences': float(inter.difference_of_differences.mean()),
             'n_positive_interactions': int((inter.difference_of_differences > 0).sum()),
             'boundary_scenarios_including_baseline': len(cfg['boundary_crops']),
             'boundary_model_scenario_rows': len(st), 'boundary_support_checks_passed': True,
             'scope': 'New overlapping partitions of the same corpus; no external dataset, population test or geological causal identification.'}
    (OUT / 'summary_notes.json').write_text(json.dumps(notes, indent=2), encoding='utf-8')
    print(json.dumps(notes, indent=2))
    print(stress_summary.to_string(index=False))


if __name__ == '__main__':
    main()
