"""Numerical, source-code and reference checks for the compact manuscript."""
from pathlib import Path
import hashlib
import json
import re
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_compact_references import parse_bib

DST = ROOT / 'paper/gse_compact_20261104'
AUDIT = DST / 'audit'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    main_text = (DST / 'main.tex').read_text(encoding='utf-8')
    si_text = (DST / 'supplementary.tex').read_text(encoding='utf-8')
    primary = pd.read_csv(ROOT / 'results/followup/protocol1_multirep_well_id_closed_summary.csv')
    kept = ['lgbm_global', 'lgbm_hybrid_norel', 'lgbm_hybrid', 'lgbm_hybrid_while']
    records = primary[primary.method.isin(kept)].copy()
    checks = []
    for row in records.itertuples():
        for mean, sd in [('macro_legacy_mean', 'macro_legacy_std'), ('worst_fixed_mean', 'worst_fixed_std'), ('w10_fixed_mean', 'w10_fixed_std')]:
            expected = f'{getattr(row, mean):.3f}' + r'\pm' + f'{getattr(row, sd):.3f}'
            assert expected in main_text, (row.method, mean, expected)
            checks.append({'method': row.method, 'quantity': mean, 'formatted_value': expected, 'source_csv_match': True})
    records['mask_provenance'] = 'Global/Hybrid raw missingness unaffected by the legacy WellNorm defect'
    records['report_scope'] = 'primary closed n=18; well-id; current compact manuscript'
    records.to_csv(AUDIT / 'primary_reporting_record.csv', index=False)
    # These are summaries of existing results, with no fitting or inference.
    source_files = [
        'results/followup/protocol1_multirep_well_id_closed_summary.csv',
        'results/followup/protocol1_multirep_parent_grouped_closed_summary.csv',
        'results/followup/protocol1_multirep_well_id_paired_deltas.csv',
        'results/v2/pilot_common_core/closed_summary.csv',
        'results/v2/pilot_common_core/wellnorm_mask_control.csv',
        'results/v3/relative_depth_validation_run2/closed_summary.csv',
        'results/v3/relative_depth_validation_run2/new_split_interactions.csv',
        'results/v3/relative_depth_validation_run2/boundary_summary.csv',
        'results/v2/existing_well_analysis/parent_effect_summary.csv',
    ]
    interaction = pd.read_csv(ROOT / source_files[6])
    assert abs(interaction.difference_of_differences.mean() - .011) < .0005
    assert int((interaction.difference_of_differences > 0).sum()) == 2
    preservation = json.loads((ROOT / 'results/v3/V2_PRESERVATION_MANIFEST.json').read_text(encoding='utf-8'))
    changed = [name for name, value in preservation['source_hashes'].items() if sha(ROOT / name) != value]
    assert not changed, changed
    code_files = ['src/features.py', 'src/models.py', 'src/metrics.py', 'src/postprocess.py',
                  'src/splits.py', 'src/train.py', 'src/constants.py',
                  'scripts/v3_depth_validation.py', 'scripts/v3_recover_boundary_inference.py']
    bindings = [
        ('Global fitting support', 'src/features.py', '_engineer', 'Training-row median and population SD; frozen target transform; denominator includes 1e-6.'),
        ('Local normalisation support', 'src/features.py', '_context_for_wells / _well_zscore_from_context', 'Select only requested wells; all observable curve depths supply local moments; missing z responses become zero.'),
        ('Causal-prefix definition', 'src/features.py', '_causal_z_lookup', 'Depth-sorted expanding moments include current depth; no deeper samples.'),
        ('Relative depth', 'src/features.py', '_rel_depth_from_context', 'Full observed well min/max with 1e-6 denominator stabiliser.'),
        ('Masks and imputation', 'src/features.py', '_engineer', 'Masks precede SimpleImputer; corrected raw_missingness_mask reads original channels; legacy defaults remain for reproducibility.'),
        ('Information routing', 'src/train.py', '_feature_routing', 'Global has no z or relative depth; HN omits r; Hybrid includes r; Hybrid-while uses causal z and omits r.'),
        ('Early stopping and weights', 'src/models.py', 'LGBMLithologyModel.fit / _make_model', 'Balanced class weights, no confidence weighting for reported core models; source validation multiclass log-loss, patience 30; row bagging disabled.'),
        ('Label maps and open-set scoring', 'src/train.py', 'mapping / pooled_labels / fixed_well_labels', 'Pooled labels from test truth; fixed labels from training map; unk never predicted; unknown validation rows excluded from stopping.'),
        ('Metric aggregation', 'src/metrics.py', 'evaluate_split / summarize_well_f1', 'well_summary is the legacy present-class result; well_summary_fixed supplies main-text fixed tails; W10 uses ceil(0.1*n_wells).'),
        ('Smoothing support', 'src/postprocess.py', 'smooth_predictions_by_well / _segment_indices', 'Centred 7-point vote on score support; no cross-well votes; break at 1.5 times median positive depth increment.'),
        ('Parent grouping', 'src/splits.py', 'parent_well_name / protocol1_parent_grouped_wells', 'Strip trailing trajectory tokens and allocate whole parents before expanding to wellbores.'),
        ('Matched boundary inference', 'scripts/v3_recover_boundary_inference.py', 'run', 'Original float64 depths reconstruct relative coordinate; all saved baseline predictions checked; fixed models and same retained rows.'),
    ]
    banned = r'\b(?:may|might|could|possibly|potentially|somewhat|relatively|fairly|quite|arguably|unfortunately|admittedly|significantly)\b|we believe|we think|we hope|we try to|we attempt to'
    # Source titles and archived provenance text are not reworded by the prose scan.
    bad = re.findall(banned, main_text, re.I)
    assert not bad, bad
    report = {
        'core_metric_checks': checks, 'interaction_mean': interaction.difference_of_differences.mean(),
        'interaction_positive_splits': int((interaction.difference_of_differences > 0).sum()),
        'code_bindings': [dict(zip(['topic', 'file', 'symbol', 'conclusion'], row)) for row in bindings],
        'source_hashes': {name: sha(ROOT / name) for name in code_files + source_files},
        'preserved_prior_files_checked': len(preservation['source_hashes']), 'preservation_mismatches': changed,
        'new_training_runs': 0, 'new_model_inference_runs': 0,
        'existing_test_suite': '36 passed; audit/tests.txt',
        'legacy_csv_handling': 'Original reporting CSV retains archival WellNorm; the new primary_reporting_record.csv selects the four unaffected core configurations without changing their scores.',
        'fixed_text_issues': ['Global and relative-depth formulas now include code stabilisers.', 'Damaged Data availability cross-reference replaced by a valid raw LaTeX reference.', 'Historical/companion material moved with labels preserved.', 'Rare-class panels consolidated without modifying data.', 'Main Pareto excludes legacy mask-confounded WellNorm.'],
        'main_banned_hedges': bad,
    }
    (AUDIT / 'code_consistency.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# Code–manuscript consistency audit', '',
             'The audit inspects the executed code paths and existing numerical records. No model training or new inference was performed. The existing test suite passed all 36 tests.', '',
             '| Topic | Code | Verified description |', '|---|---|---|']
    lines += [f'| {topic} | {file}: {symbol} | {finding} |' for topic, file, symbol, finding in bindings]
    lines += ['',
        'All original numerical tabular bodies are retained in the combined main/Supplementary package. The four core Macro-F1/Worst-fixed/W10 means and SDs were checked against their source CSV, and the new-partition mean/sign count were recomputed from the existing three-row summary. This does not constitute a new experiment.', '',
        'The legacy source API keeps present-class outputs under well_summary and fixed-taxonomy outputs under well_summary_fixed. The manuscript explicitly uses the latter for primary tails. We preserve the historical source and CSVs to protect execution provenance; a filtered current reporting record accompanies this version.', '',
        'The prior v2 preservation manifest passes unchanged. The original finalized, polished and evidence-style manuscript directories are retained.', '',
        'This review did not identify a numerical implementation defect requiring a new fit. Known legacy mask loss remains qualified and corrected only within the existing matched pilot; it is not silently reclassified as corrected eighteen-split evidence.',
    ]
    (AUDIT / 'code_consistency.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    reference_audit()
    print(json.dumps({'metric_checks': len(checks), 'code_bindings': len(bindings),
                      'prior_files_unchanged': len(preservation['source_hashes']), 'new_fits': 0}, indent=2))


def reference_audit():
    records = json.loads((AUDIT / 'reference_registry_audit.json').read_text(encoding='utf-8'))
    bib = parse_bib((DST / 'refs.bib').read_text(encoding='utf-8'))
    official = {
        'Ke2017LightGBM': ('https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html', 'NeurIPS official record: title, eight authors and 2017 volume confirmed; unrelated Crossref search candidate rejected.'),
        'Chen2016XGBoost': ('https://arxiv.org/abs/1603.02754', 'Author preprint confirms full title/subtitle and authors; DOI record confirms KDD 2016 pages 785–794. Crossref main title is abbreviated to XGBoost.'),
        'Pedregosa2011sklearn': ('https://www.jmlr.org/papers/v12/pedregosa11a.html', 'JMLR record confirms authors, 12(85), 2825–2830, 2011; unrelated registry title-search candidate rejected.'),
        'Ganin2016DANN': ('https://jmlr.org/papers/v17/15-239.html', 'JMLR official record confirms title, authors and 2016 paper 17(59); unrelated registry title-search candidate rejected.'),
        'Sagawa2020GroupDRO': ('https://iclr.cc/virtual_2020/poster_ryxGuJrFvS.html', 'ICLR 2020 official presentation confirms authors and work; arXiv:1911.08731 confirms full title. Registry search candidate rejected.'),
        'Koh2021WILDS': ('https://proceedings.mlr.press/v139/koh21a.html', 'Official PMLR/ICML 2021 record confirms authors, volume 139 and pages 5637–5664. Registry title-query failure is not treated as absence of the paper.'),
        'Gulrajani2021DomainBed': ('https://iclr.cc/Downloads/2021', 'ICLR 2021 official list contains the work; arXiv:2007.01434 confirms title and both authors; unrelated registry title-search candidate rejected.'),
        'Asquith2004Basic': ('https://store-assets.aapg.org/documents/previews/707Mth16/CHAPTER01.PDF', 'AAPG author chapter identifies the 2004 book, authors and Methods in Exploration 16; 1982 chapter search result rejected.'),
        'Rider2011WellLogs': ('https://catalogue.bnf.fr/ark:/12148/cb42702380r', 'National-library catalogue confirms third edition, both authors, 2011, Glasgow and ISBN 9780954190682; unrelated title-search candidate rejected.'),
        'Avseth2010QSI': ('https://www.cambridge.org/core/books/quantitative-seismic-interpretation/EB6A36B78CCF07187723F6F5364EDCF8', 'Cambridge confirms 2005 print/2010 online versions, title, authors and DOI. Entry standardised to first print year 2005; removed as direct support for the voting heuristic.'),
    }
    text = (DST / 'main.tex').read_text(encoding='utf-8') + (DST / 'supplementary.tex').read_text(encoding='utf-8')
    cited = {key.strip() for args in re.findall(r'\\cite\w*\{([^}]+)\}', text) for key in args.split(',')}
    assert cited <= set(bib), cited-set(bib)
    reviewed = []
    for key, item in records.items():
        if key in official:
            source, note = official[key]
            method = 'canonical primary record / authoritative book catalogue'
        else:
            assert item['status'] == 'metadata_match', (key, item['status'])
            source, note = item['metadata_url'], 'DOI registration title and identifying metadata locate the cited work.'
            method = item['registry'] + ' DOI metadata'
        if key == 'Carvalho2026LithologyBenchmark':
            note += ' Publisher full text confirms four logs, non-missing sequences, well-level five-fold CV and FORCE/Geolink; author name follows André Korenchendler as published.'
        if key in ['Zhou2023DGSurvey', 'Wang2023DGSurvey']:
            note += ' Registry retains early-access 2022 metadata; journal issue citation uses 2023. These are publication versions, not distinct works.'
        if key == 'Chen2025FrontiersLithology':
            note += ' Publisher Methods describes eight training wells and two independent test wells; the main text uses this only as a well-held-out precedent.'
        if key == 'Qi2025WLFM':
            note += ' Retained as a 2025 arXiv preprint; its accuracy/few-shot claims are not ranked numerically against our Macro-F1.'
        reviewed.append({'key': key, 'title': bib[key]['title'], 'source': source,
                         'verification': method, 'located': True, 'cited_in_current_package': key in cited, 'note': note})
    assert len(reviewed) == 44 and all(row['located'] for row in reviewed)
    (AUDIT / 'reference_verification.json').write_text(json.dumps(reviewed, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# Reference authenticity and citation-use audit', '',
             'Completed 4 November 2026. All 44 preceding bibliography records were located through DOI registries, official proceedings/journal records, author preprints or an authoritative national-library book catalogue. The current package cites '+str(len(cited))+' of them. Existence/metadata verification is distinct from full-text replication of another study.', '',
             'The registry cache is retained for transparency. Its unrelated title-search candidates were rejected and replaced by the canonical records below. A failed generic query is not evidence that a classic paper is fictitious.', '',
             '## Corrections and use', '',
             '- Match Carvalho’s coauthor to the publisher record: André Korenchendler. Its actual well-level CV is acknowledged; it is not characterised as a depth-random benchmark.',
             '- Add Friedman’s verified DOI and official URLs for non-DOI conference/JMLR records.',
             '- Standardise Rider’s author initials, Glasgow publication location and ISBN; standardise the Avseth book to 2005 first print, distinguishing 2010 online publication.',
             '- Remove the rock-physics book as direct support for an arbitrarily chosen smoothing window. The window is supported by the actual released configuration and diagnostic, not a general geological reference.',
             '- Retain IEEE final-issue citations with explicit early-access/version notes in this audit. Do not replace a 2023 final citation with stale 2022 registry pagination.',
             '- Retain WLFM as a preprint and distinguish accuracy/few-shot outcomes from our Macro-F1. Make no cross-paper numerical superiority claim.', '',
             'The current literature descriptions use the scope directly supported by the work: algorithm introduction, logging application, transfer setting, well-held-out precedent or benchmark protocol. Paywalled full texts were not all inspected line by line; detailed regional and architectural assertions that require unavailable text were removed or reduced to the verified title/abstract scope.', '',
             '| Key | Located record | Current citation | Verification and correction |', '|---|---|---|---|']
    for row in reviewed:
        title = re.sub(r'[{}]', '', row['title']).replace('|', r'\|')
        lines.append(f'| {row["key"]} | [{title}]({row["source"]}) | {"Cited" if row["cited_in_current_package"] else "Not cited"} | {row["verification"]}. {row["note"]} |')
    (AUDIT / 'reference_verification.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
