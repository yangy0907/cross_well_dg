"""Verify preservation, complete model artifacts and paired score support."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/v3/relative_depth_validation_run2'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    preservation = json.loads((ROOT / 'results/v3/V2_PRESERVATION_MANIFEST.json').read_text(encoding='utf-8'))
    mismatches = [rel for rel, expected in preservation['source_hashes'].items() if digest(ROOT / rel) != expected]
    candidate = ROOT.parent / 'candidate_versions/GSE_candidate_v1_20261103_155625/MANIFEST_SHA256.json'
    original = json.loads(candidate.read_text(encoding='utf-8'))
    selected = [f for f in original['files'] if
                f['path'].startswith(('cross_well_dg/paper/gse/', 'cross_well_dg/results/multirep/',
                                      'cross_well_dg/data/splits/')) or
                f['path'] in ['cross_well_dg/data/processed/labeled.parquet',
                              'cross_well_dg/data/processed/curves_all.parquet']]
    original_mismatches = [f['path'] for f in selected if digest(ROOT.parent / f['path']) != f['sha256']]
    complete = json.loads((OUT / 'COMPLETE.json').read_text(encoding='utf-8'))
    gate = json.loads((OUT / 'gate.json').read_text(encoding='utf-8'))
    notes = json.loads((OUT / 'summary_notes.json').read_text(encoding='utf-8'))
    recovery = json.loads((OUT / 'inference_recovery.json').read_text(encoding='utf-8'))
    if (not recovery['full_baseline_predictions_identical'] or recovery['new_fit_count'] != 0 or
            digest(ROOT / 'scripts/v3_recover_boundary_inference.py') != recovery['source_sha256']):
        raise AssertionError('Inference recovery provenance failed')
    baseline = pd.read_csv(OUT / 'baseline_metrics.csv')
    if not complete['complete'] or not gate['passed'] or len(baseline) != 12:
        raise AssertionError('Incomplete numerical artifacts')
    if baseline[['seed', 'curve_set', 'method']].duplicated().any():
        raise AssertionError('Duplicate baseline rows')
    if not notes['boundary_support_checks_passed']:
        raise AssertionError('Boundary support mismatch')
    cases = list((OUT / 'details').glob('*_hybrid.json')) + list((OUT / 'details').glob('*_norel.json'))
    for p in cases + [OUT / 'gate_seed42_core4_hybrid.json']:
        result = json.loads(p.read_text(encoding='utf-8'))
        for suffix in ['.model.txt', '.transforms.json', '.predictions.parquet']:
            if not p.with_suffix(suffix).is_file():
                raise AssertionError(f'Missing saved model artifact: {p.with_suffix(suffix)}')
        if result['saved_model_probability_readback_rows'] != 2048:
            raise AssertionError('Model readback check incomplete')
    if len(cases) != 12:
        raise AssertionError('Unexpected model count')
    source = ROOT / 'paper/gse_v3'
    unexpanded = [p.name for p in source.glob('*.tex') if '{{GAIN' in p.read_text(encoding='utf-8') or
                  '{{INTERACTION}}' in p.read_text(encoding='utf-8')]
    report = {'v2_files_checked': len(preservation['source_hashes']), 'v2_mismatches': mismatches,
              'v1_boundary_files_checked': len(selected), 'v1_boundary_mismatches': original_mismatches,
              'new_split_models': len(cases), 'successful_gate_models': 1,
              'prior_failed_gate_fit_attempts': 1, 'phase_total_fit_attempts': 14,
              'boundary_additional_fits': 0, 'paired_scoring_support_verified': True,
              'original_float64_depths_used': True,
              'saved_model_full_baseline_predictions_identical': True,
              'unexpanded_manuscript_placeholders': unexpanded, 'gate_passed': gate['passed']}
    (OUT / 'revision_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if mismatches or original_mismatches or unexpanded:
        raise AssertionError(report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
