"""Check revision boundaries and completed numerical artifacts against saved hashes."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/v2'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    candidate = ROOT.parent / 'candidate_versions/GSE_candidate_v1_20261103_155625'
    manifest = json.loads((candidate / 'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
    selected = [f for f in manifest['files'] if
                f['path'].startswith(('cross_well_dg/paper/gse/',
                                      'cross_well_dg/results/multirep/',
                                      'cross_well_dg/data/splits/')) or
                f['path'] in ['cross_well_dg/data/processed/labeled.parquet',
                              'cross_well_dg/data/processed/curves_all.parquet']]
    mismatches = [f['path'] for f in selected
                  if digest(ROOT.parent / f['path']) != f['sha256']]
    pilot = OUT / 'pilot_common_core'
    complete = json.loads((pilot / 'COMPLETE.json').read_text(encoding='utf-8'))
    notes = json.loads((pilot / 'summary_notes.json').read_text(encoding='utf-8'))
    metrics = pd.read_csv(pilot / 'matched_metrics_long.csv')
    if not complete['complete'] or complete['new_fit_count'] != 20:
        raise AssertionError('Incomplete pilot')
    if len(metrics) != 32 or metrics[['seed', 'curve_set', 'method']].duplicated().any():
        raise AssertionError('Missing or duplicate matched rows')
    if notes['new_fit_count'] != 20 or notes['reused_reference_count'] != 12:
        raise AssertionError('Evidence counts do not match')
    reproduction = json.loads((OUT / 'reproduction/gate.json').read_text(encoding='utf-8'))
    if not reproduction['passed']:
        raise AssertionError('Original source reproduction failed')
    control = json.loads((OUT / 'mask_thread_control/control_report.json').read_text(encoding='utf-8'))
    placeholder_files = [str(p.relative_to(ROOT)) for p in (ROOT / 'paper/gse_v2').glob('*.tex')
                         if '{{CORE}}' in p.read_text(encoding='utf-8') or
                         '{{THREAD}}' in p.read_text(encoding='utf-8')]
    report = {'unchanged_candidate_boundaries': not mismatches,
              'checked_legacy_files': len(selected), 'mismatches': mismatches,
              'new_pilot_fits': 20, 'additional_diagnostic_fits': 1,
              'original_source_reproduction_fits': 1, 'phase_total_fit_executions': 22,
              'reused_reference_rows': 12, 'matched_rows': len(metrics),
              'closed_seeds': notes['closed_splits'], 'open_seeds': notes['open_splits'],
              'reproduction_passed': reproduction['passed'],
              'same_thread_legacy_equivalent': control['legacy_thread_equivalence'],
              'unexpanded_manuscript_placeholders': placeholder_files,
              'scope': 'Local revision audit; not population inference or acceptance certification.'}
    (OUT / 'revision_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if mismatches or placeholder_files:
        raise AssertionError(report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
