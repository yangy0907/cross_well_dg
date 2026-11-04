"""Create a local reviewer archive, inventory and independent checksum sidecar."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'submission_final_20261103'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    files = set()
    for folder in ['src', 'scripts', 'configs', 'tests', 'data/splits', 'results']:
        files.update(p for p in (ROOT / folder).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix not in {'.pyc', '.tmp'}
                     and p.name != '.running.lock')
    for n in ['README.md', 'LICENSE', 'requirements.txt', 'requirements.lock.txt', 'run_experiment.py']:
        files.add(ROOT / n)
    for n in ['labeled.parquet', 'curves_all.parquet']:
        files.add(ROOT / 'data/processed' / n)
    for folder in ['gse', 'gse_v2', 'gse_v3', 'gse_final_20261103']:
        for p in (ROOT / 'paper' / folder).rglob('*'):
            if p.is_file() and 'qa' not in p.parts and (
                p.suffix in {'.tex', '.bib', '.cls', '.sty', '.bst', '.md'} or
                'figs' in p.parts or p.name in {'main.pdf', 'supplementary.pdf', 'highlights.txt'}):
                files.add(p)
    inventory = [{'path': str(p.relative_to(ROOT)).replace('\\', '/'),
                  'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(files)]
    manifest = {'version': 'GSE_final_20261103',
                'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                'file_count': len(inventory), 'total_uncompressed_bytes': sum(r['bytes'] for r in inventory),
                'scope': 'Local reviewer package. No submission or public deposition is asserted.',
                'files': inventory}
    readme = '''# GSE reviewer materials, 2026-11-03

This package contains manuscript sources/PDFs, code, processed inputs, saved
splits, numerical results, models and predictions. No new model training was
performed to finalize the manuscript. Earlier candidates are retained as
historical material; the submission manuscript is paper/gse_final_20261103/.

Quick evidence map (no training required):
- Historical primary 18-closed-split results: results/multirep/ and results/followup/.
- Corrected matched-row pilot: results/v2/pilot_common_core/closed_summary.csv,
  matched_metrics_long.csv and wellnorm_mask_control.csv.
- New parent-grouped comparisons: results/v3/relative_depth_validation_run2/
  new_split_interactions.csv and closed_summary.csv.
- Same-support boundary stresses: boundary_metrics.csv, boundary_summary.csv,
  boundary_well_effects.csv and details/*_boundary_predictions.parquet.
- Model/gate/inference provenance: execution_manifest.json, gate.json,
  inference_recovery.json and revision_verification.json in the same v3 folder.
- Historical WellNorm masks are confounded. The final paper reports them as
  archival diagnostics in Supplementary Tables S17-S19, not primary estimates.

Integrity check: run `python VERIFY_PACKAGE.py` from this extracted root.
Python's standard library is sufficient. The SHA-256 inventory describes final
packaging, and does not replace the original per-experiment execution manifests.

Numerical code was tested on Windows, Python 3.11.5, with requirements.lock.txt.
The final TeX files compile with TeX Live 2026 and latexmk. Processed input tables
are included to avoid rebuilding raw LAS data. The public source dataset is
FORCE 2020, https://doi.org/10.5281/zenodo.4351156; raw LAS files are not duplicated.

Existing experiment scripts have local Windows launchers and some frozen
Windows-style manifest paths. Do not assume untested cross-platform training
reproduction. No training command is needed to inspect or verify reported rows.

The original v3 training schedule persisted all twelve models before a depth-
precision check stopped boundary inference. The saved-model recovery uses
float64 input depths and does not retrain. The launcher handles this route.

No independent external-corpus result, calibrated deployment guarantee, public
software DOI or journal submission is asserted by this local archive.
'''
    verifier = '''from pathlib import Path
import hashlib, json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'REVIEW_FILES_SHA256.json').read_text(encoding='utf-8'))
fail=[]
for row in manifest['files']:
 p=root/row['path']; h=hashlib.sha256()
 if not p.is_file(): fail.append(row['path']); continue
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 if h.hexdigest()!=row['sha256']:fail.append(row['path'])
print('Checked',len(manifest['files']),'files; mismatches:',len(fail))
if fail:print('\\n'.join(fail));raise SystemExit(1)
'''
    (OUT / 'REVIEW_FILES_SHA256.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    (OUT / 'README_REVIEWER.md').write_text(readme, encoding='utf-8')
    (OUT / 'VERIFY_PACKAGE.py').write_text(verifier, encoding='utf-8')
    archive = OUT / 'GSE_review_materials_20261103.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for row in inventory:
            z.write(ROOT / row['path'], row['path'])
        for n in ['REVIEW_FILES_SHA256.json', 'README_REVIEWER.md', 'VERIFY_PACKAGE.py']:
            z.write(OUT / n, n)
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
    if bad:
        raise AssertionError(f'ZIP CRC error: {bad}')
    checksum = digest(archive)
    (OUT / (archive.name + '.sha256')).write_text(checksum + '  ' + archive.name + '\n', encoding='ascii')
    (OUT / 'PACKAGE_VERIFICATION.json').write_text(json.dumps({
        'archive_bytes': archive.stat().st_size, 'archive_sha256': checksum,
        'zip_crc_passed': True, 'inventoried_files': len(inventory),
        'manifest_in_archive': True, 'published': False}, indent=2), encoding='utf-8')
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size,
                     'files': len(inventory), 'sha256': checksum, 'zip_crc_passed': True}))


if __name__ == '__main__':
    main()
