"""Create a portable manuscript-only revision package, preserving the research archive."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / 'paper/gse_polished_20261103'
OUT = ROOT / 'submission_polished_20261104'


def main():
    OUT.mkdir(exist_ok=True)
    qa = json.loads((DST / 'qa/final_check.json').read_text(encoding='utf-8'))
    assert qa['visual_review'] == 'passed'
    files = [p for p in DST.iterdir() if p.is_file() and (
        p.suffix in {'.tex', '.bib', '.cls', '.sty', '.bst', '.pdf'} or
        p.name in {'highlights.txt', 'LANGUAGE_REVISION_CN.md', 'LANGUAGE_AUDIT.json', 'LANGUAGE_EDITS.json'})]
    files += [p for p in (DST / 'figs').rglob('*') if p.is_file()]
    files += [DST / 'qa/final_check.json']
    manifest = {p.relative_to(DST).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(files)}
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False)
    readme = '''Language revision completed 2026-11-04.

Use manuscript/main.pdf and manuscript/supplementary.pdf as the current paper.
The folder name retains the revision start date. Original final files are preserved.

Editable sources and all figures are in manuscript/. Compile there with TeX Live:
  latexmk -pdf -interaction=nonstopmode main.tex
  latexmk -pdf -interaction=nonstopmode supplementary.tex

See LANGUAGE_REVISION_CN.md for scope, LANGUAGE_EDITS.json for passage pairs,
and LANGUAGE_AUDIT.json plus qa/final_check.json for verification.

No new experiments were run. Code, data and results remain in the separate
GSE_review_materials_20261103.zip research archive. Its paper text predates
this language revision; use this package for the current manuscript wording.
'''
    target = OUT / 'GSE_language_revision_20261104.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('README.txt', readme)
        z.writestr('MANIFEST_SHA256.json', manifest_text)
        for p in sorted(files):
            z.write(p, 'manuscript/' + p.relative_to(DST).as_posix())
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for name, expected in manifest.items():
            assert hashlib.sha256(z.read('manuscript/' + name)).hexdigest() == expected
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix('.zip.sha256').write_text(digest + '  ' + target.name + '\n', encoding='utf-8')
    (OUT / 'MANIFEST_SHA256.json').write_text(manifest_text, encoding='utf-8')
    print(json.dumps({'archive': str(target), 'files': len(files), 'bytes': target.stat().st_size,
                      'sha256': digest, 'crc_and_member_hash_checks': 'passed'}, indent=2))


if __name__ == '__main__':
    main()
