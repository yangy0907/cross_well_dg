from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / 'paper/gse_compact_20261104'
OUT = ROOT / 'submission_compact_20261104'


def main():
    qa = json.loads((DST/'qa/final_check.json').read_text(encoding='utf-8'))
    assert qa['visual_review'] == 'passed'
    files = [p for p in DST.iterdir() if p.is_file() and p.suffix in {'.tex','.bib','.cls','.sty','.bst','.pdf','.txt','.md'}]
    files += [p for p in (DST/'figs').rglob('*') if p.is_file()]
    files += [p for p in (DST/'audit').iterdir() if p.is_file()]
    files += [DST/'qa/final_check.json']
    manifest = {p.relative_to(DST).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    OUT.mkdir(exist_ok=True)
    target = OUT/'GSE_compact_revision_20261104.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('README.txt','''Current compact GSE revision, 2026-11-04.
Use manuscript/main.pdf and manuscript/supplementary.pdf.
Figure 1 is supplied as editable SVG, PDF and PNG in manuscript/figs/.
REVISION_REPORT_CN.md explains scope; manuscript/audit/ contains code,
reference and compression checks plus the filtered primary reporting record.

Compile from manuscript/, with TeX Live and latexmk:
  latexmk -pdf supplementary.tex
  latexmk -pdf main.tex
  latexmk -pdf supplementary.tex
  latexmk -pdf main.tex
Both files import cross-document labels; run both again if a label changes.

Existing GSE_review_materials_20261103.zip remains the separate code/data/model
archive for these experiments. Its manuscript wording is older. No new training
or model inference was performed for the compact revision. Earlier manuscript
versions and candidate backups remain unchanged in the workspace.
''')
        z.writestr('MANIFEST_SHA256.json',json.dumps(manifest,indent=2,ensure_ascii=False))
        for p in sorted(files): z.write(p,'manuscript/'+p.relative_to(DST).as_posix())
        for name in ['build_compact_revision.py','draw_compact_overview.py','audit_compact_revision.py','audit_compact_references.py','qa_compact_revision.py']:
            z.write(ROOT/'scripts'/name,'workspace_scripts/'+name)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():
            assert hashlib.sha256(z.read('manuscript/'+name)).hexdigest()==digest
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix('.zip.sha256').write_text(digest+'  '+target.name+'\n',encoding='utf-8')
    (OUT/'MANIFEST_SHA256.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'archive':str(target),'bytes':target.stat().st_size,'verified_files':len(files),'sha256':digest},indent=2))


if __name__=='__main__': main()
