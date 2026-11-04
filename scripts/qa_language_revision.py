"""Check and package the language-only manuscript revision."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

from PIL import Image, ImageDraw
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / 'paper/gse_polished_20261103'
SRC = ROOT / 'paper/gse_final_20261103'
QA = DST / 'qa'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = {'documents': {}, 'protected_assets_identical': {}, 'visual_review': 'pending'}
    for stem, prefix in [('main', 'main'), ('supplementary', 'supp')]:
        reader = PdfReader(DST / f'{stem}.pdf')
        extracted = '\n'.join(p.extract_text() or '' for p in reader.pages)
        (QA / f'{stem}.txt').write_text(extracted, encoding='utf-8')
        log = (DST / f'{stem}.log').read_text(encoding='utf-8', errors='replace')
        checks = {
            'pages': len(reader.pages),
            'undefined_references': bool(re.search(r'(?:Reference|Citation).*undefined|undefined references', log)),
            'tex_errors': bool(re.search(r'^!', log, re.M)),
            'question_pairs': extracted.count('??'),
            'overfull_boxes_pt': re.findall(r'Overfull \\hbox \(([\d.]+)pt', log),
            'pdf_sha256': sha(DST / f'{stem}.pdf'),
        }
        assert not checks['undefined_references'] and not checks['tex_errors'] and not checks['question_pairs']
        report['documents'][stem] = checks
        pages = sorted(QA.glob(f'{prefix}-*.png'))
        assert len(pages) == len(reader.pages), (stem, len(pages), len(reader.pages))
        # Six-page sheets retain enough detail for checking page structure.
        for start in range(0, len(pages), 6):
            sheet = Image.new('RGB', (900, 3 * 445), '#dddddd')
            draw = ImageDraw.Draw(sheet)
            for i, path in enumerate(pages[start:start + 6]):
                im = Image.open(path).convert('RGB')
                im.thumbnail((430, 415))
                x, y = (i % 2) * 450 + 10, (i // 2) * 445 + 25
                sheet.paste(im, (x, y))
                draw.text((x, y - 19), f'{stem} page {start+i+1}', fill='black')
            sheet.save(QA / f'contact_{stem}_{start//6+1}.png')
    protected = list((SRC / 'figs').rglob('*'))
    protected += [SRC / n for n in ['refs.bib', 'supplementary-refs.bib', 'table_v2_common_core.tex', 'table_v3_new_splits.tex', 'highlights.txt']]
    for path in protected:
        if path.is_file():
            name = path.relative_to(SRC).as_posix()
            report['protected_assets_identical'][name] = sha(path) == sha(DST / name)
    assert all(report['protected_assets_identical'].values())
    (QA / 'final_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'documents': report['documents'], 'protected_asset_count': len(report['protected_assets_identical'])}, indent=2))


if __name__ == '__main__':
    main()
