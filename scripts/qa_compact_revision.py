from pathlib import Path
import hashlib
import json
import re
from PIL import Image, ImageDraw
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / 'paper/gse_compact_20261104'
QA = DST / 'qa'


def main():
    report = {'visual_review': 'pending', 'documents': {}}
    for stem, prefix in [('main', 'main'), ('supplementary', 'supp')]:
        pdf = DST / (stem+'.pdf')
        reader = PdfReader(pdf)
        txt = '\n'.join(page.extract_text() or '' for page in reader.pages)
        log = (DST/(stem+'.log')).read_text(encoding='utf-8', errors='replace')
        row = {'pages': len(reader.pages), 'question_pairs': txt.count('??'),
               'undefined': bool(re.search(r'(?:Reference|Citation).*undefined|undefined references', log)),
               'tex_errors': bool(re.search('^!', log, re.M)),
               'overfull_hbox_pt': re.findall(r'Overfull \\hbox \(([\d.]+)pt', log),
               'sha256': hashlib.sha256(pdf.read_bytes()).hexdigest()}
        assert not row['undefined'] and not row['tex_errors'] and not row['question_pairs'], row
        report['documents'][stem] = row
        (QA/(stem+'.txt')).write_text(txt, encoding='utf-8')
        pages = sorted(QA.glob(prefix+'-*.png'))
        assert len(pages) == len(reader.pages), (prefix, len(pages), len(reader.pages))
        for start in range(0,len(pages),6):
            sheet = Image.new('RGB',(960,1440),'#dddddd')
            draw = ImageDraw.Draw(sheet)
            for i,path in enumerate(pages[start:start+6]):
                im = Image.open(path).convert('RGB'); im.thumbnail((455,445))
                x,y=10+(i%2)*480, 25+(i//2)*480
                sheet.paste(im,(x,y)); draw.text((x,y-18),f'{stem} {start+i+1}',fill='black')
            sheet.save(QA/f'contact_{stem}_{start//6+1}.png')
    (QA/'final_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
