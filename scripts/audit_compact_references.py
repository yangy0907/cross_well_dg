"""Read-only bibliographic verification; cache registry records for review."""
from pathlib import Path
import concurrent.futures
import difflib
import json
import re
import time
import unicodedata
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/gse_compact_20261104/audit'


def parse_bib(text):
    entries = {}
    for match in re.finditer(r'@(\w+)\s*\{\s*([^,]+),', text):
        start, level, pos = match.end(), 1, match.end()
        while level and pos < len(text):
            level += (text[pos] == '{') - (text[pos] == '}')
            pos += 1
        body, fields = text[start:pos - 1], {}
        for field in re.finditer(r'(\w+)\s*=\s*\{', body):
            p, depth = field.end(), 1
            while depth and p < len(body):
                depth += (body[p] == '{') - (body[p] == '}')
                p += 1
            fields[field[1].lower()] = body[field.end():p - 1]
        entries[match[2].strip()] = {'entry_type': match[1], **fields}
    return entries


def normal(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9]', '', text)


def fetch(item):
    key, entry = item
    doi = entry.get('doi')
    result = {'key': key, 'original': entry, 'accessed': '2026-11-04'}
    try:
        if doi and doi.startswith(('10.5281/', '10.48550/')):
            url = 'https://api.datacite.org/dois/' + doi
            response = requests.get(url, timeout=35)
            response.raise_for_status()
            raw = response.json()['data']['attributes']
            result.update(registry='DataCite', metadata_url=url, metadata=raw,
                          registry_title=raw['titles'][0]['title'], registry_year=raw.get('publicationYear'))
        else:
            if doi:
                url, params = 'https://api.crossref.org/works/' + doi, None
            else:
                url = 'https://api.crossref.org/works'
                params = {'query.bibliographic': entry.get('title', ''), 'rows': 1,
                          'select': 'DOI,title,author,published,published-print,published-online,volume,issue,page,container-title,type,URL'}
            response = requests.get(url, params=params, timeout=35,
                                    headers={'User-Agent': 'ManuscriptBibliographyAudit/1.0 (read-only metadata verification)'})
            response.raise_for_status()
            message = response.json()['message']
            raw = message['items'][0] if params else message
            fields = ['DOI', 'title', 'author', 'published', 'published-print', 'published-online',
                      'volume', 'issue', 'page', 'container-title', 'type', 'URL', 'link']
            raw = {field: raw[field] for field in fields if field in raw}
            date = raw.get('published-print', raw.get('published', {})).get('date-parts', [[None]])[0]
            result.update(registry='Crossref', metadata_url=response.url, metadata=raw,
                          registry_title=raw.get('title', [''])[0], registry_year=date[0])
        title = normal(entry.get('title', ''))
        registry = normal(result['registry_title'])
        result['title_similarity'] = round(difflib.SequenceMatcher(None, title, registry).ratio(), 4)
        result['status'] = 'metadata_match' if result['title_similarity'] >= .95 else 'manual_review_required'
    except Exception as exc:
        result.update(status='access_failed', error=f'{type(exc).__name__}: {exc}')
    return key, result


def main():
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    entries = parse_bib((ROOT / 'paper/gse_evidence_20261104/refs.bib').read_text(encoding='utf-8'))
    if '--retry' in sys.argv:
        path = OUT / 'reference_registry_audit.json'
        report = json.loads(path.read_text(encoding='utf-8'))
        hints = {'Friedman2001GBM': '10.1214/aos/1013203451', 'Avseth2010QSI': '10.1017/CBO9780511600074'}
        for key, entry in entries.items():
            if report[key]['status'] != 'access_failed' or not (entry.get('doi') or key in hints):
                continue
            query_entry = dict(entry)
            if key in hints:
                query_entry['doi'] = hints[key]
            result = None
            for attempt in range(4):
                _, result = fetch((key, query_entry))
                if result['status'] != 'access_failed':
                    break
                time.sleep(3 * (attempt + 1))
            result['original'] = entry
            report[key] = result
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
            print(key, result['status'], result.get('registry_title', result.get('error', ''))[:160], flush=True)
            time.sleep(1.2)
        return
    report = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        for key, result in pool.map(fetch, entries.items()):
            report[key] = result
            print(key, result['status'], result.get('registry_title', result.get('error', ''))[:150], flush=True)
            (OUT / 'reference_registry_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('TOTAL', len(report), 'MATCH', sum(x['status'] == 'metadata_match' for x in report.values()), flush=True)


if __name__ == '__main__':
    main()
