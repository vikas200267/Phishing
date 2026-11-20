from __future__ import annotations
import argparse, csv, hashlib, json, os, re, tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
OPENPHISH_OFFICIAL = 'https://www.openphish.com/feed.txt'
OPENPHISH_MIRROR = 'https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt'
URLHAUS_RECENT = 'https://urlhaus-api.abuse.ch/v2/files/exports/{key}/recent.csv'
TRANCO_LATEST = 'https://tranco-list.eu/latest_list'


def fetch(url: str, timeout: int = 60, accept: str = '*/*') -> tuple[bytes, str, dict[str, str]]:
    req = Request(url, headers={'User-Agent': 'PhishGuard/phase7-realworld-ingest/1.0', 'Accept': accept})
    with urlopen(req, timeout=timeout) as r:
        return r.read(), r.geturl(), {k.lower(): v for k, v in r.headers.items()}


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def normalize_source_urls(raw: bytes) -> list[str]:
    seen: set[str] = set(); out: list[str] = []
    for line in raw.decode('utf-8', 'replace').splitlines():
        u = line.strip()
        if re.match(r'^https?://[^\s]+$', u, re.I) and u not in seen:
            seen.add(u); out.append(u)
    return out


def write_labeled_csv(path: Path, rows: list[dict]) -> None:
    fields = ['url','label','source','observed_at','label_status','source_sha256','source_url']
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def collect_openphish(out_dir: Path, timeout: int) -> dict:
    errors=[]; raw=None; resolved=None; headers={}
    for source in (OPENPHISH_OFFICIAL, OPENPHISH_MIRROR):
        try:
            raw, resolved, headers = fetch(source, timeout, 'text/plain')
            break
        except Exception as e:
            errors.append(f'{source}: {type(e).__name__}: {e}')
    if raw is None:
        raise RuntimeError('OpenPhish retrieval failed: ' + '; '.join(errors))
    sha=hashlib.sha256(raw).hexdigest(); ts=datetime.now(timezone.utc).isoformat()
    atomic_write_bytes(out_dir/'openphish_raw.txt', raw)
    urls=normalize_source_urls(raw)
    rows=[{'url':u,'label':1,'source':'openphish','observed_at':ts,'label_status':'source-positive','source_sha256':sha,'source_url':resolved} for u in urls]
    write_labeled_csv(out_dir/'openphish.csv', rows)
    meta={'source':'OpenPhish','source_url':resolved,'retrieved_at':ts,'sha256':sha,'entries':len(rows),'listed_urls_visited':False,'errors':errors,'last_modified':headers.get('last-modified')}
    (out_dir/'openphish.meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return meta


def collect_urlhaus(out_dir: Path, key: str, timeout: int) -> dict:
    if not key:
        return {'source':'URLhaus','enabled':False,'reason':'URLHAUS_AUTH_KEY not configured'}
    endpoint=URLHAUS_RECENT.format(key=key)
    raw,resolved,_=fetch(endpoint,timeout,'text/csv')
    sha=hashlib.sha256(raw).hexdigest(); ts=datetime.now(timezone.utc).isoformat()
    atomic_write_bytes(out_dir/'urlhaus_raw.csv',raw)
    lines=[x for x in raw.decode('utf-8','replace').splitlines() if x and not x.startswith('#')]
    reader=csv.DictReader(lines)
    rows=[]; seen=set()
    for r in reader:
        u=(r.get('url') or r.get('URL') or '').strip()
        if re.match(r'^https?://[^\s]+$',u,re.I) and u not in seen:
            seen.add(u); rows.append({'url':u,'label':1,'source':'urlhaus','observed_at':ts,'label_status':'source-positive','source_sha256':sha,'source_url':resolved})
    write_labeled_csv(out_dir/'urlhaus.csv',rows)
    meta={'source':'URLhaus','source_url':resolved,'retrieved_at':ts,'sha256':sha,'entries':len(rows),'listed_urls_visited':False}
    (out_dir/'urlhaus.meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return meta


def collect_tranco(out_dir: Path, timeout: int) -> dict:
    html,resolved,headers=fetch(TRANCO_LATEST,timeout,'text/html')
    text=html.decode('utf-8','replace')
    match=re.search(r'/list/([A-Za-z0-9]+)/1000000',resolved+' '+text)
    if not match:
        raise RuntimeError('Could not resolve current Tranco list ID')
    list_id=match.group(1)
    endpoint=f'https://tranco-list.eu/download/{list_id}/1000000'
    raw,resolved_csv,headers_csv=fetch(endpoint,timeout,'text/csv')
    sha=hashlib.sha256(raw).hexdigest(); ts=datetime.now(timezone.utc).isoformat()
    atomic_write_bytes(out_dir/'tranco_raw.csv',raw)
    domains=[]
    for line in raw.decode('utf-8','replace').splitlines():
        if not line or line.lower().startswith('rank,domain'): continue
        parts=line.split(',',1)
        domain=(parts[1] if len(parts)==2 else line).strip()
        if domain and '.' in domain and ' ' not in domain: domains.append(domain)
    rows=[{'domain':d,'label':'','source':'tranco','observed_at':ts,'label_status':'benign_candidate_not_ground_truth','source_sha256':sha,'source_url':resolved_csv} for d in dict.fromkeys(domains)]
    out=out_dir/'tranco_candidates.csv'; out.parent.mkdir(parents=True,exist_ok=True)
    fields=['domain','label','source','observed_at','label_status','source_sha256','source_url']
    fd,tmp=tempfile.mkstemp(prefix=f'.{out.name}.',dir=str(out.parent),text=True)
    try:
        with os.fdopen(fd,'w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
        os.replace(tmp,out)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    meta={'source':'Tranco','list_id':list_id,'source_url':resolved_csv,'retrieved_at':ts,'sha256':sha,'entries':len(rows),'label_semantics':'candidate-only; not used as verified benign ground truth','last_modified':headers_csv.get('last-modified') or headers.get('last-modified')}
    (out_dir/'tranco.meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return meta


def main():
    ap=argparse.ArgumentParser(description='Fetch real-world threat/popularity sources. Never visits listed URLs and never fabricates labels.')
    ap.add_argument('--out-dir',default=str(ROOT/'data'/'feeds'/'live'))
    ap.add_argument('--timeout',type=int,default=60)
    ap.add_argument('--skip-urlhaus',action='store_true')
    ap.add_argument('--skip-tranco',action='store_true')
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    results={'generated_at':datetime.now(timezone.utc).isoformat(),'visited_listed_urls':False,'sources':[]}
    results['sources'].append(collect_openphish(out,args.timeout))
    if not args.skip_urlhaus:
        try: results['sources'].append(collect_urlhaus(out,os.getenv('URLHAUS_AUTH_KEY',''),args.timeout))
        except Exception as e: results['sources'].append({'source':'URLhaus','error':f'{type(e).__name__}: {e}'})
    if not args.skip_tranco:
        try: results['sources'].append(collect_tranco(out,args.timeout))
        except Exception as e: results['sources'].append({'source':'Tranco','error':f'{type(e).__name__}: {e}'})
    (out/'ingestion_manifest.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))

if __name__=='__main__': main()
