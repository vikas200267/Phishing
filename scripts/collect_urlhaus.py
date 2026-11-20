from __future__ import annotations
import argparse, csv, datetime as dt, pathlib, urllib.request

ENDPOINT='https://urlhaus-api.abuse.ch/v2/files/exports/{key}/recent.csv'

def main():
    ap=argparse.ArgumentParser(description='Download a recent URLhaus CSV export. Requires an Auth-Key.')
    ap.add_argument('--auth-key',required=True); ap.add_argument('--out',default='data/feeds/urlhaus_recent.csv')
    a=ap.parse_args(); pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    req=urllib.request.Request(ENDPOINT.format(key=a.auth_key),headers={'User-Agent':'PhishGuard/5.x URLhaus collector'})
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    with urllib.request.urlopen(req,timeout=60) as r: raw=r.read().decode('utf-8','replace')
    lines=raw.splitlines(); start=next((i for i,x in enumerate(lines) if x and not x.startswith('#')),0)
    reader=csv.DictReader(lines[start:]); fieldnames=reader.fieldnames or []
    url_key=next((x for x in ['url','URL'] if x in fieldnames),None)
    if not url_key: raise RuntimeError('No URL column found in URLhaus export')
    rows=[]
    for row in reader:
        u=(row.get(url_key) or '').strip()
        if u: rows.append({'url':u,'label':1,'source':'urlhaus','collected_at':now})
    with open(a.out,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['url','label','source','collected_at']); w.writeheader(); w.writerows(rows)
    print(f'Wrote {len(rows)} URLhaus malware URLs to {a.out}')
if __name__=='__main__': main()
