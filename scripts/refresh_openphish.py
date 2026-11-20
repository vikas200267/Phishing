from __future__ import annotations
import argparse, csv, datetime as dt, hashlib, json, os, pathlib, tempfile, urllib.request

OFFICIAL='https://www.openphish.com/feed.txt'
MIRROR='https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt'
ROOT=pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUT=ROOT/'data/feeds/live/openphish.csv'


def fetch(url, ua):
    req=urllib.request.Request(url,headers={'User-Agent':ua})
    with urllib.request.urlopen(req,timeout=30) as r:
        return r.read(), r.geturl()


def main():
    ap=argparse.ArgumentParser(description='Refresh the local OpenPhish snapshot without visiting listed URLs')
    ap.add_argument('--out',default=str(DEFAULT_OUT))
    ap.add_argument('--user-agent',default='PhishGuard/2.2 OpenPhish feed refresher')
    a=ap.parse_args()
    out=pathlib.Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    raw=resolved=None; errors=[]
    for source in (OFFICIAL,MIRROR):
        try:
            raw,resolved=fetch(source,a.user_agent); break
        except Exception as e:
            errors.append(f'{source}: {type(e).__name__}: {e}')
    if raw is None:
        raise RuntimeError('OpenPhish refresh failed: '+'; '.join(errors))
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    seen=set(); rows=[]
    for line in raw.decode('utf-8','replace').splitlines():
        u=line.strip()
        if u and u.lower().startswith(('http://','https://')) and u not in seen:
            seen.add(u); rows.append({'url':u,'label':1,'source':'openphish','collected_at':now})
    # Atomic replace prevents a reader from observing a partially written feed.
    fd,tmp=tempfile.mkstemp(prefix='.openphish.',dir=str(out.parent),text=True)
    try:
        with os.fdopen(fd,'w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['url','label','source','collected_at']); w.writeheader(); w.writerows(rows); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,out)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    meta={'source_url':resolved,'retrieved_at':now,'sha256':hashlib.sha256(raw).hexdigest(),'entries':len(rows),'errors':errors,'listed_urls_visited':False}
    out.with_suffix(out.suffix+'.meta.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))


if __name__=='__main__': main()
