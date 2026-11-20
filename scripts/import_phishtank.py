from __future__ import annotations
import argparse, pandas as pd
from pathlib import Path
from urllib.parse import urlparse

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',default='data/feeds/phishtank_phishing.csv'); a=ap.parse_args()
    df=pd.read_csv(a.input)
    required={'url','verified','online'}
    if not required.issubset(df.columns): raise SystemExit(f'Missing columns: {sorted(required-set(df.columns))}')
    df=df[(df.verified.astype(str).str.lower()=='yes')&(df.online.astype(str).str.lower()=='yes')].copy()
    rows=[]
    for u in df.url.dropna().astype(str):
        p=urlparse(u); rows.append({'url':u,'protocol':p.scheme,'domain_name':p.hostname or '', 'address':p.path + (('?' + p.query) if p.query else ''), 'label':1})
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).drop_duplicates('url').to_csv(out,index=False)
    print(f'wrote {len(rows)} verified phishing URLs -> {out}')
if __name__=='__main__': main()
