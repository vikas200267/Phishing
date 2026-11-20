from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from urllib.parse import urlparse
import pandas as pd
from app.features import normalize_url

TWO={'co.uk','org.uk','ac.uk','gov.uk','com.au','net.au','org.au','co.in','com.br','co.jp','co.nz','co.za'}

def registrable(url: str)->str:
    try: host=(urlparse(normalize_url(url)).hostname or '').lower().strip('.')
    except Exception: host=''
    labels=[x for x in host.split('.') if x]
    if len(labels)<=2:return host
    suffix='.'.join(labels[-2:])
    return '.'.join(labels[-3:]) if suffix in TWO and len(labels)>=3 else suffix

def load(path: Path):
    df=pd.read_csv(path)
    if 'url' in df.columns: urls=df['url'].fillna('').astype(str).map(normalize_url)
    elif '_normalized_url' in df.columns: urls=df['_normalized_url'].fillna('').astype(str).map(normalize_url)
    else:
        cols=[c for c in ('protocol','domain_name','address') if c in df.columns]
        if 'domain_name' not in df.columns: raise ValueError(f'{path} has no URL/domain columns')
        urls=df.apply(lambda r: normalize_url('',r.get('protocol'),r.get('domain_name'),r.get('address')),axis=1)
    domains=urls.map(registrable)
    return set(urls[urls.str.len()>=8]), set(domains[domains.str.len()>0])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',required=True,nargs='+');ap.add_argument('--test',required=True,nargs='+');ap.add_argument('--allow-domain-overlap',action='store_true');ap.add_argument('--out',default='data/manifests/leakage_audit.json');a=ap.parse_args()
    train_sets=[load(Path(x)) for x in a.train]
    tu=set().union(*(x[0] for x in train_sets)); td=set().union(*(x[1] for x in train_sets))
    report={'train':a.train,'tests':{},'exact_overlap_total':0,'domain_overlap_total':0}
    failed=[]
    for s in a.test:
        u,d=load(Path(s)); eu=sorted(tu & u); ed=sorted(td & d)
        report['tests'][s]={'exact_overlap_count':len(eu),'domain_overlap_count':len(ed),'exact_overlap_sample':eu[:10],'domain_overlap_sample':ed[:10]}
        report['exact_overlap_total']+=len(eu);report['domain_overlap_total']+=len(ed)
        if eu or (ed and not a.allow_domain_overlap): failed.append(s)
    report['passed']=not failed;report['failed_tests']=failed
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2));
    raise SystemExit(1 if failed else 0)
if __name__=='__main__':main()
