from __future__ import annotations
import argparse, csv, hashlib, json, tempfile
from datetime import datetime, timezone
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from urllib.parse import urlparse
import pandas as pd
from app.features import normalize_url

TWO={'co.uk','org.uk','ac.uk','gov.uk','com.au','net.au','org.au','co.in','com.br','co.jp','co.nz','co.za'}

def sha256(path: Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()

def regdomain(u:str)->str:
    host=(urlparse(normalize_url(u)).hostname or '').lower().strip('.')
    labs=[x for x in host.split('.') if x]
    if len(labs)<=2:return host
    suffix='.'.join(labs[-2:])
    return '.'.join(labs[-3:]) if suffix in TWO else suffix

def frame_from(path:Path, label:int|None=None)->pd.DataFrame:
    df=pd.read_csv(path).copy()
    if label is not None:
        df['label']=label
    if 'label' not in df.columns:raise ValueError(f'{path}: label is required')
    if 'url' not in df.columns:
        df['_normalized_url']=df.apply(lambda r: normalize_url('',r.get('protocol'),r.get('domain_name'),r.get('address')),axis=1)
        df['url']=df['_normalized_url']
    df['url']=df['url'].fillna('').astype(str).map(normalize_url)
    df=df[df['url'].str.len()>=8].copy()
    df['label']=pd.to_numeric(df['label'],errors='coerce')
    df=df[df['label'].isin([0,1])].copy()
    if 'source' not in df.columns: df['source']=path.name
    if 'observed_at' not in df.columns and 'collected_at' in df.columns: df['observed_at']=df['collected_at']
    if 'observed_at' not in df.columns: df['observed_at']=''
    return df[['url','label','source','observed_at']]

def main():
    ap=argparse.ArgumentParser(description='Build a traceable production training corpus from labeled sources. Synthetic or benign-seed rows are rejected.')
    ap.add_argument('--phishing',required=True);ap.add_argument('--legitimate',required=True);ap.add_argument('--extra',action='append',default=[])
    ap.add_argument('--exclude-eval',action='append',default=[]);ap.add_argument('--exclude-frozen',action='store_true');ap.add_argument('--out',default='data/training/production_training.csv');a=ap.parse_args()
    frames=[frame_from(Path(a.phishing),1),frame_from(Path(a.legitimate),0)]
    for p in a.extra:
        ex=frame_from(Path(p))
        if (ex['source'].astype(str).str.lower().str.strip()=='benign_seed').any(): raise SystemExit('Refusing benign_seed rows in production training.')
        frames.append(ex)
    df=pd.concat(frames,ignore_index=True)
    df['domain']=df['url'].map(regdomain)
    # Exact and domain exclusions from all evaluation sources.
    eu=set();ed=set()
    for p in a.exclude_eval:
        q=Path(p)
        if not q.exists():continue
        x=frame_from(q)
        eu.update(x['url'].tolist());ed.update(x['url'].map(regdomain).tolist())
    if a.exclude_frozen:
        q=Path('data/frozen_regression.csv')
        if q.exists():
            x=frame_from(q);eu.update(x['url'].tolist());ed.update(x['url'].map(regdomain).tolist())
    before=len(df);df=df[~df['url'].isin(eu) & ~df['domain'].isin(ed)].copy()
    # Remove contradictory labels and duplicate exact URLs deterministically.
    conflicts=df.groupby('url')['label'].nunique();bad=set(conflicts[conflicts>1].index)
    df=df[~df['url'].isin(bad)].copy()
    df=df.drop_duplicates('url').sample(frac=1,random_state=42).reset_index(drop=True)
    if df['label'].nunique()<2 or len(df)<200:raise SystemExit(f'Insufficient labeled production corpus after exclusions: {len(df)} rows')
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    payload=df[['url','label','source','observed_at']]
    fd,tmp=tempfile.mkstemp(prefix=f'.{out.name}.',dir=str(out.parent),text=True)
    try:
        payload.to_csv(tmp,index=False)
        import os;os.replace(tmp,out)
    finally:
        import os
        if os.path.exists(tmp):os.unlink(tmp)
    meta={'generated_at':datetime.now(timezone.utc).isoformat(),'rows':len(payload),'phishing_rows':int((payload.label==1).sum()),'legitimate_rows':int((payload.label==0).sum()),'unique_domains':int(df.domain.nunique()),'removed_by_eval_exact_or_domain':int(before-len(df)),'source_files':[{'path':str(a.phishing),'sha256':sha256(Path(a.phishing))},{'path':str(a.legitimate),'sha256':sha256(Path(a.legitimate))}]+[{'path':str(p),'sha256':sha256(Path(p))} for p in a.extra],'evaluation_exclusions':a.exclude_eval,'exclude_frozen':a.exclude_frozen,'synthetic_training_data':False,'benign_seed_allowed':False}
    (out.with_suffix(out.suffix+'.meta.json')).write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
