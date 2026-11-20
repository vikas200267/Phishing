from __future__ import annotations
import argparse, json, sys, pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.registry import current
from app.url_model import load_bundle
from app.train import evaluate
from app.features import normalize_url

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',default=str(ROOT/'data'/'frozen_regression.csv'))
    ap.add_argument('--version',default=None)
    a=ap.parse_args()
    version=a.version or current()
    model_path=ROOT/'models'/version/'model.joblib'
    model=load_bundle(model_path)
    df=pd.read_csv(a.csv); urls=[]
    for _,r in df.iterrows():
        raw=r.get('url')
        if pd.isna(raw) or not str(raw).strip():
            raw=normalize_url('',r.get('protocol'),r.get('domain_name'),r.get('address'))
        urls.append(str(raw))
    y=df.label.astype(int).to_numpy(); p=model.predict_proba(urls)
    out=evaluate(p,y,model.threshold); out['model_version']=version; out['samples']=len(urls)
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
