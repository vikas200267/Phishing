from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import argparse, json
from pathlib import Path
import pandas as pd
from app.train import evaluate, url_series
from app.url_model import load_bundle

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description='Compare a challenger against production on the immutable regression set')
    ap.add_argument('--challenger',required=True); ap.add_argument('--regression',default=str(ROOT/'data'/'frozen_regression.csv'))
    args=ap.parse_args()
    d=pd.read_csv(args.regression); d['_normalized_url']=url_series(d)
    prod_ver=(ROOT/'models'/'CURRENT').read_text().strip(); prod=load_bundle(ROOT/'models'/prod_ver/'model.joblib')
    chal=load_bundle(ROOT/'models'/args.challenger/'model.joblib')
    y=d.label.astype(int).to_numpy(); urls=d._normalized_url.tolist()
    pm=evaluate(prod.predict_proba(urls),y,prod.threshold); cm=evaluate(chal.predict_proba(urls),y,chal.threshold)
    print(json.dumps({'production':{'version':prod_ver,**pm},'challenger':{'version':args.challenger,**cm}},indent=2))
if __name__=='__main__': main()
