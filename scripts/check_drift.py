from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import argparse, json, pandas as pd
from app.drift import dataset_drift

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--baseline',required=True); ap.add_argument('--current',required=True); a=ap.parse_args()
    b=pd.read_csv(a.baseline); c=pd.read_csv(a.current); print(json.dumps(dataset_drift(b,c),indent=2))
if __name__=='__main__': main()
