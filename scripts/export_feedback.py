from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import argparse, csv
from app.feedback import verified_rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='data/verified_feedback.csv'); a=ap.parse_args()
    rows=verified_rows()
    with open(a.out,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['url','label']); w.writerows(rows)
    print(f'wrote {len(rows)} rows -> {a.out}')
if __name__=='__main__': main()
