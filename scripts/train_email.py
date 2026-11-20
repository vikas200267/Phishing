from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import argparse
from pathlib import Path
from app.email_model import train_email_csv

def main():
    ap=argparse.ArgumentParser(description='Train the optional labeled email classifier; use --eval-csv for an independent chronological holdout.')
    ap.add_argument('--csv',required=True); ap.add_argument('--eval-csv'); ap.add_argument('--out',required=True); a=ap.parse_args(); Path(a.out).parent.mkdir(parents=True,exist_ok=True); metrics=train_email_csv(a.csv,a.out,a.eval_csv); print({'path':a.out,'metrics':metrics})
if __name__=='__main__': main()
