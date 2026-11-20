from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import argparse, json
from app.registry import current, versions, promote, rollback

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    sub.add_parser('list'); p=sub.add_parser('promote'); p.add_argument('version'); sub.add_parser('rollback'); a=ap.parse_args()
    if a.cmd=='list': print(json.dumps({'current':current(),'versions':versions()},indent=2))
    elif a.cmd=='promote': print(json.dumps(promote(a.version),indent=2))
    else: print(json.dumps(rollback(),indent=2))
if __name__=='__main__': main()
