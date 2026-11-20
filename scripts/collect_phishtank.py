#!/usr/bin/env python3
from __future__ import annotations
import argparse, gzip
from pathlib import Path
import httpx

URL='https://data.phishtank.com/data/{key}/online-valid.csv.gz'

def main():
    ap=argparse.ArgumentParser(description='Download the current verified/online PhishTank CSV feed.')
    ap.add_argument('--app-key',required=True); ap.add_argument('--out',default='data/feeds/phishtank_online_valid.csv'); a=ap.parse_args()
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    with httpx.Client(timeout=60,headers={'User-Agent':'PhishGuard/2.0 security research'}) as c:
        r=c.get(URL.format(key=a.app_key)); r.raise_for_status()
    raw=gzip.decompress(r.content) if r.content[:2]==b'\x1f\x8b' else r.content
    out.write_bytes(raw); print(f'wrote {len(raw)} bytes -> {out}')
if __name__=='__main__': main()
