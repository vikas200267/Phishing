from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import pandas as pd
from app.features import normalize_url


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def inspect_dataset(path: Path) -> dict:
    """Return evidence-backed counts for CSV datasets and JSON URL corpora.

    A file-level capture timestamp is metadata, not a per-row timestamp, and
    is never used to claim that temporal model validation is ready.
    """
    try:
        if path.suffix.lower() == '.json':
            payload = json.loads(path.read_text(encoding='utf-8'))
            rows = payload.get('urls', []) if isinstance(payload, dict) else []
            urls = [r.get('url') if isinstance(r, dict) else r for r in rows]
            normalized = pd.Series([normalize_url(str(u)) for u in urls if u], dtype='string')
            usable = normalized[normalized.str.len() >= 8]
            return {
                'format': 'json_url_corpus',
                'rows': int(len(urls)),
                'valid_url_rows': int(len(usable)),
                'unique_normalized_urls': int(usable.drop_duplicates().shape[0]),
                'capture_generated_at': payload.get('generated_at'),
                'source_role': payload.get('source_role'),
                'listed_urls_visited': payload.get('visited_urls', payload.get('listed_urls_visited')),
                'declared_source_counts': payload.get('source_counts', {}),
                'embedded_url_hash_sha256': payload.get('url_hash_sha256'),
            }

        df = pd.read_csv(path)
        if 'url' in df.columns:
            u = df['url'].fillna('').astype(str).map(normalize_url)
        elif '_normalized_url' in df.columns:
            u = df['_normalized_url'].fillna('').astype(str).map(normalize_url)
        elif 'domain_name' in df.columns:
            u = df.apply(lambda r: normalize_url('', r.get('protocol'), r.get('domain_name'), r.get('address')), axis=1)
        else:
            u = pd.Series([], dtype='string')
        usable = u[u.str.len() >= 8]
        result = {
            'format': 'csv',
            'rows': int(len(df)),
            'valid_url_rows': int(len(usable)),
            'unique_normalized_urls': int(usable.drop_duplicates().shape[0]),
        }
        if 'label' in df.columns:
            labels = pd.to_numeric(df['label'], errors='coerce')
            result['label_counts'] = {str(int(k)): int(v) for k, v in labels.dropna().value_counts().sort_index().items()}
        if 'source' in df.columns:
            result['source_counts'] = {str(k): int(v) for k, v in df['source'].fillna('unknown').astype(str).value_counts().sort_index().items()}
        ts_col = next((c for c in ('collected_at', 'observed_at', 'created_at', 'timestamp', 'date', 'datetime') if c in df.columns), None)
        if ts_col:
            ts = pd.to_datetime(df[ts_col], errors='coerce', utc=True).dropna()
            result['timestamp_column'] = ts_col
            result['timestamp_valid_rows'] = int(len(ts))
            result['timestamp_distinct_values'] = int(ts.nunique())
            result['timestamp_min'] = ts.min().isoformat() if len(ts) else None
            result['timestamp_max'] = ts.max().isoformat() if len(ts) else None
        return result
    except Exception as exc:
        return {'rows': 0, 'valid_url_rows': 0, 'unique_normalized_urls': 0,
                'inspection_error': f'{type(exc).__name__}: {exc}'}


def main():
    ap=argparse.ArgumentParser(description='Build a provenance manifest for real training/evaluation files.')
    ap.add_argument('files',nargs='+')
    ap.add_argument('--out',default='data/manifests/dataset_manifest.json')
    args=ap.parse_args()
    items=[]
    for s in args.files:
        p=Path(s)
        stats = inspect_dataset(p)
        items.append({'path':str(p),'sha256':sha256(p),'bytes':p.stat().st_size,**stats})
    payload={'generated_at':datetime.now(timezone.utc).isoformat(),'items':items,'synthetic_data_declared':False,'note':'Hashes and counts describe captured files; provenance and label semantics must be reviewed separately.'}
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
