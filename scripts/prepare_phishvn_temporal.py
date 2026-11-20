from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.features import normalize_url
from app.train import domain_groups

SOURCE = ROOT / 'data/realworld/external/phishvn/data/splits'
OUT = ROOT / 'data/realworld/evaluations/phishvn_temporal'
DATASET_URL = 'https://data.mendeley.com/datasets/b97hxbxtpd/5'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_split(name: str) -> pd.DataFrame:
    frame = pd.read_csv(SOURCE / f'url_{name}.csv', low_memory=False)
    # collected_at is the source event date; scraped_at is retrieval time and
    # must never be substituted for missing event time.
    frame['event_time'] = pd.to_datetime(
        frame['collected_at'].astype('string'), errors='coerce',
        format='mixed', dayfirst=True, utc=True,
    )
    frame = frame[
        frame['event_time'].notna()
        & frame['tier'].isin(['gold', 'silver'])
        & frame['label'].isin(['benign', 'phishing'])
        & frame['url'].notna()
    ].copy()
    frame['url'] = frame['url'].map(lambda value: normalize_url(str(value)))
    frame = frame[frame.url.str.startswith(('http://', 'https://'), na=False)]
    frame['label'] = frame.label.map({'benign': 0, 'phishing': 1}).astype('int8')
    conflicts = frame.groupby('url', dropna=False).label.nunique()
    conflicts = conflicts[conflicts > 1]
    if len(conflicts):
        raise ValueError(f'PhishVN {name} split contains {len(conflicts)} URLs with conflicting labels.')
    frame = frame.drop_duplicates(subset=['url'], keep='first')
    return frame[['url', 'label', 'event_time', 'domain', 'source', 'tier', 'label_source']]


def main() -> int:
    parser = argparse.ArgumentParser(description='Build a source-attested PhishVN URL temporal evaluation dataset using only official group-aware splits and collected_at event dates.')
    parser.add_argument('--cutoff', default='2023-01-01')
    parser.add_argument('--out-dir', type=Path, default=OUT)
    args = parser.parse_args()
    cutoff = pd.Timestamp(args.cutoff, tz='UTC')
    train = load_split('train')
    test = load_split('test')
    # Preserve chronological integrity even if the upstream group-aware train
    # fold contains later dated rows. The untouched official test fold supplies
    # held-out future records; validation is never used for threshold tuning.
    train = train[train.event_time < cutoff].copy()
    test = test[test.event_time >= cutoff].copy()
    if len(train) < 200 or len(test) < 200:
        raise ValueError(f'Insufficient temporal support at {cutoff}: train={len(train)}, test={len(test)}')
    if set(train.label.unique()) != {0, 1} or set(test.label.unique()) != {0, 1}:
        raise ValueError('Both classes must be present in both temporal partitions.')

    train_groups = domain_groups(pd.DataFrame({'_normalized_url': train.url})).astype(str)
    test_groups = domain_groups(pd.DataFrame({'_normalized_url': test.url})).astype(str)
    overlap = set(train_groups) & set(test_groups)
    if overlap:
        raise ValueError(f'Upstream official group split unexpectedly overlaps {len(overlap)} registrable domains.')
    if set(train.url) & set(test.url):
        raise ValueError('Exact URL overlap in the chronological split.')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_out, test_out = train.copy(), test.copy()
    train_out['registrable_domain'] = train_groups.to_numpy()
    test_out['registrable_domain'] = test_groups.to_numpy()
    train_out['event_time'] = train_out.event_time.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    test_out['event_time'] = test_out.event_time.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    train_out.to_csv(args.out_dir / 'train.csv', index=False)
    test_out.to_csv(args.out_dir / 'test.csv', index=False)
    combined = pd.concat([train_out, test_out], ignore_index=True).sort_values('event_time')
    dataset_path = args.out_dir / 'dataset.csv'
    combined.to_csv(dataset_path, index=False)
    metadata = {
        'dataset': 'PhishVN v5, official URL train/test folds, gold/silver labels only',
        'source_url': DATASET_URL,
        'doi': '10.17632/b97hxbxtpd.5',
        'license': 'CC BY 4.0; attribution required; source data is retained locally and ignored by git',
        'attribution': 'Thai Nguyen Vu; upstream sources: NCSC Tin Nhiem Mang, ChongLuaDao, and Tranco',
        'timestamp_column': 'event_time',
        'timestamp_provenance': 'source_attested',
        'timestamp_semantics': 'source-attested collection date of detection or organization certification; scraped_at is retrieval time and excluded',
        'timestamp_timezone': 'UTC date normalization; source supplies calendar dates without time-of-day',
        'cutoff': cutoff.isoformat(),
        'split_method': 'official PhishVN registrable-domain group-aware folds; train additionally restricted to event_time before cutoff; test restricted to event_time on/after cutoff',
        'label_mapping': {'0': 'benign', '1': 'phishing'},
        'tier_filter': ['gold', 'silver'],
        'train_rows': len(train_out),
        'test_rows': len(test_out),
        'train_class_counts': {str(k): int(v) for k, v in train.label.value_counts().sort_index().items()},
        'test_class_counts': {str(k): int(v) for k, v in test.label.value_counts().sort_index().items()},
        'train_domain_count': int(train_groups.nunique()),
        'test_domain_count': int(test_groups.nunique()),
        'registrable_domain_overlap': len(overlap),
        'exact_url_overlap': len(set(train.url) & set(test.url)),
        'train_start': str(train.event_time.min()),
        'train_end': str(train.event_time.max()),
        'test_start': str(test.event_time.min()),
        'test_end': str(test.event_time.max()),
        'source_files_sha256': {name: sha256(SOURCE / f'url_{name}.csv') for name in ('train', 'test')},
        'dataset_sha256': sha256(dataset_path),
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'listed_urls_visited': False,
        'production_model_trained_on_this_data': False,
    }
    dataset_path.with_suffix('.csv.meta.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (args.out_dir / 'report.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
