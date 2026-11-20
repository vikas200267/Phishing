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

from app.train import domain_groups, url_series


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Build a reproducible, balanced training mix from the full eligible LegitPhish v2 training split.'
    )
    parser.add_argument('--legitphish', type=Path, default=ROOT/'data/training/legitphish_v2_train.csv')
    parser.add_argument('--production', type=Path, default=ROOT/'data/training/production_training.csv')
    parser.add_argument('--out', type=Path, default=ROOT/'data/realworld/evaluations/phishvn_temporal/combined_extra_full_legitphish.csv')
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    if not args.legitphish.exists() or not args.production.exists():
        raise FileNotFoundError('Both the prepared LegitPhish v2 training split and production training CSV are required.')

    legit = pd.read_csv(args.legitphish, low_memory=False)
    if not {'url', 'label'}.issubset(legit.columns):
        raise ValueError('LegitPhish training split must contain url,label columns.')
    legit['label'] = pd.to_numeric(legit['label'], errors='coerce')
    legit = legit[legit['label'].isin([0, 1]) & legit['url'].notna()].copy()
    legit['url'] = url_series(legit)
    legit['_normalized_url'] = legit['url']
    legit['_domain'] = domain_groups(legit)
    legit = legit[legit['url'].str.len().ge(8) & legit['_domain'].str.len().gt(0)]

    # Balance by deterministic URL hash, not dataset row order. Keep every
    # legitimate example and sample the same number of phishing examples.
    class_counts = legit['label'].astype(int).value_counts().to_dict()
    target = min(int(class_counts.get(0, 0)), int(class_counts.get(1, 0)))
    legit['_sample_key'] = legit['url'].map(lambda value: hashlib.sha256(value.encode('utf-8')).hexdigest())
    balanced = pd.concat([
        legit[legit['label'].astype(int) == label].sort_values('_sample_key').head(target)
        for label in (0, 1)
    ], ignore_index=True)
    legit_out = balanced[['url', 'label']].copy()
    legit_out['source'] = 'LegitPhish_v2_full_balanced_train'
    legit_out['observed_at'] = ''  # The source has no authentic per-URL event dates.

    production = pd.read_csv(args.production, low_memory=False)
    if not {'url', 'label'}.issubset(production.columns):
        raise ValueError('Production training CSV must contain url,label columns.')
    production['url'] = url_series(production)
    production['label'] = pd.to_numeric(production['label'], errors='coerce')
    production = production[production['label'].isin([0, 1]) & production['url'].str.len().ge(8)].copy()
    if 'source' not in production:
        production['source'] = 'original_production_training'
    if 'observed_at' not in production:
        production['observed_at'] = ''
    production = production[['url', 'label', 'source', 'observed_at']]

    combined = pd.concat([production, legit_out], ignore_index=True)
    conflicts = combined.groupby('url')['label'].nunique()
    conflict_urls = set(conflicts[conflicts > 1].index)
    conflict_rows = int(combined['url'].isin(conflict_urls).sum())
    combined = combined[~combined['url'].isin(conflict_urls)].drop_duplicates(['url', 'label'])
    combined['label'] = combined['label'].astype(int)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)

    manifest_path = args.manifest or args.out.with_suffix('.meta.json')
    metadata = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'dataset_role': 'candidate_training_only',
        'source_dataset': 'LegitPhish Dataset, Version 2',
        'source_url': 'https://data.mendeley.com/datasets/hx4m73v2sf/2',
        'license': 'CC BY 4.0',
        'authors': ['Rachana Potpelwar', 'Uday Kulkarni', 'Jaishri Waghmare'],
        'source_timestamp_status': 'no authentic per-URL timestamps; observed_at intentionally blank',
        'sampling': 'All legitimate rows plus deterministic SHA-256(URL)-ordered phishing sample equal in size to the legitimate class.',
        'balance_seed': 'SHA-256(normalized URL); no random state required',
        'counts_before_sampling': {str(int(k)): int(v) for k, v in sorted(class_counts.items())},
        'balanced_legitphish_rows': int(len(legit_out)),
        'production_training_rows': int(len(production)),
        'conflicting_url_rows_removed': conflict_rows,
        'output_rows': int(len(combined)),
        'output_label_counts': {str(int(k)): int(v) for k, v in combined['label'].value_counts().sort_index().items()},
        'output_sha256': sha256(args.out),
        'inputs': {
            'legitphish_train': {'path': str(args.legitphish.relative_to(ROOT)), 'sha256': sha256(args.legitphish)},
            'production_training': {'path': str(args.production.relative_to(ROOT)), 'sha256': sha256(args.production)},
        },
        'warning': 'Cross-source mixture is not temporally qualified; training and serving metrics must be evaluated independently.',
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
