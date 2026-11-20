from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description='Create a deterministic, balanced compute-bounded LegitPhish v2 training sample.')
    parser.add_argument('--input', type=Path, default=ROOT/'data/training/legitphish_v2_train.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'data/training/legitphish_v2_train_sampled.csv')
    parser.add_argument('--per-class', type=int, default=10_000)
    args = parser.parse_args()
    if args.per_class < 1:
        raise ValueError('--per-class must be positive')

    source = pd.read_csv(args.input)
    if not {'url', 'label'}.issubset(source.columns):
        raise ValueError('Input CSV must contain url and label')
    source['label'] = pd.to_numeric(source.label, errors='raise').astype(int)
    source['_rank'] = source.url.astype(str).map(lambda url: hashlib.sha256(url.encode('utf-8')).hexdigest())
    sampled = pd.concat([
        group.sort_values('_rank').head(args.per_class)
        for _, group in source.groupby('label', sort=True)
    ], ignore_index=True).drop(columns='_rank').sort_values(['label', 'url']).reset_index(drop=True)
    if source.label.nunique() != 2 or set(sampled.label.unique()) != {0, 1}:
        raise ValueError('Input must contain both labels 0 and 1')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(args.output, index=False)
    source_meta_path = args.input.with_suffix(args.input.suffix+'.meta.json')
    source_meta = json.loads(source_meta_path.read_text(encoding='utf-8')) if source_meta_path.exists() else {}
    metadata = {
        **source_meta,
        'dataset_role': 'candidate_training_deterministic_balanced_sample',
        'source_path': str(args.input.relative_to(ROOT)) if args.input.is_relative_to(ROOT) else str(args.input),
        'source_sha256': sha256(args.input),
        'sampling': f'up to {args.per_class} rows per class by ascending SHA-256(normalized URL); deterministic',
        'rows': int(len(sampled)),
        'label_counts': {str(label): int(count) for label, count in sampled.label.value_counts().sort_index().items()},
        'sha256': sha256(args.output),
    }
    args.output.with_suffix(args.output.suffix+'.meta.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'rows': len(sampled), 'label_counts': metadata['label_counts'], 'sha256': metadata['sha256']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
