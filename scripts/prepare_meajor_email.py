from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'data/realworld/external/meajor_cleaned_preprocessed.csv'
TRAIN = ROOT/'data/training/meajor_email_train.csv'
TEST = ROOT/'data/realworld/evaluations/meajor_email_temporal_test.csv'
CUTOFF = '2007-01-01T00:00:00+00:00'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description='Prepare a content-deduplicated temporal split of the timestamped MeAJOR email corpus.')
    ap.add_argument('--raw', type=Path, default=RAW)
    ap.add_argument('--train-out', type=Path, default=TRAIN)
    ap.add_argument('--test-out', type=Path, default=TEST)
    ap.add_argument('--cutoff', default=CUTOFF, help='UTC timestamp; rows at or after this time are evaluation-only.')
    ap.add_argument('--min-year', type=int, default=1998)
    ap.add_argument('--max-year', type=int, default=2010)
    a = ap.parse_args()

    df = pd.read_csv(a.raw, low_memory=False)
    required = {'date', 'subject', 'body', 'label', 'source'}
    if not required.issubset(df.columns):
        raise ValueError(f'MeAJOR dataset is missing required fields: {sorted(required-set(df.columns))}')
    raw_count = len(df)
    df['_label'] = pd.to_numeric(df.label, errors='coerce')
    unlabeled = int(df._label.isna().sum())
    df = df[df._label.isin([0, 1])].copy()
    df['label'] = df['_label'].astype(int)
    df['_timestamp'] = pd.to_datetime(df.date, errors='coerce', utc=True)
    df = df[df['_timestamp'].notna()].copy()
    df['_year'] = df['_timestamp'].dt.year
    valid_date_mask = df['_year'].between(a.min_year, a.max_year)
    invalid_date_rows = int((~valid_date_mask).sum())
    df = df[valid_date_mask].copy()

    def clean_text(v: object) -> str:
        if pd.isna(v):
            return ''
        return ' '.join(str(v).casefold().split())
    df['_content_hash'] = (df['subject'].map(clean_text) + '\n' + df['body'].map(clean_text)).map(
        lambda value: hashlib.sha256(value.encode('utf-8')).hexdigest()
    )
    conflicting = df.groupby('_content_hash')['label'].nunique()
    conflicting_hashes = set(conflicting[conflicting > 1].index)
    conflict_rows = int(df['_content_hash'].isin(conflicting_hashes).sum())
    df = df[~df['_content_hash'].isin(conflicting_hashes)].copy()
    before_dedup = len(df)
    df = df.drop_duplicates('_content_hash', keep='first').copy()
    duplicate_rows = before_dedup-len(df)

    cutoff = pd.to_datetime(a.cutoff, utc=True)
    train = df[df['_timestamp'] < cutoff].copy()
    test = df[df['_timestamp'] >= cutoff].copy()
    if train.empty or test.empty or train.label.nunique() != 2 or test.label.nunique() != 2:
        raise ValueError('Temporal split must retain both labels in train and test.')
    overlap = set(train['_content_hash']) & set(test['_content_hash'])
    if overlap:
        raise RuntimeError(f'Exact content leakage across temporal split: {len(overlap)}')

    out_cols = [c for c in ('sender','sender_domain','receiver_domain','date','subject','body','urls','url_count','label','source') if c in df.columns]
    for frame, path in ((train, a.train_out), (test, a.test_out)):
        path.parent.mkdir(parents=True, exist_ok=True)
        output = frame[out_cols].copy()
        output['observed_at'] = frame['_timestamp'].map(lambda d: d.isoformat())
        output.to_csv(path, index=False)

    generated = datetime.now(timezone.utc).isoformat()
    raw_hash = sha256(a.raw)
    meta_base = {
        'dataset': 'MeAJOR: Merged email Assets from Joint Open-source Repositories',
        'doi': '10.5281/zenodo.18471483',
        'source_url': 'https://zenodo.org/records/18471483',
        'license': 'CC BY 4.0 (attribution required)',
        'authors': ['Francisco Cardoso','João Vitorino','Paulo Mendes','Eva Maia','Isabel Praça'],
        'source_label_mapping': {'0':'benign','1':'phishing'},
        'raw_rows': raw_count,
        'unlabeled_rows_removed': unlabeled,
        'rows_with_unusable_or_out_of-range_dates_removed': invalid_date_rows,
        'conflicting_exact_content_rows_removed': conflict_rows,
        'exact_duplicate_content_rows_removed': duplicate_rows,
        'raw_sha256': raw_hash,
        'timestamp_field': 'date (RFC 2822 email message timestamp parsed as UTC)',
        'timestamp_year_policy': {'min':a.min_year,'max':a.max_year},
        'temporal_cutoff_exclusive_for_training': cutoff.isoformat(),
        'exact_content_hash_overlap': 0,
        'email_body_anonymized_by_source': True,
        'listed_urls_visited': False,
        'generated_at': generated,
        'known_source_scope': 'The v2 Zenodo file currently contains source values trec5, trec6, and trec7; results are specific to this released file.',
    }
    for frame, path, role in ((train,a.train_out,'training_only_before_cutoff'),(test,a.test_out,'temporal_evaluation_only_at_or_after_cutoff')):
        meta = {
            **meta_base,
            'dataset_role': role,
            'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            'rows': len(frame),
            'label_counts': {str(k):int(v) for k,v in frame.label.value_counts().sort_index().items()},
            'source_counts': {str(k):int(v) for k,v in frame.source.fillna('unknown').value_counts().sort_index().items()},
            'timestamp_coverage': float(frame['_timestamp'].notna().mean()),
            'timestamp_start': frame['_timestamp'].min().isoformat(),
            'timestamp_end': frame['_timestamp'].max().isoformat(),
            'sha256': sha256(path),
        }
        path.with_suffix(path.suffix+'.meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    summary = {
        'raw_rows':raw_count,'usable_rows':len(df),'train_rows':len(train),'test_rows':len(test),
        'invalid_date_rows_removed':invalid_date_rows,'exact_duplicates_removed':duplicate_rows,
        'train_labels':train.label.value_counts().sort_index().to_dict(),
        'test_labels':test.label.value_counts().sort_index().to_dict(),
        'train_domains':int(train.sender_domain.nunique()) if 'sender_domain' in train else None,
        'test_domains':int(test.sender_domain.nunique()) if 'sender_domain' in test else None,
        'outputs':[str(a.train_out),str(a.test_out)],
    }
    print(json.dumps(summary,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
