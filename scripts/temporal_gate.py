from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate source-attested URL timestamps, temporal class support, and exact URL split integrity.')
    parser.add_argument('csv')
    parser.add_argument('--timestamp-col', default='')
    parser.add_argument('--source-metadata', default='', help='Sidecar with source timestamp semantics and provenance; required to pass.')
    parser.add_argument('--cutoff', default='', help='Explicit chronological cutoff; if omitted, use the 80th percentile timestamp.')
    parser.add_argument('--min-rows', type=int, default=200)
    parser.add_argument('--min-class-rows', type=int, default=50)
    parser.add_argument('--out', default='data/manifests/temporal_gate.json')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path, low_memory=False)
    metadata_path = Path(args.source_metadata) if args.source_metadata else csv_path.with_suffix(csv_path.suffix + '.meta.json')
    try:
        source_meta = json.loads(metadata_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        source_meta = {}
    candidates = [args.timestamp_col] if args.timestamp_col else ['first_seen', 'submission_time', 'verification_time', 'timestamp', 'created_at', 'date', 'datetime', 'observed_at', 'collected_at']
    chosen = None
    times = None
    for column in candidates:
        if column and column in df.columns:
            parsed = pd.to_datetime(df[column], errors='coerce', utc=True)
            if parsed.notna().sum() >= max(args.min_rows, int(0.7 * len(df))) and parsed.dropna().nunique() > 1:
                chosen, times = column, parsed
                break

    timestamp_provenance = source_meta.get('timestamp_provenance')
    timestamp_semantics = str(source_meta.get('timestamp_semantics') or '')
    provenance_ok = timestamp_provenance in {'source_provided', 'source_attested'} and any(term in timestamp_semantics.lower() for term in ('first-seen', 'first seen', 'first observed', 'first-observed', 'submission time', 'verification time', 'collection date', 'collected_at'))
    report = {
        'dataset': str(csv_path),
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'source_metadata': str(metadata_path),
        'source_dataset': source_meta.get('dataset'),
        'source_url': source_meta.get('source_url'),
        'license': source_meta.get('license'),
        'timestamp_column': chosen,
        'timestamp_provenance': timestamp_provenance,
        'timestamp_semantics': timestamp_semantics or None,
        'timestamp_coverage': float(times.notna().mean()) if times is not None else 0.0,
        'rows': len(df),
        'status': 'not_ready',
        'model_release_status': 'not_evaluated',
        'checks': {
            'timestamp_column_coverage': bool(chosen),
            'source_timestamp_provenance': provenance_ok,
            'both_classes_in_train_and_test': False,
            'zero_exact_url_overlap': False,
            'zero_registrable_domain_overlap': False,
            'production_model_trained_before_cutoff': False,
        },
    }
    if chosen and times is not None:
        valid = times.notna()
        parsed_times = times[valid]
        if args.cutoff:
            cutoff = pd.Timestamp(args.cutoff)
            cutoff = cutoff.tz_localize('UTC') if cutoff.tzinfo is None else cutoff.tz_convert('UTC')
            train_mask = valid & times.lt(cutoff)
            test_mask = valid & times.ge(cutoff)
        else:
            ordered = parsed_times.sort_values()
            cut_index = max(1, min(len(ordered) - 1, int(len(ordered) * 0.8)))
            cutoff = ordered.iloc[cut_index]
            train_mask = valid & times.lt(cutoff)
            test_mask = valid & times.ge(cutoff)
        train, test = df[train_mask].copy(), df[test_mask].copy()
        report.update({
            'cutoff': str(cutoff),
            'train_rows': len(train),
            'test_rows': len(test),
            'train_start': str(times[train_mask].min()),
            'train_end': str(times[train_mask].max()),
            'test_start': str(times[test_mask].min()),
            'test_end': str(times[test_mask].max()),
            'train_class_counts': {},
            'test_class_counts': {},
        })
        both_classes = False
        if 'label' in df.columns:
            train_counts = train.label.astype(str).value_counts()
            test_counts = test.label.astype(str).value_counts()
            report['train_class_counts'] = {k: int(v) for k, v in train_counts.items()}
            report['test_class_counts'] = {k: int(v) for k, v in test_counts.items()}
            both_classes = len(train_counts) >= 2 and len(test_counts) >= 2 and min(train_counts) >= args.min_class_rows and min(test_counts) >= args.min_class_rows
        report['checks']['both_classes_in_train_and_test'] = both_classes
        exact_overlap = 0
        if 'url' in df.columns:
            train_urls = set(train.url.dropna().astype(str).str.strip())
            test_urls = set(test.url.dropna().astype(str).str.strip())
            exact_overlap = len(train_urls & test_urls)
        report['exact_url_overlap'] = exact_overlap
        report['checks']['zero_exact_url_overlap'] = exact_overlap == 0
        domain_overlap = None
        if 'registrable_domain' in df.columns:
            train_domains = set(train.registrable_domain.dropna().astype(str))
            test_domains = set(test.registrable_domain.dropna().astype(str))
            domain_overlap = len(train_domains & test_domains)
            report['registrable_domain_overlap'] = domain_overlap
        report['checks']['zero_registrable_domain_overlap'] = domain_overlap == 0
        all_data_checks = provenance_ok and both_classes and exact_overlap == 0 and domain_overlap == 0 and len(train) >= args.min_rows and len(test) >= args.min_rows
        report['status'] = 'pass' if all_data_checks else 'not_ready'
        report['message'] = None if all_data_checks else 'The source timestamps, split support, or leakage checks do not meet the temporal dataset gate.'

    report['interpretation'] = 'A pass validates this source-attested temporal dataset split. It does not establish temporal readiness of a production model unless that exact model was trained only on pre-cutoff data and evaluated on the held-out rows.'
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'pass' else 2


if __name__ == '__main__':
    raise SystemExit(main())
