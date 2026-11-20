from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA = ROOT / 'data/realworld/external/deepurlbench'
OUT = ROOT / 'data/realworld/evaluations/deepurlbench_temporal'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description='Prepare an authentic-date, source-held-out URL benchmark and train a research-only temporal baseline.')
    parser.add_argument('--input-dir', type=Path, default=DATA)
    parser.add_argument('--out-dir', type=Path, default=OUT)
    parser.add_argument('--cutoff', default='2022-01-01')
    parser.add_argument('--per-row-group', type=int, default=1600, help='Maximum rows per group, sampled at the corpus prevalence for benign/phishing labels.')
    parser.add_argument('--seed', type=int, default=20261118)
    args = parser.parse_args()
    files = [args.input_dir / 'deepurlbench_0000.parquet', args.input_dir / 'deepurlbench_0001.parquet']
    if not all(path.exists() for path in files):
        raise FileNotFoundError('Run scripts/download_deepurlbench.py first.')
    if args.per_row_group < 200:
        parser.error('--per-row-group must be at least 200')
    cutoff = pd.Timestamp(args.cutoff)
    samples = []
    scanned_rows = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        for row_group_idx in range(parquet.metadata.num_row_groups):
            frame = parquet.read_row_group(row_group_idx, columns=['url', 'first_seen', 'label']).to_pandas()
            scanned_rows += len(frame)
            frame = frame[frame['label'].isin(['benign', 'phishing']) & frame['url'].notna() & frame['first_seen'].notna()].copy()
            frame['label'] = (frame['label'] == 'phishing').astype('int8')
            frame['first_seen'] = pd.to_datetime(frame['first_seen'], errors='coerce')
            caps = {0: int(args.per_row_group * 0.682), 1: args.per_row_group - int(args.per_row_group * 0.682)}
            for label in (0, 1):
                group = frame[frame.label == label]
                if len(group) > caps[label]:
                    group = group.sample(caps[label], random_state=args.seed + row_group_idx + label)
                if len(group):
                    samples.append(group[['url', 'label', 'first_seen']])
    if not samples:
        raise RuntimeError('No labeled rows with source first_seen values were available.')
    data = pd.concat(samples, ignore_index=True)
    data['url'] = data.url.astype(str).str.strip()
    data = data[data.url.str.match(r'^https?://', case=False, na=False)]
    data = data.sort_values('first_seen').drop_duplicates(subset=['url'], keep='first').reset_index(drop=True)
    data['registrable_domain'] = data.url.map(registrable_domain)
    train = data[data.first_seen < cutoff].copy()
    time_test = data[data.first_seen >= cutoff].copy()
    train_domains = set(train.registrable_domain)
    test = time_test[~time_test.registrable_domain.isin(train_domains)].copy()
    for name, frame in (('train', train), ('test', test)):
        if len(frame) < 200 or set(frame.label.unique()) != {0, 1}:
            raise RuntimeError(f'{name} split needs at least 200 rows and both labels; got {len(frame)} rows, labels={sorted(frame.label.unique())}')
    train_urls = set(train.url)
    test_urls = set(test.url)
    overlap = train_urls & test_urls
    if overlap:
        raise RuntimeError(f'Exact URL leakage across chronological split: {len(overlap)}')

    # Set the operating threshold from a chronological calibration slice that
    # remains inside the pre-cutoff training period; never tune on temporal test.
    train = train.sort_values('first_seen').reset_index(drop=True)
    calibration_start = max(1, int(len(train) * 0.8))
    threshold_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=2, max_features=120_000, sublinear_tf=True, lowercase=True, dtype=np.float32)
    x_fit = threshold_vectorizer.fit_transform(train.url.iloc[:calibration_start])
    x_calibration = threshold_vectorizer.transform(train.url.iloc[calibration_start:])
    threshold_model = LogisticRegression(max_iter=300, class_weight='balanced', solver='liblinear', random_state=args.seed)
    threshold_model.fit(x_fit, train.label.iloc[:calibration_start])
    calibration_scores = threshold_model.predict_proba(x_calibration)[:, 1]
    calibration_labels = train.label.iloc[calibration_start:].to_numpy()
    calibration_fpr, calibration_tpr, calibration_thresholds = roc_curve(calibration_labels, calibration_scores)
    allowed = calibration_fpr <= 0.05
    if not allowed.any():
        raise RuntimeError('Could not select an operating threshold under the 5% calibration FPR limit.')
    best_idx = int(np.argmax(np.where(allowed, calibration_tpr, -1.0)))
    threshold = float(calibration_thresholds[best_idx])

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=2, max_features=120_000, sublinear_tf=True, lowercase=True, dtype=np.float32)
    x_train = vectorizer.fit_transform(train.url)
    x_test = vectorizer.transform(test.url)
    model = LogisticRegression(max_iter=300, class_weight='balanced', solver='liblinear', random_state=args.seed)
    model.fit(x_train, train.label)
    scores = model.predict_proba(x_test)[:, 1]
    predictions = (scores >= threshold).astype('int8')
    tn, fp, fn, tp = confusion_matrix(test.label, predictions, labels=[0, 1]).ravel()
    metrics = {
        'accuracy': float(accuracy_score(test.label, predictions)),
        'precision': float(precision_score(test.label, predictions, zero_division=0)),
        'recall': float(recall_score(test.label, predictions, zero_division=0)),
        'roc_auc': float(roc_auc_score(test.label, scores)),
        'false_positive_rate': float(fp / max(1, fp + tn)),
        'false_negative_rate': float(fn / max(1, fn + tp)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'threshold': threshold,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (('train', train), ('test', test)):
        output = frame[['url', 'label', 'first_seen', 'registrable_domain']].sort_values('first_seen')
        output.to_csv(args.out_dir / f'{name}.csv', index=False)
    prediction_output = test[['url', 'label', 'first_seen', 'registrable_domain']].copy()
    prediction_output['score'] = scores
    prediction_output['prediction'] = predictions
    prediction_output = prediction_output.sort_values('first_seen')
    prediction_output.to_csv(args.out_dir / 'predictions.csv', index=False)
    report = {
        'dataset': 'DeepInstinct/DeepURLBench',
        'source_url': 'https://huggingface.co/datasets/DeepInstinct/DeepURLBench',
        'license': 'CC BY-NC 4.0; independent research benchmark only; do not use to train or ship the production model or redistribute source rows',
        'timestamp_column': 'first_seen',
        'timestamp_semantics': 'source-provided first-observed time; timezone unspecified in the dataset schema',
        'cutoff': str(cutoff),
        'sampling': {'method': 'deterministic per-row-group sample allocated at source prevalence', 'seed': args.seed, 'cap_per_row_group': args.per_row_group, 'rows_scanned': scanned_rows, 'rows_sampled_after_url_dedup': len(data)},
        'train': split_summary(train),
        'test': split_summary(test),
        'time_test_before_domain_exclusion': split_summary(time_test),
        'exact_url_overlap': len(overlap),
        'registrable_domain_overlap': len(set(train.registrable_domain) & set(test.registrable_domain)),
        'test_rows_excluded_for_training_domain_overlap': len(time_test) - len(test),
        'temporal_baseline': {'family': 'character 3–5 gram TF-IDF + class-balanced logistic regression', 'trained_only_before_cutoff': True, 'calibration': {'period_start': str(train.first_seen.iloc[calibration_start]), 'period_end': str(train.first_seen.iloc[-1]), 'threshold_target': 'maximum recall at calibration FPR <= 5%', 'threshold': threshold, 'achieved_fpr': float(calibration_fpr[best_idx]), 'achieved_recall': float(calibration_tpr[best_idx])}, 'metrics': metrics},
        'production_model_temporal_status': 'not_evaluated_by_this_research_baseline; requires a production-model artifact trained strictly before the cutoff to claim production temporal readiness',
        'source_parquet': [{'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)} for path in files],
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'listed_urls_visited': False,
        'production_training_use': False,
    }
    combined = data[['url', 'label', 'first_seen', 'registrable_domain']].sort_values('first_seen')
    combined_path = args.out_dir / 'dataset.csv'
    combined.to_csv(combined_path, index=False)
    combined_meta = {
        'dataset': report['dataset'],
        'source_url': report['source_url'],
        'license': report['license'],
        'timestamp_column': 'first_seen',
        'timestamp_provenance': 'source_provided',
        'timestamp_semantics': report['timestamp_semantics'],
        'timestamp_timezone': 'unspecified in source schema',
        'label_column': 'label',
        'label_mapping': {'0': 'benign', '1': 'phishing'},
        'cutoff': str(cutoff),
        'sampling': report['sampling'],
        'source_parquet': report['source_parquet'],
        'rows': len(combined),
        'sha256': sha256(combined_path),
        'listed_urls_visited': False,
        'production_training_use': False,
    }
    combined_path.with_suffix(combined_path.suffix + '.meta.json').write_text(json.dumps(combined_meta, indent=2), encoding='utf-8')
    report['prepared_dataset'] = {'path': str(combined_path.relative_to(ROOT)), 'sha256': combined_meta['sha256']}
    (args.out_dir / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0


def registrable_domain(url: str) -> str:
    try:
        host = (urlsplit(url).hostname or '').lower().strip('.')
    except ValueError:
        return ''
    if not host:
        return ''
    labels = host.split('.')
    if len(labels) > 2:
        from app.features import TWO_PART_SUFFIXES
        if '.'.join(labels[-2:]) in TWO_PART_SUFFIXES:
            return '.'.join(labels[-3:])
    return '.'.join(labels[-2:]) if len(labels) >= 2 else host


def split_summary(frame: pd.DataFrame) -> dict:
    return {
        'rows': len(frame),
        'class_counts': {str(key): int(value) for key, value in frame.label.value_counts().sort_index().items()},
        'timestamp_start': str(frame.first_seen.min()),
        'timestamp_end': str(frame.first_seen.max()),
        'timestamp_coverage': float(frame.first_seen.notna().mean()),
        'unique_registrable_domains': int(frame.registrable_domain.nunique()),
    }


if __name__ == '__main__':
    raise SystemExit(main())
