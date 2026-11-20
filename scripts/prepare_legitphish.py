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
from app.policy import valid_http_url
from app.train import domain_groups, url_series

RAW_DEFAULT = ROOT / 'data/realworld/external/legitphish_v2.csv'
TRAIN_DEFAULT = ROOT / 'data/training/legitphish_v2_train.csv'
HOLDOUT_DEFAULT = ROOT / 'data/realworld/evaluations/legitphish_v2_holdout.csv'
EXCLUSIONS_DEFAULT = ROOT / 'data/realworld/evaluations/legitphish_v2_exclusions.csv'
CHALLENGE_DEFAULT = ROOT / 'data/realworld/legitimate_challenge.csv'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def file_hash_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description='Normalize and domain-split the manually labeled LegitPhish v2 CSV.')
    ap.add_argument('--raw', type=Path, default=RAW_DEFAULT)
    ap.add_argument('--train-out', type=Path, default=TRAIN_DEFAULT)
    ap.add_argument('--holdout-out', type=Path, default=HOLDOUT_DEFAULT)
    ap.add_argument('--exclusions-out', type=Path, default=EXCLUSIONS_DEFAULT)
    ap.add_argument('--challenge', type=Path, default=CHALLENGE_DEFAULT)
    ap.add_argument('--holdout-modulus', type=int, default=5, help='One residue out of this many registrable domains is reserved for evaluation.')
    args = ap.parse_args()
    if args.holdout_modulus < 2:
        raise ValueError('--holdout-modulus must be at least 2')

    raw = pd.read_csv(args.raw)
    if not {'URL', 'ClassLabel'}.issubset(raw.columns):
        raise ValueError('LegitPhish v2 needs URL and ClassLabel columns')
    raw_count = len(raw)
    raw = raw[['URL', 'ClassLabel']].copy()
    raw['ClassLabel'] = pd.to_numeric(raw['ClassLabel'], errors='coerce')
    raw = raw[raw['ClassLabel'].isin([0, 1]) & raw['URL'].notna()].copy()
    labeled_rows = len(raw)
    # Source coding is 0=phishing, 1=legitimate; PhishGuard coding is 1=phishing.
    raw['label'] = 1 - raw['ClassLabel'].astype(int)
    raw['url'] = raw['URL'].astype(str).map(normalize_url)
    valid_mask = raw['url'].map(valid_http_url)
    invalid_url_rows = int((~valid_mask).sum())
    raw = raw[valid_mask & raw['url'].str.len().ge(8)].copy()
    valid_url_rows = len(raw)
    raw['_normalized_url'] = url_series(raw)

    # Remove contradictory normalized URL labels before deduplication.
    conflicts = raw.groupby('_normalized_url')['label'].nunique()
    conflicting_urls = set(conflicts[conflicts > 1].index)
    conflicting_rows = int(raw['_normalized_url'].isin(conflicting_urls).sum())
    raw = raw[~raw['_normalized_url'].isin(conflicting_urls)].copy()
    before_dedup = len(raw)
    raw = raw.drop_duplicates('_normalized_url').copy()
    exact_duplicate_rows = before_dedup - len(raw)
    raw['domain'] = domain_groups(raw)
    raw = raw[raw['domain'].astype(str).str.len() > 0].copy()

    protected_domains: set[str] = set()
    production = ROOT / 'data/training/production_training.csv'
    if production.exists():
        p = pd.read_csv(production)
        p['_normalized_url'] = url_series(p)
        protected_domains.update(domain_groups(p).astype(str))
    challenge_urls: list[str] = []
    reserved_eval_urls: set[str] = set()
    if args.challenge.exists():
        challenge = pd.read_csv(args.challenge)
        if 'url' in challenge:
            challenge_urls = challenge['url'].dropna().astype(str).tolist()
            c = pd.DataFrame({'_normalized_url': [normalize_url(u) for u in challenge_urls]})
            protected_domains.update(domain_groups(c).astype(str))
    evaluation_inputs = [
        ROOT/'data/release_gate_v1.csv',
        ROOT/'data/frozen_regression.csv',
        ROOT/'data/historical_regression_v1.csv',
        ROOT/'data/realworld/archive/openphish_20261005_unseen.csv',
    ]
    for eval_path in evaluation_inputs:
        if not eval_path.exists():
            continue
        evaluation = pd.read_csv(eval_path)
        eval_urls = url_series(evaluation).dropna().astype(str).tolist()
        reserved_eval_urls.update(eval_urls)
        eval_frame = pd.DataFrame({'_normalized_url': eval_urls})
        protected_domains.update(domain_groups(eval_frame).astype(str))

    # Keep the reference set independent of domains already in the active
    # release or in the explicitly reported legitimate homepage challenge.
    before_protected = len(raw)
    raw = raw[~raw['domain'].isin(protected_domains)].copy()
    excluded_protected_rows = before_protected - len(raw)

    def is_holdout(domain: str) -> bool:
        digest = hashlib.sha256(domain.encode('utf-8')).digest()
        return int.from_bytes(digest[:8], 'big') % args.holdout_modulus == 0

    holdout_mask = raw['domain'].map(is_holdout)
    holdout = raw[holdout_mask].copy()
    training = raw[~holdout_mask].copy()

    # Every evaluation domain, plus the user's six named sites, is excluded
    # from all training sources by the main training CLI.
    exclusions = pd.concat([
        holdout[['url', 'label']],
        pd.DataFrame({'url': challenge_urls, 'label': 0}),
        pd.DataFrame({'url': sorted(reserved_eval_urls), 'label': -1}),
    ], ignore_index=True).drop_duplicates('url')

    def write_rows(frame: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame[['url', 'label']].assign(
            source='LegitPhish_v2', observed_at=''
        ).to_csv(path, index=False)

    write_rows(training, args.train_out)
    write_rows(holdout, args.holdout_out)
    args.exclusions_out.parent.mkdir(parents=True, exist_ok=True)
    exclusions.to_csv(args.exclusions_out, index=False)
    generated = datetime.now(timezone.utc).isoformat()
    common = {
        'source_dataset': 'LegitPhish Dataset, Version 2',
        'doi': '10.17632/hx4m73v2sf.2',
        'source_page': 'https://data.mendeley.com/datasets/hx4m73v2sf/2',
        'license': 'CC BY 4.0',
        'authors': ['Rachana Potpelwar', 'Uday Kulkarni', 'Jaishri Waghmare'],
        'source_label_mapping': {'0': 'phishing', '1': 'legitimate'},
        'phishguard_label_mapping': {'0': 'legitimate', '1': 'phishing'},
        'source_published_at': '2025-05-22',
        'source_row_count': raw_count,
        'labeled_rows': int(labeled_rows),
        'valid_http_url_rows': int(valid_url_rows),
        'invalid_http_urls_removed': int(invalid_url_rows),
        'cleaned_rows_before_domain_filter': int(len(raw)),
        'removed_for_active_release_or_evaluation_domain_overlap': int(excluded_protected_rows),
        'removed_conflicting_normalized_url_groups': int(len(conflicting_urls)),
        'removed_rows_in_conflicting_groups': int(conflicting_rows),
        'removed_exact_duplicate_rows': int(exact_duplicate_rows),
        'independent_evaluation_sources_reserved': [str(p.relative_to(ROOT)) for p in evaluation_inputs if p.exists()],
        'observed_at_is_blank_because_source_has_no_per_url_timestamp': True,
        'listed_urls_visited': False,
        'split_method': f'sha256(registrable_domain) mod {args.holdout_modulus}; one residue reserved for evaluation',
        'holdout_domains_are_removed_from_candidate_training_via_exclusions_file': str(args.exclusions_out.relative_to(ROOT)),
        'generated_at': generated,
    }
    for path, role, frame in [
        (args.train_out, 'training_only', training),
        (args.holdout_out, 'domain_disjoint_evaluation_only', holdout),
        (args.exclusions_out, 'reserved_evaluation_domains_and_user_challenge', exclusions),
    ]:
        meta = {
            **common,
            'dataset_role': role,
            'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            'rows': int(len(frame)),
            'unique_registrable_domains': int(frame['domain'].nunique()) if 'domain' in frame else None,
            'label_counts': {str(k): int(v) for k, v in frame['label'].value_counts().items()},
            'sha256': file_hash_bytes(path),
        }
        path.with_suffix(path.suffix + '.meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    print(json.dumps({
        'raw_rows': raw_count,
        'training_rows': len(training),
        'training_domains': int(training.domain.nunique()),
        'holdout_rows': len(holdout),
        'holdout_domains': int(holdout.domain.nunique()),
        'holdout_phishing_rows': int((holdout.label == 1).sum()),
        'holdout_legitimate_rows': int((holdout.label == 0).sum()),
        'excluded_production_or_challenge_rows': excluded_protected_rows,
        'invalid_url_rows_removed': invalid_url_rows,
        'conflicting_urls_removed': len(conflicting_urls),
        'duplicate_urls_removed': exact_duplicate_rows,
        'exclusion_rows': len(exclusions),
        'outputs': [str(args.train_out), str(args.holdout_out), str(args.exclusions_out)],
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
