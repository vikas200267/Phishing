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

from app.policy import valid_http_url
from app.train import evaluate, url_series
from app.url_model import load_bundle


DATASETS = {
    'frozen_regression': ROOT/'data/frozen_regression.csv',
    'homepage_challenge': ROOT/'data/realworld/legitimate_challenge.csv',
    'phishvn_post_cutoff_holdout': ROOT/'data/realworld/evaluations/phishvn_temporal/test.csv',
    'legitphish_domain_holdout': ROOT/'data/realworld/evaluations/legitphish_v2_holdout.csv',
    'openphish_20261005_archive': ROOT/'data/realworld/archive/openphish_20261005_unseen.csv',
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def score(model, path: Path, chunk_size: int) -> dict:
    frame = pd.read_csv(path, low_memory=False)
    if 'label' not in frame.columns or not any(column in frame.columns for column in ('url', '_normalized_url', 'domain_name')):
        raise ValueError(f'{path} must contain labels and URL fields')
    frame = frame[frame.label.isin([0, 1])].copy()
    normalized = url_series(frame)
    valid = normalized.map(valid_http_url)
    invalid = int((~valid).sum())
    frame = frame.loc[valid]
    urls = normalized.loc[valid].tolist()
    probabilities = []
    for start in range(0, len(urls), chunk_size):
        probabilities.extend(model.predict_proba(urls[start:start+chunk_size]).astype(float).tolist())
    labels = pd.to_numeric(frame.label, errors='raise').astype(int).to_numpy()
    metrics = evaluate(pd.Series(probabilities).to_numpy(), labels, float(model.threshold))
    metrics.update({
        'rows_in_source': int(len(pd.read_csv(path, usecols=['label']).query('label in [0, 1]'))),
        'valid_http_rows_scored': int(len(urls)),
        'invalid_http_rows_excluded': invalid,
        'source_sha256': sha256(path),
    })
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description='Reproduce cross-source evaluations for unpromoted URL challengers; listed URLs are never visited.')
    parser.add_argument(
        '--models',
        default=(
            'v20261118_143528,v20261118_144004,v20261119_063145,'
            'v20261119_063844,v20261119_064457,v20261119_083010'
        ),
        help='Comma-separated challenger versions (defaults to all cross-source candidates).',
    )
    parser.add_argument('--chunk-size', type=int, default=512)
    parser.add_argument('--out', type=Path, default=ROOT/'data/realworld/evaluations/crosssource_candidate_metrics.json')
    args = parser.parse_args()
    if args.chunk_size < 1:
        raise ValueError('--chunk-size must be positive')
    results = {}
    for version in [item.strip() for item in args.models.split(',') if item.strip()]:
        model_dir = ROOT/'models'/version
        model = load_bundle(model_dir/'model.joblib')
        metadata_path = model_dir/'metadata.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        results[version] = {
            'threshold': float(model.threshold),
            'promotion_eligible': bool(metadata.get('promotion_eligible')),
            'training_data_sources': metadata.get('data_sources'),
            'strict_temporal_qualification': False,
            'strict_temporal_reason': 'The candidate includes source rows without authentic per-URL timestamps (and may include data collected after the PhishVN cutoff).',
            'evaluations': {name: score(model, path, args.chunk_size) for name, path in DATASETS.items() if path.exists()},
        }
    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'method': 'Model scores measured on frozen regression, legitimate homepage challenge, source-attested PhishVN post-cutoff holdout, the separate LegitPhish v2 domain holdout, and the captured OpenPhish archive. Thresholds are the model artifacts’ frozen thresholds; they were not tuned on these evaluation sets.',
        'source_attribution': {
            'phishvn': {'url': 'https://data.mendeley.com/datasets/b97hxbxtpd/5', 'license': 'CC BY 4.0', 'authors': ['Thai Nguyen Vu']},
            'legitphish': {'url': 'https://data.mendeley.com/datasets/hx4m73v2sf/2', 'license': 'CC BY 4.0', 'authors': ['Rachana Potpelwar', 'Uday Kulkarni', 'Jaishri Waghmare']},
        },
        'models': results,
        'listed_urls_visited': False,
        'release_decision': 'Neither cross-source candidate is promoted; frozen, homepage, and source-shift metrics disagree materially.',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
