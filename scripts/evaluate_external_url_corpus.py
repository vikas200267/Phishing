from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.train import domain_groups, evaluate, url_series
from app.url_model import load_bundle


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description='Evaluate a model on a timestamp-honest, registrable-domain holdout corpus.')
    ap.add_argument('--csv', type=Path, default=ROOT/'data/realworld/evaluations/legitphish_v2_holdout.csv')
    ap.add_argument('--model', default='CURRENT', help='Model version under models/ or a candidate version under models/candidates/.')
    ap.add_argument('--chunk-size', type=int, default=512)
    ap.add_argument('--out-dir', type=Path)
    args = ap.parse_args()
    if args.chunk_size < 1:
        raise ValueError('--chunk-size must be positive')

    version = (ROOT/'models/CURRENT').read_text(encoding='utf-8').strip() if args.model == 'CURRENT' else args.model
    model_path = ROOT/'models'/version/'model.joblib'
    if not model_path.exists():
        model_path = ROOT/'models'/'candidates'/version/'model.joblib'
    if not model_path.exists():
        raise FileNotFoundError(f'Model artifact not found for {version}')
    model = load_bundle(model_path)

    data = pd.read_csv(args.csv)
    if not {'url', 'label'}.issubset(data.columns):
        raise ValueError('Evaluation CSV needs url and label columns')
    data = data[data['label'].isin([0, 1])].copy().reset_index(drop=True)
    data['_normalized_url'] = url_series(data)
    data['label'] = pd.to_numeric(data['label'], errors='raise').astype(int)
    domains = domain_groups(data)
    training_domains_path = model_path.parent/'training_domains.txt'
    training_domains = set(training_domains_path.read_text(encoding='utf-8').splitlines()) if training_domains_path.exists() else set()
    domain_overlap = len(training_domains.intersection(set(domains.astype(str))))
    if args.csv.name.endswith('holdout.csv') and training_domains and domain_overlap:
        raise RuntimeError(f'Evaluation set overlaps model training domains: {domain_overlap}')

    urls = data['_normalized_url'].astype(str).tolist()
    probabilities = []
    for start in range(0, len(urls), args.chunk_size):
        probabilities.extend(model.predict_proba(urls[start:start + args.chunk_size]).astype(float).tolist())
    p = np.asarray(probabilities, dtype=float)
    y = data['label'].to_numpy(dtype=int)
    threshold = float(model.threshold)
    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'model_version': version,
        'model_sha256': sha256(model_path),
        'dataset_path': str(args.csv.relative_to(ROOT)) if args.csv.is_relative_to(ROOT) else str(args.csv),
        'dataset_sha256': sha256(args.csv),
        'rows': int(len(data)),
        'unique_urls': int(data['_normalized_url'].nunique()),
        'unique_registrable_domains': int(domains.nunique()),
        'label_counts': {str(k): int(v) for k, v in data.label.value_counts().sort_index().items()},
        'observation_time_coverage': float(data['observed_at'].fillna('').astype(str).str.strip().ne('').mean()) if 'observed_at' in data else 0.0,
        'threshold': threshold,
        'metrics_at_model_threshold': evaluate(p, y, threshold),
        'metrics_at_0_5': evaluate(p, y, 0.5),
        'score_quantiles': {str(q): float(np.quantile(p, q)) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)},
        'model_training_domain_count': len(training_domains),
        'overlap_with_model_training_domains': domain_overlap,
        'domain_disjoint_from_model_training': (domain_overlap == 0) if training_domains else None,
        'evaluation_urls_visited': False,
    }
    predictions = data[['url', 'label']].copy()
    predictions['risk_score'] = p
    predictions['prediction'] = (p >= threshold).astype(int)
    predictions['model_version'] = version
    out_dir = args.out_dir or ROOT/'artifacts/realworld/external_evaluations'/f'{Path(args.csv).stem}_{version}'
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir/'predictions.csv', index=False)
    (out_dir/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
