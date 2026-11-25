"""Score a held-out .gov registry sample as a population proxy, not ground truth."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from app.domain import registrable_domain
from app.train import url_series
from app.url_model import load_bundle
from scripts.realworld_eval import load_input_domains, load_training_domains, overlaps_training_domain

SOURCE = 'https://raw.githubusercontent.com/cisagov/dotgov-data/main/current-federal.csv'
RESERVED = (
    'data/frozen_regression.csv',
    'data/realworld/legitimate_challenge.csv',
    'data/realworld/evaluations/phishvn_temporal/test.csv',
    'data/realworld/evaluations/legitphish_v2_holdout.csv',
    'data/realworld/archive/openphish_20261005_unseen.csv',
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', required=True, help='Comma-separated model versions; all input domains are excluded.')
    parser.add_argument('--sample-size', type=int, default=500)
    parser.add_argument('--out-dir', type=Path, default=ROOT/'artifacts/realworld/gov_registry_probe')
    args = parser.parse_args()
    versions = [v.strip() for v in args.models.split(',') if v.strip()]
    if not versions or args.sample_size < 1:
        raise ValueError('Provide model versions and a positive sample size')
    excluded = set()
    for version in versions:
        excluded |= load_training_domains(version)
        excluded |= load_input_domains(version)
    for name in RESERVED:
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f'Required reserved evaluation file is missing: {name}')
        frame = pd.read_csv(path, low_memory=False)
        excluded.update(registrable_domain(u) for u in url_series(frame).astype(str))
    excluded.discard('')
    request = Request(SOURCE, headers={'User-Agent': 'PhishGuard-Domain-Registry-Probe/1.0'})
    with urlopen(request, timeout=60) as response:
        raw = response.read()
    collected_at = datetime.now(timezone.utc).isoformat()
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    if not reader.fieldnames or 'Domain name' not in reader.fieldnames:
        raise ValueError('Unexpected .gov registry schema')
    domains = {str(row.get('Domain name') or '').strip().lower().rstrip('.') for row in reader}
    domains = {d for d in domains if d.endswith('.gov') and not overlaps_training_domain('https://' + d, excluded)}
    domains = sorted(domains, key=lambda d: hashlib.sha256(('phishguard-gov-probe-v1:' + d).encode()).hexdigest())
    urls = ['https://' + d + '/' for d in domains[:args.sample_size]]
    if len(urls) < min(100, args.sample_size):
        raise ValueError('Insufficient disjoint registry domains for the requested probe')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.out_dir / 'registered_domains.csv'
    pd.DataFrame({'url': urls}).to_csv(sample_path, index=False)
    scores = {}
    for version in versions:
        model = load_bundle(ROOT/'models'/version/'model.joblib')
        probs = model.predict_proba(urls)
        scores[version] = {
            'threshold': float(model.threshold), 'domains': len(urls),
            'scores_at_or_above_threshold': int((probs >= model.threshold).sum()),
            'fraction_at_or_above_threshold': float((probs >= model.threshold).mean()),
        }
    report = {
        'collected_at': collected_at, 'source_url': SOURCE,
        'source_sha256': hashlib.sha256(raw).hexdigest(),
        'sample_sha256': hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        'sample_method': 'SHA-256 sorted registered federal .gov roots after excluding all supplied model input domains and prior challenge domains',
        'sample_size': len(urls), 'models': scores,
        'role': 'registered government domain proxy; not independently verified benign URL labels',
        'limitations': ['A domain registration does not establish that a website exists or is safe.', 'This is not a false-positive-rate or general-population accuracy estimate.', 'Listed websites were not visited.'],
    }
    (args.out_dir / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
