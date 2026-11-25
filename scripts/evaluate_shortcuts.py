"""Compare URL scores with a scheme-only baseline and scheme changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from app.policy import valid_http_url
from app.train import evaluate, url_series
from app.url_model import load_bundle


def scheme_class_counts(frame):
    labels = pd.to_numeric(frame['label'], errors='raise')
    if not labels.isin([0, 1]).all():
        raise ValueError('Labels must be binary')
    urls = url_series(frame).astype(str)
    valid = urls.map(valid_http_url)
    counts = {f'{label}_{scheme}': 0 for label in (0, 1) for scheme in ('http', 'https')}
    for url, label in zip(urls[valid], labels[valid]):
        counts[f'{int(label)}_{urlsplit(url).scheme.lower()}'] += 1
    return counts


def assess(model, frame):
    labels = pd.to_numeric(frame['label'], errors='raise')
    if not labels.isin([0, 1]).all():
        raise ValueError('Labels must be binary')
    urls = url_series(frame).astype(str)
    valid = urls.map(valid_http_url)
    urls = urls[valid].tolist()
    y = labels[valid].astype(int).to_numpy()
    if not urls:
        raise ValueError('No valid URLs to evaluate')
    flipped = []
    for url in urls:
        p = urlsplit(url)
        flipped.append(urlunsplit(('https' if p.scheme.lower() == 'http' else 'http', p.netloc, p.path, p.query, p.fragment)))
    original = np.concatenate([model.predict_proba(urls[i:i+512]) for i in range(0, len(urls), 512)])
    alternative = np.concatenate([model.predict_proba(flipped[i:i+512]) for i in range(0, len(flipped), 512)])
    baseline = np.array([urlsplit(url).scheme.lower() == 'http' for url in urls], dtype=int)
    decisions = original >= model.threshold
    return {
        'rows': len(urls), 'invalid_http_rows_excluded': int((~valid).sum()),
        'model': evaluate(original, y, model.threshold),
        'http_only_baseline': evaluate(baseline.astype(float), y, .5),
        'baseline_model_decision_agreement': float(np.mean(baseline == decisions)),
        'scheme_class_counts': scheme_class_counts(frame),
        'scheme_change_decision_flip_rate': float(np.mean(decisions != (alternative >= model.threshold))),
        'scheme_changed_urls_are_not_assigned_labels': True,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    model_path = ROOT/'models'/args.model/'model.joblib'
    model = load_bundle(model_path)
    report = {'model_version': args.model, 'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
              'dataset_sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
              'metrics': assess(model, pd.read_csv(args.dataset, low_memory=False)),
              'interpretation': 'The scheme-only rule is a diagnostic baseline, not a phishing detector. Changed-scheme URLs were not visited or assigned ground-truth labels.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
