"""Evaluate thresholds selected exclusively from a saved calibration fold.

This produces diagnostic results only; it never changes a model or promotes it.
"""
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

from app.policy import valid_http_url
from app.train import evaluate, url_series
from app.url_model import load_bundle


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def threshold_for_fpr(probabilities: np.ndarray, labels: np.ndarray, max_fpr: float) -> float:
    negatives = np.sort(probabilities[labels == 0])
    if len(negatives) < 100 or not 0 <= max_fpr < 1:
        raise ValueError('Need at least 100 negatives and 0 <= max_fpr < 1')
    allowed = int(np.floor(max_fpr * len(negatives)))
    return float(np.nextafter(negatives[-allowed - 1], np.inf))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    model_dir = ROOT / 'models' / args.model
    meta = json.loads((model_dir / 'metadata.json').read_text(encoding='utf-8'))
    cal_path = model_dir / 'calibration_scores.csv'
    if sha256(cal_path) != meta.get('calibration_scores_sha256'):
        raise ValueError('Calibration scores do not match the model metadata')
    cal = pd.read_csv(cal_path, float_precision='round_trip')
    y = pd.to_numeric(cal['label'], errors='raise').to_numpy()
    p = pd.to_numeric(cal['probability'], errors='raise').to_numpy(dtype=float)
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Invalid calibration rows')
    y = y.astype(int)
    thresholds = {str(rate): threshold_for_fpr(p, y, rate) for rate in (.001, .002, .005)}
    bundle = load_bundle(model_dir / 'model.joblib')
    datasets = {
        'frozen_development': ROOT/'data/frozen_regression.csv',
        'homepage_development': ROOT/'data/realworld/legitimate_challenge.csv',
        'phishvn_test_development': ROOT/'data/realworld/evaluations/phishvn_temporal/test.csv',
        'legitphish_holdout_development': ROOT/'data/realworld/evaluations/legitphish_v2_holdout.csv',
        'openphish_archive_development': ROOT/'data/realworld/archive/openphish_20261005_unseen.csv',
    }
    results = {}
    for name, path in datasets.items():
        frame = pd.read_csv(path, low_memory=False)
        urls = url_series(frame)
        valid = urls.map(valid_http_url)
        frame, urls = frame.loc[valid], urls.loc[valid].tolist()
        labels = pd.to_numeric(frame['label'], errors='raise').to_numpy()
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f'Non-binary labels in {path}')
        labels = labels.astype(int)
        scores = np.concatenate([bundle.predict_proba(urls[i:i+512]) for i in range(0, len(urls), 512)])
        results[name] = {'rows':len(urls), 'invalid_rows':int((~valid).sum()), 'sha256':sha256(path),
                         'metrics':{key:evaluate(scores, labels, threshold) for key, threshold in thresholds.items()}}
    gov = ROOT/'artifacts/realworld/gov_registry_probe_20261125/registered_domains.csv'
    if gov.exists():
        urls = pd.read_csv(gov)['url'].astype(str).tolist()
        scores = bundle.predict_proba(urls)
        results['gov_registration_proxy_development'] = {'rows': len(urls), 'sha256':sha256(gov),
            'above_threshold': {key:int((scores >= threshold).sum()) for key, threshold in thresholds.items()},
            'limitation': 'Registry membership is not a verified benign URL label.'}
    report = {'generated_at':datetime.now(timezone.utc).isoformat(), 'model':args.model,
              'model_sha256':sha256(model_dir/'model.joblib'), 'calibration_sha256':sha256(cal_path),
              'thresholds':thresholds, 'calibration_metrics':{key:evaluate(p,y,t) for key,t in thresholds.items()},
              'evaluation':results, 'status':'diagnostic only; prior stacking used this fold indirectly and fit the calibrator on it; listed populations were also examined previously, so no independent release claim is possible',
              'model_modified':False, 'promoted':False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'model':args.model,'thresholds':thresholds,'output':str(args.out),'promoted':False}))


if __name__ == '__main__':
    main()
