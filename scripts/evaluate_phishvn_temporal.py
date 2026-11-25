from __future__ import annotations

import argparse
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
from scripts.model_temporal_evidence import verify_model_temporal_inputs


def score_file(bundle, csv_path: Path, *, exclude_invalid: bool = True) -> dict:
    frame = pd.read_csv(csv_path)
    urls = url_series(frame)
    valid = urls.map(valid_http_url)
    invalid_count = int((~valid).sum())
    if invalid_count and not exclude_invalid:
        raise ValueError(f'{invalid_count} invalid HTTP URLs in {csv_path}')
    frame = frame.loc[valid].copy()
    urls = urls.loc[valid].tolist()
    labels = pd.to_numeric(frame['label'], errors='raise').astype(int).to_numpy()
    scores = bundle.predict_proba(urls)
    result = evaluate(scores, labels, bundle.threshold)
    result['rows_scored'] = len(frame)
    result['invalid_http_rows_excluded'] = invalid_count
    result['threshold'] = float(bundle.threshold)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description='Evaluate a pre-cutoff URL model on the source-attested PhishVN temporal holdout and frozen benign homepage challenge.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', type=Path, default=ROOT/'data/realworld/evaluations/phishvn_temporal/model_evaluation.json')
    args = parser.parse_args()
    model_dir = ROOT/'models'/args.model
    if not model_dir.exists():
        model_dir = ROOT/'models'/'candidates'/args.model
    if not (model_dir/'model.joblib').exists():
        raise FileNotFoundError(f'Model artifact not found: {args.model}')
    bundle = load_bundle(model_dir/'model.joblib')
    data_dir = ROOT/'data/realworld/evaluations/phishvn_temporal'
    temporal = score_file(bundle, data_dir/'test.csv')
    homepage = score_file(bundle, ROOT/'data/realworld/legitimate_challenge.csv')
    metadata = json.loads((model_dir/'metadata.json').read_text(encoding='utf-8'))
    temporal_gate = json.loads((ROOT/'data/manifests/url_temporal_gate.json').read_text(encoding='utf-8'))
    evidence = verify_model_temporal_inputs(model_dir, data_dir)
    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'candidate_version': args.model,
        'candidate_training_cutoff': evidence.get('cutoff'),
        'training_data_max_event_time': evidence.get('max_source_training_time'),
        'training_data_rows': metadata.get('prepared_dataset', {}).get('rows_after_exclusions'),
        'recorded_training_sources': metadata.get('data_sources'),
        'temporal_reference_source': evidence.get('source'),
        'model_temporal_input_evidence': evidence,
        'temporal_dataset_gate': temporal_gate.get('status'),
        'temporal_holdout': temporal,
        'legitimate_homepage_challenge': homepage,
        'frozen_regression_gate': metadata.get('promotion_gate', {}).get('frozen_regression_passed'),
        'frozen_regression_metrics': json.loads((model_dir/'frozen_regression_metrics.json').read_text(encoding='utf-8')) if (model_dir/'frozen_regression_metrics.json').exists() else None,
        'promotion_eligible': metadata.get('promotion_eligible', False),
        'production_model_version': (ROOT/'models/CURRENT').read_text(encoding='utf-8').strip(),
        'model_release_status': 'temporal_replay_inputs_qualified' if evidence['qualified'] else 'diagnostic_only_model_training_not_temporally_qualified',
        'source_url': 'https://data.mendeley.com/datasets/b97hxbxtpd/5',
        'license': 'CC BY 4.0; Thai Nguyen Vu attribution required',
        'evaluation_urls_visited': False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
