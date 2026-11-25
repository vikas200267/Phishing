"""Verify whether a model's recorded inputs qualify for a dated replay.

The checks are intentionally conservative. An undated input, changed source
file, or unrecognized row makes the model ineligible for this temporal test.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from app.train import domain_groups, url_series

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data/realworld/evaluations/phishvn_temporal'


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _rows(frame: pd.DataFrame) -> set[tuple[str, int]]:
    labels = pd.to_numeric(frame['label'], errors='raise').astype(int)
    if not labels.isin([0, 1]).all():
        raise ValueError('non-binary label')
    return set(zip(url_series(frame).astype(str), labels))


def verify_model_temporal_inputs(model_dir: Path, data_dir: Path = DATA_DIR) -> dict:
    """Verify actual model inputs against the source-attested pre-cutoff rows."""
    result = {'qualified': False, 'reason': None, 'source': 'PhishVN v5 source-attested collection dates'}
    try:
        metadata = json.loads((model_dir / 'metadata.json').read_text(encoding='utf-8'))
        source = json.loads((data_dir / 'dataset.csv.meta.json').read_text(encoding='utf-8'))
        dataset_path, train_path, test_path = (data_dir / name for name in ('dataset.csv', 'train.csv', 'test.csv'))
        if source.get('timestamp_provenance') not in {'source_attested', 'source_provided'}:
            raise ValueError('dataset timestamps are not source-attested')
        if 'collection date' not in str(source.get('timestamp_semantics', '')).lower():
            raise ValueError('dataset timestamp semantics are not collection dates')
        if _sha(dataset_path) != source.get('dataset_sha256'):
            raise ValueError('dated dataset hash does not match its provenance manifest')
        cutoff = pd.to_datetime(source['cutoff'], utc=True, errors='raise')
        dataset, train, test = (pd.read_csv(p, low_memory=False) for p in (dataset_path, train_path, test_path))
        for name, frame in [('dataset', dataset), ('train', train), ('test', test)]:
            if not {'url', 'label', 'event_time'}.issubset(frame.columns):
                raise ValueError(f'{name} is missing URL, label, or source event time')
        train_times = pd.to_datetime(train['event_time'], utc=True, errors='coerce')
        test_times = pd.to_datetime(test['event_time'], utc=True, errors='coerce')
        if train_times.isna().any() or test_times.isna().any() or not train_times.lt(cutoff).all() or not test_times.ge(cutoff).all():
            raise ValueError('dated split contains missing or out-of-cutoff timestamps')
        def timed_rows(frame):
            return set(zip(url_series(frame).astype(str), pd.to_numeric(frame['label'], errors='raise').astype(int), pd.to_datetime(frame['event_time'], utc=True, errors='raise').astype(str)))
        dataset_rows = timed_rows(dataset)
        if not timed_rows(train).issubset(dataset_rows) or not timed_rows(test).issubset(dataset_rows):
            raise ValueError('dated split does not match the hashed source dataset')
        train_rows, test_rows = _rows(train), _rows(test)
        train_urls, test_urls = {u for u, _ in train_rows}, {u for u, _ in test_rows}
        train_domains = set(domain_groups(pd.DataFrame({'_normalized_url': list(train_urls)})).astype(str))
        test_domains = set(domain_groups(pd.DataFrame({'_normalized_url': list(test_urls)})).astype(str))
        if train_urls & test_urls or train_domains & test_domains:
            raise ValueError('dated training and holdout URLs or domain groups overlap')
        if any(int((test['label'] == label).sum()) < 50 for label in (0, 1)):
            raise ValueError('dated holdout has insufficient class support')
        sources = metadata.get('data_sources') or {}
        params = metadata.get('training_parameters') or {}
        checked = 0
        for name, fixed_label in (('phishing', 1), ('legitimate', 0), ('extra', None)):
            entry = sources.get(name)
            path_text = entry.get('path') if isinstance(entry, dict) else params.get(name)
            expected = entry.get('sha256') if isinstance(entry, dict) else entry
            if not path_text:
                if name != 'extra':
                    raise ValueError(f'{name} training source is missing')
                continue
            path = Path(str(path_text).replace('\\', '/'))
            if not path.is_absolute():
                path = ROOT / path
            if not expected or _sha(path) != expected:
                raise ValueError(f'{name} training source hash does not match the model metadata')
            frame = pd.read_csv(path, low_memory=False)
            if fixed_label is not None:
                frame['label'] = fixed_label
            if 'label' not in frame:
                raise ValueError(f'{name} training source has no labels')
            rows = _rows(frame)
            if not rows.issubset(train_rows):
                raise ValueError(f'{name} training source contains rows without pre-cutoff source evidence')
            checked += len(rows)
        result.update({'qualified': True, 'reason': None, 'cutoff': cutoff.isoformat(),
                       'verified_training_input_rows': checked, 'test_rows': len(test),
                       'max_source_training_time': train_times.max().isoformat(),
                       'dated_dataset_sha256': _sha(dataset_path), 'test_sha256': _sha(test_path)})
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result['reason'] = str(exc)
    return result
