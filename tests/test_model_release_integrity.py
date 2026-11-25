import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.api as api
import app.registry as registry
from app.domain import registrable_domain
from app.train import choose_operating_threshold, choose_split, domain_groups, evaluate, source_weights, url_series, validate_temporal_sources
from scripts.model_temporal_evidence import verify_model_temporal_inputs
from scripts.realworld_eval import metric_summary, overlaps_training_domain
from scripts.promote_if_gates import _calibration_metrics
from scripts.evaluate_shortcuts import assess


def test_threshold_search_matches_exhaustive_reference():
    rng = np.random.default_rng(41)
    for _ in range(5):
        p = np.round(rng.random(80), 2)
        y = rng.integers(0, 2, len(p))
        candidates = np.unique(np.clip(np.r_[np.linspace(.1, .75, 261), p], .1, .75))
        rows = [(float(t), evaluate(p, y, float(t))) for t in candidates]
        feasible = [(t, m) for t, m in rows if m['precision'] >= .8 and m['recall'] >= .7]
        if feasible:
            expected = max(feasible, key=lambda x: (min(x[1]['precision'], x[1]['recall']), x[1]['f1'], x[1]['accuracy'], -abs(x[0] - .5)))
        else:
            def fallback(row):
                t, m = row
                penalty = max(0, .8 - m['precision']) + max(0, .7 - m['recall'])
                return (m['f1'] - .75 * penalty, min(m['precision'], m['recall']), m['accuracy'], -abs(t - .5))
            expected = max(rows, key=fallback)
        threshold, measured, _ = choose_operating_threshold(p, y, .8, .7)
        assert threshold == expected[0]
        assert measured == expected[1]


def test_source_weights_are_explicit_and_validated():
    frame = pd.DataFrame({'source': ['legacy', 'new', None, 'legacy']})
    assert source_weights(frame, ['legacy=12']).tolist() == [12, 1, 1, 12]
    for spec in (['missing=2'], ['legacy=0'], ['legacy=nan'], ['legacy=2', 'legacy=3']):
        with pytest.raises(ValueError):
            source_weights(frame, spec)


def test_calibration_threshold_respects_false_positive_limit():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    probabilities = np.array([.1, .2, .3, .7, .5, .8, .9, .95])
    threshold, report, mode = choose_operating_threshold(probabilities, labels, min_precision=.5, min_recall=.5, lower=.1, upper=.95, max_fpr=0)
    assert mode == 'bounded_precision_recall'
    assert threshold > .7
    assert report['false_positive_rate'] == 0


def test_calibration_file_keeps_boundary_scores(tmp_path):
    boundary = 0.033683209552273004
    path = tmp_path / 'calibration.csv'
    pd.DataFrame({'label': [0, 1], 'probability': [0.001, boundary]}).to_csv(path, index=False)
    result = _calibration_metrics(path, hashlib.sha256(path.read_bytes()).hexdigest(), boundary)
    assert result['tp'] == 1
    assert result['fn'] == 0


def test_domain_groups_use_public_and_private_suffixes():
    assert registrable_domain('https://a.example.co.za/path') == 'example.co.za'
    assert registrable_domain('https://alice.github.io/') == 'alice.github.io'
    assert registrable_domain('https://bob.github.io/') == 'bob.github.io'
    assert registrable_domain('https://a.b.invalid/') == 'b.invalid'
    assert overlaps_training_domain('https://tenant.github.io/path', {'github.io'})
    assert not overlaps_training_domain('https://another.example/path', {'example.com'})


def test_positive_only_feed_does_not_claim_precision_or_accuracy():
    report = metric_summary({'tp': 9, 'fn': 1, 'fp': 0, 'tn': 0})
    assert report['recall'] == .9
    assert report['accuracy'] is None
    assert report['precision'] is None
    assert report['false_positive_rate'] is None


def test_scheme_diagnostic_reports_shortcut_without_labeling_counterfactuals():
    class SchemeModel:
        threshold = .5
        def predict_proba(self, urls):
            return np.array([.9 if url.startswith('http://') else .1 for url in urls])
    frame = pd.DataFrame({'url': ['http://bad.example/', 'https://good.example/'], 'label': [1, 0]})
    report = assess(SchemeModel(), frame)
    assert report['baseline_model_decision_agreement'] == 1
    assert report['scheme_change_decision_flip_rate'] == 1
    assert report['scheme_changed_urls_are_not_assigned_labels'] is True


def test_local_feed_expires_even_when_file_is_unchanged(monkeypatch, tmp_path):
    import app.local_intel as intel
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    feed = tmp_path / 'openphish.csv'
    feed.write_text('url,label,collected_at\nhttps://feed-check.example/path,1,2026-01-01T00:00:00+00:00\n')
    with monkeypatch.context() as patch:
        patch.setattr(intel, 'PHISHING_FEED_PATH', feed)
        patch.setattr(intel, 'URLHAUS_FEED_PATH', tmp_path / 'missing.csv')
        patch.setattr(intel, 'settings', replace(intel.settings, local_intel_max_age_seconds=30))
        patch.setattr(intel, '_utcnow', lambda: now)
        intel.reload_local_intel()
        assert intel.local_phishing('https://feed-check.example/path')
        patch.setattr(intel, '_utcnow', lambda: now + timedelta(seconds=31))
        assert not intel.local_phishing('https://feed-check.example/path')
        assert intel.local_intel_metadata()['stale'] is True
    intel.reload_local_intel()


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_temporal_qualification_checks_each_recorded_input(tmp_path):
    model = tmp_path / 'model'; model.mkdir()
    data = tmp_path / 'data'; data.mkdir()
    train = pd.DataFrame([{'url': f'https://train{i}.example.com/', 'label': i % 2, 'event_time': '2022-01-01T00:00:00Z'} for i in range(200)])
    # Distinct registrable domains keep the dated split domain-disjoint.
    train['url'] = [f'https://train{i}.example/' for i in range(len(train))]
    test = pd.DataFrame([{'url': f'https://test{i}.example/', 'label': i % 2, 'event_time': '2023-02-01T00:00:00Z'} for i in range(120)])
    train.to_csv(data / 'train.csv', index=False)
    test.to_csv(data / 'test.csv', index=False)
    pd.concat([train, test]).to_csv(data / 'dataset.csv', index=False)
    (data / 'dataset.csv.meta.json').write_text(json.dumps({'timestamp_provenance': 'source_attested', 'timestamp_semantics': 'collection date', 'cutoff': '2023-01-01T00:00:00Z', 'dataset_sha256': _sha(data / 'dataset.csv')}))
    sources = {}
    for name, label in [('phishing', 1), ('legitimate', 0)]:
        path = tmp_path / f'{name}.csv'
        train.loc[train.label == label, ['url']].to_csv(path, index=False)
        sources[name] = {'path': str(path), 'sha256': _sha(path)}
    (model / 'metadata.json').write_text(json.dumps({'data_sources': sources}))
    assert verify_model_temporal_inputs(model, data)['qualified'] is True
    pd.concat([test, test.iloc[:1]]).to_csv(data / 'test.csv', index=False)
    duplicate_result = verify_model_temporal_inputs(model, data)
    assert duplicate_result['qualified'] is False
    assert 'duplicate normalized URLs' in duplicate_result['reason']
    test.to_csv(data / 'test.csv', index=False)
    extra = tmp_path / 'extra.csv'
    pd.DataFrame([{'url': 'https://undated.example/', 'label': 0}]).to_csv(extra, index=False)
    sources['extra'] = {'path': str(extra), 'sha256': _sha(extra)}
    (model / 'metadata.json').write_text(json.dumps({'data_sources': sources}))
    result = verify_model_temporal_inputs(model, data)
    assert result['qualified'] is False
    assert 'without pre-cutoff source evidence' in result['reason']


def test_temporal_training_requires_verified_source_manifest(tmp_path):
    phishing = tmp_path / 'p.csv'; phishing.write_text('url,event_time\nhttps://p.example/,2022-01-01\n')
    legitimate = tmp_path / 'b.csv'; legitimate.write_text('url,event_time\nhttps://b.example/,2022-01-01\n')
    args = SimpleNamespace(phishing=str(phishing), legitimate=str(legitimate), extra=None, temporal_source_metadata=None)
    with pytest.raises(ValueError, match='temporal-source-metadata'):
        validate_temporal_sources(args)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'timestamp_provenance': 'source_attested', 'timestamp_semantics': 'collection date', 'timestamp_column': 'event_time', 'source_files': [{'path': str(p), 'sha256': _sha(p)} for p in (phishing, legitimate)]}))
    args.temporal_source_metadata = str(manifest)
    assert validate_temporal_sources(args) == 'event_time'
    phishing.write_text('changed')
    with pytest.raises(ValueError, match='phishing input'):
        validate_temporal_sources(args)


def test_temporal_split_excludes_later_sibling_domains():
    rows=[]
    for i in range(300):
        host=f'host{i}.example' if i < 295 else 'new.host0.example'
        rows.append({'_normalized_url': f'https://{host}/', 'label': i % 2, 'event_time': (pd.Timestamp('2020-01-01', tz='UTC') + pd.Timedelta(days=i)).isoformat()})
    frame=pd.DataFrame(rows)
    train, test, kind, _ = choose_split(frame, require_temporal=True, timestamp_column='event_time')
    assert kind == 'temporal'
    assert set(domain_groups(frame.iloc[train])).isdisjoint(set(domain_groups(frame.iloc[test])))
    assert len(test) == 55


def test_normalized_url_column_is_respected():
    frame=pd.DataFrame({'_normalized_url': ['https://known.example/path']})
    assert url_series(frame).tolist() == ['https://known.example/path']


def test_release_approval_is_bound_to_actual_challenge(monkeypatch, tmp_path):
    models = tmp_path / 'models'; model = models / 'v1'; model.mkdir(parents=True)
    challenge = tmp_path / 'data/realworld/legitimate_challenge.csv'; challenge.parent.mkdir(parents=True)
    challenge.write_text('url,label\nhttps://example.com/,0\n')
    (model / 'model.joblib').write_bytes(b'model')
    evidence = {'unique_registrable_domains': 15, 'verified_unique_domains': 15, 'false_positives': 0, 'exact_training_overlap': 0, 'domain_training_overlap': 0, 'observed_at_coverage': 1, 'provenance_sha256_match': True, 'challenge_sha256': _sha(challenge)}
    approval = {'schema_version': 2, 'approved': True, 'model_version': 'v1', 'model_sha256': _sha(model / 'model.joblib'), 'challenge_sha256': _sha(challenge), 'legitimate_challenge': evidence, 'passed_gates': {gate: True for gate in registry.REQUIRED_RELEASE_GATES}}
    (model / 'BLOCKING_APPROVED.json').write_text(json.dumps(approval))
    monkeypatch.setattr(registry, 'ROOT', tmp_path)
    monkeypatch.setattr(registry, 'MODELS', models)
    assert registry.validate_release_approval('v1')['approved'] is True
    challenge.write_text('changed')
    with pytest.raises(RuntimeError, match='approval'):
        registry.validate_release_approval('v1')


def test_url_blocking_approval_does_not_enable_email_blocking(monkeypatch):
    monkeypatch.setattr(api, 'settings', replace(api.settings, ml_blocking_enabled=True))
    monkeypatch.setattr(api, 'EMAIL_MODEL', object())
    monkeypatch.setattr(api, 'email_model_predict', lambda *_: [0.99])
    response = TestClient(api.app).post('/v1/predict/email', headers={'X-API-Key': 'test-inference-key'}, json={'subject': 'Notice', 'body': 'Review this message'})
    assert response.status_code == 200
    assert response.json()['binary_verdict'] == 'unknown'
    assert response.json()['blocking_eligible'] is False


def test_homepage_policy_remains_nonblocking_for_single_and_batch(monkeypatch):
    monkeypatch.setattr(api, 'predict_urls', lambda urls, model: ([.99] * len(urls), 0, 0, ['model'] * len(urls), ['model'] * len(urls)))
    client = TestClient(api.app)
    headers = {'X-API-Key': 'test-inference-key'}
    url = 'https://github.com/'
    single = client.post('/v1/predict', headers=headers, json={'url': url}).json()
    batch = client.post('/v1/predict/batch', headers=headers, json={'urls': [url]}).json()['results'][0]
    assert single['decision'] == batch['decision'] == 'verified_official_homepage_root'
    assert single['blocking_eligible'] is batch['blocking_eligible'] is False
