from fastapi.testclient import TestClient
from app.api import app
from app.dns_tls import resolve_host
from app.reputation import verified_verdict, verified_verdicts

client = TestClient(app)


def test_feedback_requires_admin_key():
    r = client.post('/v1/feedback', json={'url':'https://audit-feedback-auth.example/a','label':0,'verified':True})
    assert r.status_code == 401


def test_prediction_requires_key():
    r = client.post('/v1/predict', json={'url':'https://example.com/a'})
    assert r.status_code == 401


def test_prediction_with_key():
    r = client.post('/v1/predict', headers={'X-API-Key':'test-inference-key'}, json={'url':'https://example.com/a'})
    assert r.status_code == 200


def test_feedback_verified_is_exact_url_only():
    base = 'https://audit-feedback-scope.example'
    r = client.post('/v1/feedback', headers={'X-Admin-Key':'test-admin-key'}, json={'url':base+'/a-v8','label':1,'verified':True})
    assert r.status_code == 202
    assert verified_verdict(base+'/a-v8') == 1
    # A verified URL must not silently relabel every URL in the same domain.
    assert verified_verdict(base+'/b-v8') is None
    assert verified_verdicts([base+'/a-v8', base+'/b-v8']) == {base+'/a-v8': 1}


def test_bad_scheme_rejected():
    r = client.post('/v1/predict', headers={'X-API-Key':'test-inference-key'}, json={'url':'javascript:alert(1)'})
    assert r.status_code == 422


def test_private_literal_blocked():
    assert resolve_host('127.0.0.1')['ok'] is False


def test_dns_private_resolution_is_blocked(monkeypatch):
    import app.dns_tls as dns_tls
    def fake_getaddrinfo(*args, **kwargs):
        return [
            (2, 1, 6, '', ('10.0.0.8', 443)),
            (10, 1, 6, '', ('fd00::8', 443, 0, 0)),
        ]
    monkeypatch.setattr(dns_tls.socket, 'getaddrinfo', fake_getaddrinfo)
    result = dns_tls.resolve_host('rebind-check.example')
    assert result['ok'] is False
    assert result.get('resolved_ips') == ['10.0.0.8', 'fd00::8']


def test_request_body_limit():
    r = client.post('/v1/predict/batch', headers={'X-API-Key':'test-inference-key','content-length':'2000001'}, content=b'{}')
    assert r.status_code == 413


def test_enrichment_disabled_by_default():
    r = client.post('/v1/enrich', headers={'X-Admin-Key':'test-admin-key'}, json={'url':'https://example.com/a'})
    assert r.status_code == 403


def test_no_hardcoded_benign_seed():
    from app import local_intel
    assert not hasattr(local_intel, 'local_benign_exact')
    assert not getattr(local_intel, 'LOCAL_BENIGN_URLS', None)


def test_email_without_model_is_unknown(monkeypatch):
    import app.api as api
    monkeypatch.setattr(api, 'EMAIL_MODEL', None)
    r = client.post('/v1/predict/email', headers={'X-API-Key':'test-inference-key'}, json={'subject':'Account notice','body':'Please review the message'})
    assert r.status_code == 200
    body = r.json()
    assert body['verdict'] == 'unknown'
    assert body['binary_verdict'] == 'unknown'


def test_cascade_enabled_by_default_with_validated_bounds():
    from app.config import settings
    assert settings.cascade_enabled is True
    assert settings.cascade_low_threshold == 0.10
    assert settings.cascade_high_threshold == 0.90


def test_production_model_has_no_manual_benign_seed():
    from pathlib import Path
    import json
    from app.registry import current
    root = Path(__file__).resolve().parents[1]
    meta = json.loads((root/'models'/current()/'metadata.json').read_text())
    extra = (meta.get('training_parameters') or {}).get('extra')
    assert extra != 'data/source/realworld_extra_v1.csv'
    assert extra in (None, 'data/source/openphish_training_extra_clean.csv')
    assert not (root/'models'/current()/'metadata.json').read_text().lower().count('benign_seed')


def test_latest_realworld_evaluation_endpoint():
    from app.registry import current
    r = client.get('/v1/evaluation/latest', headers={'X-Admin-Key':'test-admin-key'})
    assert r.status_code == 200
    body = r.json()
    assert body['model_version'] == current()
    assert body['active_model_version'] == current()
    assert body['generated_at']
    assert body['visited_listed_urls'] is False


def test_cascade_suspicious_is_not_binary_benign():
    from app.api import classify_fast_score
    verdict, decision = classify_fast_score(0.5)
    assert verdict == 'suspicious'
    assert decision == 'full-model-required'


def test_verified_feedback_score_is_not_mislabeled_as_calibrated_model_probability(monkeypatch):
    import app.api as api
    url = 'https://score-provenance.example/safe-path'
    monkeypatch.setattr(api, 'verified_verdicts', lambda urls: {api.normalize_url(url): 0})
    r = client.post('/v1/predict', headers={'X-API-Key':'test-inference-key'}, json={'url':url})
    assert r.status_code == 200
    body = r.json()
    assert body['verdict'] == 'benign'
    assert body['score_type'] == 'verified_label'
    assert body['inference_stage'] == 'verified_feedback'
    assert body['decision'] == 'verified_exact_url_label'


def test_local_feed_score_is_not_mislabeled_as_calibrated_model_probability(monkeypatch):
    import app.api as api
    url = 'https://score-provenance.example/feed-match'
    monkeypatch.setattr(api, 'verified_verdicts', lambda urls: {})
    monkeypatch.setattr(api, 'local_phishing', lambda candidate: candidate == url)
    r = client.post('/v1/predict', headers={'X-API-Key':'test-inference-key'}, json={'url':url})
    assert r.status_code == 200
    body = r.json()
    assert body['verdict'] == 'phishing'
    assert body['score_type'] == 'local_threat_intelligence'
    assert body['inference_stage'] == 'local_intel'
    assert body['decision'] == 'exact_local_threat_feed_match'
