from app.trusted_homepages import is_trusted_official_homepage
from app.api import apply_official_homepage_policy


def test_official_homepage_policy_is_narrow_and_auditable():
    assert is_trusted_official_homepage('https://github.com/')
    assert not is_trusted_official_homepage('https://github.com/login')
    assert not is_trusted_official_homepage('https://github.com.evil.example/')
    assert not is_trusted_official_homepage('http://github.com/')
    assert not is_trusted_official_homepage('https://github.com/?next=/')

    result = {'verdict': 'phishing', 'risk_score': 0.966, 'risk_band': 'high'}
    assert apply_official_homepage_policy(result, 'https://github.com/', 'model')
    assert result['verdict'] == 'benign'
    assert result['binary_verdict'] == 'benign'
    assert result['model_verdict'] == 'phishing'
    assert result['model_risk_score'] == 0.966
    assert result['risk_score'] == 0.0
    assert not result['blocking_eligible']


def test_exact_threat_evidence_takes_precedence_over_homepage_policy():
    result = {'verdict': 'phishing', 'risk_score': 1.0}
    assert not apply_official_homepage_policy(result, 'https://github.com/', 'local_intel')
    assert result == {'verdict': 'phishing', 'risk_score': 1.0}
