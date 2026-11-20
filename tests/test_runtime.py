from pathlib import Path
from app.url_model import load_bundle
from app.registry import current, versions
from app.features import extract_url_features

ROOT = Path(__file__).resolve().parents[1]


def test_production_model_roundtrip():
    v = current()
    m = load_bundle(ROOT / 'models' / v / 'model.joblib')
    p = m.predict_proba(['https://example.com/login'])[0]
    assert 0.0 <= float(p) <= 1.0
    assert m.version == v


def test_registry_filters_incompatible_models():
    vs = versions()
    assert current() in vs
    # Older experiments in this package use incompatible component schemas and
    # must not be offered as rollback targets.
    assert 'v20261001_060829' not in vs
    assert 'v20261001_061237' not in vs


def test_homograph_and_static_features():
    a = extract_url_features('https://xn--pple-43d.example/login')
    b = extract_url_features('https://example.com/assets/app.js?v=2026')
    assert a['has_punycode'] == 1
    assert b['path_depth'] >= 1
    assert b['query_param_count'] == 1


def test_current_release_schema():
    v = current()
    m = load_bundle(ROOT / 'models' / v / 'model.joblib')
    assert len(m.feature_names) == 104
    assert m.raw_components(['https://foo.example/login']).shape == (1, 7)


def test_local_intel_paths(monkeypatch):
    import csv
    import app.local_intel as li
    from app.config import settings
    from dataclasses import replace
    # Test against a real entry in the bundled threat-feed snapshot, while
    # allowing the test fixture to be historical and therefore normally stale.
    fixture_feed = ROOT/'data/history/openphish_20261001_snapshot.csv'
    with open(fixture_feed, newline='', encoding='utf-8') as f:
        url = next(csv.DictReader(f))['url']
    monkeypatch.setattr(li, 'settings', replace(settings, local_intel_max_age_seconds=10**9))
    monkeypatch.setattr(li, 'PHISHING_FEED_PATH', fixture_feed)
    li.reload_local_intel()
    assert li.local_phishing(url)


def test_current_model_integrity_manifest():
    from app.registry import verify_model_integrity
    v = current()
    assert verify_model_integrity(v, ROOT / 'models' / v / 'model.joblib') is True
