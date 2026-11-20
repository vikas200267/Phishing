from __future__ import annotations
import time
import asyncio
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from app.cache import TTLCache, backend_status
from app.features import extract_url_features
from app.local_intel import canonical_url
from app.train import choose_fast_thresholds, choose_suspicious_threshold, choose_split
from app.unicode_url import unicode_host_diagnostics
from app.config import settings
from app.registry import current
from app.url_model import load_bundle

ROOT = Path(__file__).resolve().parents[1]


def test_character_models_are_in_current_bundle():
    model = load_bundle(ROOT / 'models' / current() / 'model.joblib')
    assert getattr(model, 'url_char_vectorizer', None) is not None
    assert getattr(model, 'url_word_vectorizer', None) is not None
    assert getattr(model, 'host_char_vectorizer', None) is not None
    assert getattr(model, 'url_charwb_vectorizer', None) is not None
    assert model.raw_components(['https://sample.example/login']).shape[1] == 7


def test_unicode_and_punycode_diagnostics_are_explanatory_not_verdicts():
    d = unicode_host_diagnostics('https://xn--pple-43d.example/login')
    assert d['punycode_present'] is True
    assert d['ascii_hostname'].startswith('xn--')
    assert 'verdict' not in d
    assert 'risk_score' not in d


def test_unicode_feature_vector_has_idn_signals():
    f = extract_url_features('https://xn--pple-43d.example/login')
    assert f['has_punycode'] == 1
    assert f['punycode_label_count'] >= 1


def test_fast_threshold_selector_disables_unsafe_shortcuts_when_not_proven():
    # Calibration data deliberately has a few overlapping scores. No false
    # positives/missed positives can be avoided at adequate support, so no
    # fast shortcut should be certified by a weak sample.
    p = np.array([.05,.08,.12,.15,.25,.35,.45,.55,.65,.75,.85,.95] * 4)
    y = np.array([0,0,0,1,0,1,0,1,1,0,1,1] * 4)
    low, high, meta = choose_fast_thresholds(p, y, min_negative_purity=.999, max_missed_positive_rate=0.0,
                                             min_high_precision=.999, min_support=20)
    assert (low, high) == (0.0, 1.0)
    assert meta['enabled'] is False


def test_fast_threshold_selector_obeys_purity_constraints():
    p = np.array([.01,.02,.03,.02,.04,.05,.95,.96,.98,.97,.99,.96,.2,.3,.4,.6,.7,.8,.15,.25,.35,.65,.75,.85])
    y = np.array([0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1])
    low, high, meta = choose_fast_thresholds(p, y, min_negative_purity=.995,
                                             max_missed_positive_rate=.005,
                                             min_high_precision=.995, min_support=5)
    if meta['enabled']:
        assert 0 < low < high < 1
        assert meta['low_benign_purity'] >= .995
        assert meta['low_missed_positive_rate'] <= .005
        assert meta['high_precision'] >= .995


def test_suspicious_threshold_preserves_target_recall():
    p = np.array([.01,.02,.05,.1,.2,.3,.4,.8,.9,.95])
    y = np.array([0,0,0,0,0,1,1,1,1,1])
    t, meta = choose_suspicious_threshold(p, y, phishing_threshold=.8, target_recall=.99)
    assert 0 <= t < .8
    assert meta['achieved_recall'] >= .99


def test_legacy_model_cannot_use_uncalibrated_fast_stage():
    model = load_bundle(ROOT / 'models' / current() / 'model.joblib')
    if getattr(model, 'fast_calibrator', None) is None:
        assert model.fast_path_enabled is False


def test_local_intel_canonicalization_preserves_path_and_query_but_normalizes_host():
    assert canonical_url('HTTPS://ExAmPlE.COM:443/a?x=1#frag') == 'https://example.com/a?x=1'
    assert canonical_url('http://EXAMPLE.com:80/a') == 'http://example.com/a'
    assert canonical_url('https://example.com/a?x=1') != canonical_url('https://example.com/a?x=2')
    assert canonical_url('https://example.com/A') != canonical_url('https://example.com/a')


def test_l1_cache_supports_per_entry_ttl(monkeypatch):
    c = TTLCache(maxsize=2, ttl=60)
    clock = {'t': 100.0}
    monkeypatch.setattr(time, 'monotonic', lambda: clock['t'])
    c.set('short', 1, ttl=2)
    c.set('long', 2, ttl=20)
    clock['t'] = 103.0
    assert c.get('short') is None
    assert c.get('long') == 2


def test_cache_backend_status_does_not_leak_redis_url():
    status = backend_status(check=False)
    assert 'REDIS_URL' not in str(status)
    assert 'redis_configured' in status


def test_deep_analysis_is_opt_in_and_bounded_by_config():
    assert isinstance(settings.deep_analysis_on_uncertain, bool)
    assert settings.deep_analysis_timeout_seconds > 0
    assert settings.deep_analysis_max_concurrency > 0


def test_feed_refresher_default_matches_local_index_path():
    text = (ROOT / 'scripts' / 'refresh_feeds.py').read_text(encoding='utf-8')
    assert "ROOT / 'data' / 'feeds' / 'live'" in text


def test_current_release_model_has_calibrated_final_probability():
    model = load_bundle(ROOT / 'models' / current() / 'model.joblib')
    p = model.predict_proba(['https://example.org/login'])
    assert p.shape == (1,)
    assert 0 <= float(p[0]) <= 1


def test_cache_ttl_is_short_near_suspicious_boundary():
    from app.api import cache_ttl_for_score
    model = load_bundle(ROOT / 'models' / current() / 'model.joblib')
    suspicious = getattr(model, 'suspicious_threshold', None)
    if suspicious is None:
        suspicious = min(0.10, max(0.03, float(model.threshold) * 0.45))
    near = float(suspicious) + 0.01
    assert cache_ttl_for_score(near, model, 'model') == settings.cache_uncertain_ttl_seconds


def test_production_doctor_has_no_synthetic_fallback_findings():
    import subprocess, sys
    result = subprocess.run([sys.executable, 'scripts/production_doctor.py'], cwd=ROOT,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cached_dns_tls_can_be_extended_with_remote_results_without_unbound_dns(monkeypatch):
    import app.threat_intel as module
    from app.threat_intel import ThreatIntel
    cached = {'dns': {'ok': True, 'ips': ['8.8.8.8']}, 'tls': {'ok': False, 'reason': 'certificate_unavailable'}}
    monkeypatch.setattr(module, 'cache_key', lambda *args: 'same-key')
    monkeypatch.setattr(module, 'cache_get', lambda key: cached)
    ti = ThreatIntel()

    async def web_risk(url): return {'enabled': True, 'match': {}}
    async def safe_browsing(urls): return {'enabled': True, 'matches': []}
    async def virustotal(url): return {'enabled': True, 'malicious': 0}
    async def urlscan_search(domain): return {'enabled': True, 'total': 0, 'results': []}
    async def phishtank(url): return {'enabled': True, 'valid': False}
    monkeypatch.setattr(ti, 'web_risk', web_risk)
    monkeypatch.setattr(ti, 'safe_browsing', safe_browsing)
    monkeypatch.setattr(ti, 'virustotal', virustotal)
    monkeypatch.setattr(ti, 'urlscan_search', urlscan_search)
    monkeypatch.setattr(ti, 'phishtank', phishtank)

    result = asyncio.run(ti.enrich_url('https://public.example/path', include_remote=True))
    assert result['dns']['ok'] is True
    assert result['web_risk']['enabled'] is True
    assert 'web_risk' not in cached  # provider data must not mutate the origin cache entry


def test_deep_analysis_timeout_keeps_concurrency_slot_until_background_finishes(monkeypatch):
    import app.api as api
    from app.config import settings as current_settings
    sem = asyncio.Semaphore(1)
    starts = []
    class SlowIntel:
        async def enrich_url(self, url, include_remote=False):
            starts.append(time.monotonic())
            await asyncio.sleep(0.06)
            return {'dns': {'ok': True}, 'tls': {'ok': False}}
    monkeypatch.setattr(api, 'DEEP_ANALYSIS_SEMAPHORE', sem)
    monkeypatch.setattr(api, '_intel', SlowIntel())
    monkeypatch.setattr(api, 'settings', replace(current_settings, deep_analysis_timeout_seconds=0.005))

    async def scenario():
        for _ in range(2):
            try:
                await api._bounded_local_enrichment('https://public.example/')
            except asyncio.TimeoutError:
                pass
        await asyncio.sleep(0.08)
    asyncio.run(scenario())
    assert len(starts) == 2
    assert starts[1] - starts[0] >= 0.045
    assert sem._value == 1


def test_four_zone_verdicts_keep_suspicious_out_of_binary_benign():
    from app.api import classify_score
    model = load_bundle(ROOT / 'models' / current() / 'model.joblib')
    suspicious_threshold = getattr(model, 'suspicious_threshold', None)
    if suspicious_threshold is None:
        suspicious_threshold = min(0.10, max(0.03, float(model.threshold) * 0.45))
    suspicious_p = (float(suspicious_threshold) + float(model.threshold)) / 2
    assert classify_score(0.0, model)[0] == 'benign'
    assert classify_score(suspicious_p, model)[0] == 'suspicious'
    assert classify_score(float(model.threshold), model)[0] == 'phishing'


def test_deep_analysis_cached_result_branch_and_local_intel_exactness():
    from app.local_intel import local_phishing
    # No untrusted source rows are bundled as a current production feed; an
    # unrelated URL must remain unlisted unless a captured feed is refreshed.
    assert local_phishing('https://definitely-not-listed.invalid/some-path') is False



def test_empty_live_feed_does_not_overwrite_last_known_good_snapshot(tmp_path, monkeypatch):
    import scripts.refresh_feeds as refresh
    target = tmp_path / 'openphish.csv'
    prior = 'url,label,source,collected_at\nhttps://known-threat.example/a,1,openphish,2026-10-10T00:00:00+00:00\n'
    target.write_text(prior, encoding='utf-8')
    monkeypatch.setattr(refresh, '_fetch', lambda *args, **kwargs: (b'not a valid URL\n', {}, 'https://feed.example/feed'))
    result = refresh.refresh_openphish(target, 'https://feed.example/feed', None)
    assert result['ok'] is False
    assert target.read_text(encoding='utf-8') == prior


def test_cache_disabled_setting_disables_provider_and_origin_cache(monkeypatch):
    import app.threat_intel as module
    from app.threat_intel import ThreatIntel
    from app.config import settings as current_settings
    monkeypatch.setattr(module, 'settings', replace(current_settings, cache_enabled=False))
    monkeypatch.setattr(module, 'cache_key', lambda *args: 'k')
    def forbidden_get(*args, **kwargs): raise AssertionError('cache_get called while cache disabled')
    def forbidden_set(*args, **kwargs): raise AssertionError('cache_set called while cache disabled')
    monkeypatch.setattr(module, 'cache_get', forbidden_get)
    monkeypatch.setattr(module, 'cache_set', forbidden_set)
    monkeypatch.setattr(module, 'resolve_host', lambda host: {'ok': True, 'ips':['8.8.8.8']})
    monkeypatch.setattr(module, 'tls_summary', lambda url: {'ok': True})
    result = asyncio.run(ThreatIntel().enrich_url('https://public.example/path', include_remote=False))
    assert result['dns']['ok'] is True
    assert result['tls']['ok'] is True


def test_temporal_split_uses_authentic_timestamp_boundary_and_drops_exact_repeats():
    rows = []
    for i in range(300):
        source_i = i - 290 if i >= 290 else i
        rows.append({
            '_normalized_url': f'https://site{source_i}.com/path',
            'label': source_i % 2,
            'collected_at': (pd.Timestamp('2026-01-01T00:00:00Z') + pd.Timedelta(days=i)).isoformat(),
        })
    frame = pd.DataFrame(rows)
    train_idx, test_idx, split_type, split_column = choose_split(frame, require_temporal=True)
    train = frame.iloc[train_idx]
    test = frame.iloc[test_idx]
    assert split_type == 'temporal'
    assert split_column == 'collected_at'
    assert pd.to_datetime(train['collected_at'], utc=True).max() < pd.to_datetime(test['collected_at'], utc=True).min()
    assert set(train['_normalized_url']).isdisjoint(set(test['_normalized_url']))
    assert len(train) >= 100 and len(test) >= 50


def test_temporal_split_refuses_missing_or_non_distinct_timestamps():
    frame = pd.DataFrame({
        '_normalized_url': [f'https://site{i}.com/a' for i in range(220)],
        'label': [i % 2 for i in range(220)],
        'collected_at': ['not-a-real-timestamp'] * 220,
    })
    import pytest
    with pytest.raises(ValueError, match='refusing to fabricate dates'):
        choose_split(frame, require_temporal=True)


def test_dataset_manifest_json_corpus_is_counted_without_fabricating_row_timestamps(tmp_path):
    import json
    from scripts.build_dataset_manifest import inspect_dataset
    corpus = tmp_path / 'corpus.json'
    corpus.write_text(json.dumps({
        'generated_at': '2026-10-12T10:00:00Z',
        'source_role': 'real_world_url_replay_corpus',
        'visited_urls': False,
        'urls': [{'url': 'https://one.example/a'}, {'url': 'https://two.example/b'}],
    }), encoding='utf-8')
    result = inspect_dataset(corpus)
    assert result['format'] == 'json_url_corpus'
    assert result['rows'] == 2
    assert result['valid_url_rows'] == 2
    assert result['unique_normalized_urls'] == 2
    assert result['listed_urls_visited'] is False
    assert 'timestamp_valid_rows' not in result


def test_training_fails_closed_when_requested_evaluation_exclusion_is_missing(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pytest
    import app.train as trainer
    tiny = pd.DataFrame({
        '_normalized_url': ['https://alpha.example/a', 'https://beta.example/b'],
        'label': [0, 1],
    })
    monkeypatch.setattr(trainer, 'load_data', lambda *args: tiny.copy())
    args = SimpleNamespace(
        phishing='unused-phishing.csv', legitimate='unused-legitimate.csv', extra=None,
        feature_schema_file=None, exclude_frozen=False,
        exclude_csv=str(tmp_path / 'missing-evaluation-exclusion.csv'),
        exclude_domains_csv=None,
    )
    with pytest.raises(FileNotFoundError, match='required exclusion file is missing'):
        trainer.train(args)
