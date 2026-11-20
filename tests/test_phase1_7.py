from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]

def test_environment_examples_exist():
    assert (ROOT/'config'/'dev.env.example').exists()
    assert (ROOT/'config'/'staging.env.example').exists()
    assert (ROOT/'config'/'prod.env.example').exists()

def test_provenance_policy_forbids_synthetic_training():
    text=(ROOT/'data'/'PROVENANCE.md').read_text().lower()
    assert 'synthetic' in text and 'must not' in text

def test_current_realworld_capture_has_provenance():
    p=ROOT/'data/realworld/archive/openphish_20261005_unseen.csv.meta.json'
    assert p.exists()
    m=json.loads(p.read_text())
    assert m['source_url'].startswith('https://')
    assert m['source_capture_note'].lower().find('not visited') >= 0


def test_feature_train_serve_parity():
    import pandas as pd
    from app.features import dataframe_to_features, urls_to_features
    urls=['https://login.example.test/account?x=1','http://192.0.2.1/login']
    a=urls_to_features(urls)
    b=dataframe_to_features(pd.DataFrame({'url':urls}))
    assert list(a.columns)==list(b.columns)
    assert a.equals(b)


def test_production_training_has_no_benign_seed_or_synthetic_flag():
    meta=ROOT/'data'/'training'/'production_training.csv.meta.json'
    assert meta.exists()
    m=json.loads(meta.read_text())
    assert m.get('synthetic_training_data') is False
    assert m.get('benign_seed_allowed') is False
