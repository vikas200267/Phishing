from app.features import extract_url_features, FEATURE_NAMES

def test_features_complete():
    x=extract_url_features('https://login-example.com/account/verify?x=1')
    assert set(FEATURE_NAMES)==set(x)
    assert x['has_https']==1
    assert x['suspicious_token_count']>=2

def test_ip_detection():
    x=extract_url_features('http://192.0.2.1/login')
    assert x['has_ip']==1
