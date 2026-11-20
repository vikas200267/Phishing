from app.policy import valid_http_url

def test_policy_is_validation_only():
    assert valid_http_url('https://www.google.com/')
    assert not valid_http_url('javascript:alert(1)')
    assert not valid_http_url('https://user:pass@example.com/')
