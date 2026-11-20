from scripts.capacity_plan import effective_rps

def test_capacity_factor_positive_and_correct():
    assert effective_rps(100.0, 1.30) == 100.0 / 1.30
    assert effective_rps(100.0, 0.30) == 70.0

def test_capacity_invalid_values():
    import pytest
    with pytest.raises(ValueError): effective_rps(0, 1.3)
    with pytest.raises(ValueError): effective_rps(100, 0)
