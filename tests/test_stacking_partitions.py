import pandas as pd
import pytest
import numpy as np

from app.train import domain_groups, split_stacking_partitions
from scripts.evaluate_calibration_thresholds import threshold_for_fpr


def test_stacking_partitions_reserve_disjoint_domains():
    rows = []
    for i in range(600):
        for sub in ('a', 'b'):
            rows.append({'_normalized_url': f'https://{sub}.site{i}.example/path', 'label': i % 2})
    frame = pd.DataFrame(rows)
    fit, calibration, threshold = split_stacking_partitions(frame)
    groups = [set(domain_groups(frame.iloc[idx])) for idx in (fit, calibration, threshold)]
    assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i + 1, 3))
    assert len(set(fit) | set(calibration) | set(threshold)) == len(frame)
    assert all(frame.iloc[idx]['label'].nunique() == 2 for idx in (fit, calibration, threshold))


def test_stacking_partitions_fail_for_single_class():
    frame = pd.DataFrame({'_normalized_url': [f'https://host{i}.example/' for i in range(500)], 'label': [1] * 500})
    with pytest.raises(ValueError, match='support'):
        split_stacking_partitions(frame)


def test_diagnostic_threshold_obeys_limit_with_ties():
    scores = np.r_[np.repeat(.1, 98), .7, .7, [.8, .9]]
    labels = np.r_[np.zeros(100, dtype=int), np.ones(2, dtype=int)]
    threshold = threshold_for_fpr(scores, labels, .01)
    assert threshold > .7
    assert ((scores[:100] >= threshold).sum() / 100) <= .01
