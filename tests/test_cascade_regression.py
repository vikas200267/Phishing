from pathlib import Path
import numpy as np
import pandas as pd
from app.registry import current
from app.train import url_series
from app.url_model import load_bundle

ROOT = Path(__file__).resolve().parents[1]
LOW = 0.10
HIGH = 0.90


def cascade_scores(model, urls):
    full = model.predict_proba(urls)
    if not getattr(model, 'fast_path_enabled', False):
        return full
    fast = model.fast_predict_proba(urls)
    shortcut = (fast <= float(model.fast_low_threshold)) | (fast >= float(model.fast_high_threshold))
    out = full.copy()
    out[shortcut] = fast[shortcut]
    return out


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1), (tp + tn) / max(len(y), 1)


def test_validated_cascade_does_not_degrade_release_gates():
    model = load_bundle(ROOT / 'models' / current() / 'model.joblib')

    frozen = pd.read_csv(ROOT / 'data' / 'frozen_regression.csv')
    frozen_urls = url_series(frozen).tolist()
    y_frozen = frozen['label'].astype(int).to_numpy()
    pp, rr, aa = metrics(y_frozen, cascade_scores(model, frozen_urls), model.threshold)
    assert pp >= 0.99
    assert rr >= 0.98
    assert aa >= 0.98

    val = pd.read_csv(ROOT / 'models' / current() / 'validation_predictions.csv')
    val_urls = val['url'].astype(str).tolist()
    y_val = val['label'].astype(int).to_numpy()
    pp, rr, aa = metrics(y_val, cascade_scores(model, val_urls), model.threshold)
    assert pp >= 0.99
    assert rr >= 0.99
    assert aa >= 0.99
