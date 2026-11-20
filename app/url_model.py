from __future__ import annotations
from dataclasses import dataclass
import joblib, numpy as np, pandas as pd
from scipy.sparse import hstack
from scipy.special import expit
from .features import MODEL_FEATURE_NAMES, dataframe_to_features, urls_to_features, normalize_url
from .policy import valid_http_url
from .config import settings
from .registry import verify_model_integrity


def _meta_features(comps: np.ndarray) -> np.ndarray:
    if comps.size == 0:
        return np.empty((0, 12 if comps.shape[1] >= 6 else 9), dtype=np.float32)
    mean = comps.mean(axis=1)
    std = comps.std(axis=1)
    mx = comps.max(axis=1)
    mn = comps.min(axis=1)
    q25 = np.quantile(comps, .25, axis=1)
    q75 = np.quantile(comps, .75, axis=1)
    if comps.shape[1] >= 6:
        return np.column_stack([comps, mean, std, mx, mn, q25, q75]).astype(np.float32)
    return np.column_stack([comps, mean, std, mx, mn]).astype(np.float32)


@dataclass
class URLModelBundle:
    xgb_model: object
    lgbm_model: object
    extra_model: object
    text_url_model: object
    text_host_model: object
    svc_url_model: object | None
    url_char_vectorizer: object
    url_word_vectorizer: object
    host_char_vectorizer: object
    svc_char_vectorizer: object | None
    meta_model: object
    calibrator: object
    threshold: float
    feature_names: list[str]
    version: str
    url_charwb_model: object | None = None
    url_charwb_vectorizer: object | None = None
    # Separately calibrated fast-stage probabilities. Legacy models omit this
    # field and are deliberately not permitted to short-circuit the ensemble.
    fast_calibrator: object | None = None
    fast_low_threshold: float = 0.0
    fast_high_threshold: float = 1.0
    fast_calibration_metadata: dict | None = None
    suspicious_threshold: float | None = None
    fast_path_validated: bool = False

    def _structured(self, urls: list[str]):
        return urls_to_features(urls)[self.feature_names].astype(np.float32)

    def _text(self, urls: list[str]):
        normalized = [normalize_url(u) for u in urls]
        from urllib.parse import urlparse
        hosts = [(urlparse(u).hostname or '').lower() for u in normalized]
        url_char = self.url_char_vectorizer.transform(normalized)
        url_word = self.url_word_vectorizer.transform(normalized)
        host_char = self.host_char_vectorizer.transform(hosts)
        svc_char = self.svc_char_vectorizer.transform(normalized) if getattr(self, 'svc_char_vectorizer', None) is not None else None
        charwb = self.url_charwb_vectorizer.transform(normalized) if getattr(self, 'url_charwb_vectorizer', None) is not None else None
        return hstack([url_char, url_word], format='csr'), host_char, svc_char, charwb

    def raw_components(self, urls: list[str]):
        n = len(urls)
        if n == 0:
            ncomp=5 + int(getattr(self, 'svc_url_model', None) is not None) + int(getattr(self, 'url_charwb_model', None) is not None)
            return np.empty((0, ncomp))
        structured = self._structured(urls)
        text_url, text_host, text_svc, text_charwb = self._text(urls)
        p_xgb = self.xgb_model.predict_proba(structured)[:, 1]
        p_lgbm = self.lgbm_model.predict_proba(structured)[:, 1]
        p_extra = self.extra_model.predict_proba(structured)[:, 1]
        p_url = self.text_url_model.predict_proba(text_url)[:, 1]
        p_host = self.text_host_model.predict_proba(text_host)[:, 1]
        comps = [p_xgb, p_lgbm, p_extra, p_url, p_host]
        svc = getattr(self, 'svc_url_model', None)
        if svc is not None and text_svc is not None:
            comps.append(expit(np.clip(svc.decision_function(text_svc), -20, 20)))
        charwb_model=getattr(self, 'url_charwb_model', None)
        if charwb_model is not None and text_charwb is not None:
            comps.append(charwb_model.predict_proba(text_charwb)[:,1])
        return np.column_stack(comps)

    @property
    def fast_path_enabled(self) -> bool:
        """Fast decisions are enabled only when a dedicated calibrator and
        non-degenerate calibration-derived operating bounds are present.
        Old bundles fall back to the full calibrated ensemble.
        """
        calibrator = getattr(self, 'fast_calibrator', None)
        try:
            low = float(getattr(self, 'fast_low_threshold', 0.0))
            high = float(getattr(self, 'fast_high_threshold', 1.0))
        except (TypeError, ValueError):
            return False
        validated = bool(getattr(self, 'fast_path_validated', False))
        return validated and calibrator is not None and 0.0 < low < high < 1.0

    def fast_predict_proba(self, urls: list[str]) -> np.ndarray:
        """Return calibrated first-stage probabilities.

        If a legacy artifact lacks a dedicated fast-stage calibrator, return the raw
        XGBoost probability for diagnostics only. The API must not use it for a
        final verdict (``fast_path_enabled`` will be false).
        """
        bad=[u for u in urls if not valid_http_url(u)]
        if bad:
            raise ValueError(f'Only valid http/https URLs are supported; invalid count={len(bad)}')
        if not urls:
            return np.empty((0,))
        raw = self.xgb_model.predict_proba(self._structured(urls))[:, 1]
        calibrator = getattr(self, 'fast_calibrator', None)
        if calibrator is None:
            return raw
        clipped = np.clip(raw, 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        return calibrator.predict_proba(logits)[:, 1]

    def predict_proba(self, urls: list[str]) -> np.ndarray:
        bad=[u for u in urls if not valid_http_url(u)]
        if bad:
            raise ValueError(f'Only valid http/https URLs are supported; invalid count={len(bad)}')
        comps = self.raw_components(urls)
        if len(comps) == 0:
            return np.empty((0,))
        meta = _meta_features(comps)
        raw = self.meta_model.predict_proba(meta)[:, 1]
        return self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]

    def predict(self, urls: list[str]):
        p = self.predict_proba(urls)
        return (p >= self.threshold).astype(int)


def save_bundle(path, bundle):
    joblib.dump(bundle, path, compress=3)


def _limit_inference_threads(bundle):
    """Prevent worker x model thread oversubscription in horizontally scaled serving."""
    n=max(1,int(settings.inference_threads))
    for name in ('xgb_model','lgbm_model','extra_model','text_url_model','text_host_model','svc_url_model','url_charwb_model','meta_model'):
        obj=getattr(bundle,name,None)
        if obj is None: continue
        try:
            params=obj.get_params() if hasattr(obj,'get_params') else {}
            if 'n_jobs' in params: obj.set_params(n_jobs=n)
            elif hasattr(obj,'n_jobs'): obj.n_jobs=n
        except Exception:
            pass
    return bundle

def load_bundle(path):
    path = __import__('pathlib').Path(path)
    if path.name != 'model.joblib':
        raise RuntimeError(f'expected model.joblib, got {path.name}')
    if settings.require_model_manifest:
        verify_model_integrity(path.parent.name, path)
    bundle = joblib.load(path)
    version = str(getattr(bundle, 'version', ''))
    if version != path.parent.name:
        raise RuntimeError(f'model version mismatch: artifact={version!r}, directory={path.parent.name!r}')
    names = list(getattr(bundle, 'feature_names', []) or [])
    if not names or len(names) != len(set(names)):
        raise RuntimeError('invalid feature schema in model bundle')
    required = ('xgb_model','lgbm_model','extra_model','text_url_model','text_host_model','meta_model','calibrator')
    if any(getattr(bundle, name, None) is None for name in required):
        raise RuntimeError('model bundle is missing required components')
    return _limit_inference_threads(bundle)
