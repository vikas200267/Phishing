from __future__ import annotations

import asyncio
import functools
import json
import hmac
import random
import re
import time
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from typing import Any, Optional, Union

import joblib
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, ORJSONResponse
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import metrics as prom
from .cache import backend_status as cache_backend_status, cache_key, get as cache_get, set as cache_set
from .config import settings
from .email_features import extract_email_features
from .email_model import predict_email as email_model_predict
from .feedback import add as add_feedback
from .features import normalize_url
from .unicode_url import unicode_host_diagnostics
from .local_intel import (
    local_intel_metadata,
    local_phishing,
    maybe_reload_local_intel,
    reload_local_intel,
)
from .policy import valid_http_url
from .trusted_homepages import is_trusted_official_homepage
from .reputation import reload_verified_feedback, verified_verdicts
from .threat_intel import ThreatIntel
from .url_model import load_bundle
from .registry import artifact_sha256, verify_artifact_integrity

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'models'
MODEL_LOCK = RLock()

MODEL = None
MODEL_VERSION = None
EMAIL_MODEL = None
SHADOW_MODEL = None
SHADOW_VERSION = None
INFERENCE_EXECUTOR = ThreadPoolExecutor(max_workers=max(1, settings.inference_executor_workers), thread_name_prefix='phishguard-infer')
DEEP_ANALYSIS_SEMAPHORE = asyncio.Semaphore(max(1, settings.deep_analysis_max_concurrency))


def _current_state():
    with MODEL_LOCK:
        return MODEL, MODEL_VERSION, EMAIL_MODEL, SHADOW_MODEL, SHADOW_VERSION


def _swap_state(model, version, email_model, shadow_model, shadow_version):
    global MODEL, MODEL_VERSION, EMAIL_MODEL, SHADOW_MODEL, SHADOW_VERSION
    with MODEL_LOCK:
        MODEL, MODEL_VERSION, EMAIL_MODEL = model, version, email_model
        SHADOW_MODEL, SHADOW_VERSION = shadow_model, shadow_version


def load_current():
    pointer_path = MODEL_DIR / 'CURRENT'
    pointer = pointer_path.read_text(encoding='utf-8').strip()
    if not pointer:
        raise RuntimeError('models/CURRENT is empty')

    # Load everything before swapping the live state. A bad artifact cannot
    # partially replace a healthy model.
    model = load_bundle(MODEL_DIR / pointer / 'model.joblib')
    blocking_approval = None
    if settings.ml_blocking_enabled:
        approval_path = MODEL_DIR / pointer / 'BLOCKING_APPROVED.json'
        try:
            blocking_approval = json.loads(approval_path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise RuntimeError('ML_BLOCKING_ENABLED requires a valid promotion-generated BLOCKING_APPROVED.json') from exc
        challenge = blocking_approval.get('legitimate_challenge', {})
        approval_gates = blocking_approval.get('passed_gates', {})
        required_approval_gates = {
            'legitimate_challenge_zero_false_positives',
            'legitimate_challenge_minimum_coverage',
            'legitimate_challenge_training_disjoint',
            'legitimate_challenge_provenance',
            'legitimate_challenge_no_duplicate_urls',
            'outer_precision', 'outer_recall', 'frozen_precision', 'frozen_recall',
        }
        if (
            blocking_approval.get('schema_version') != 1
            or blocking_approval.get('approved') is not True
            or blocking_approval.get('model_version') != pointer
            or blocking_approval.get('model_sha256') != artifact_sha256(MODEL_DIR / pointer / 'model.joblib')
            or blocking_approval.get('challenge_sha256') != challenge.get('challenge_sha256')
            or not isinstance(approval_gates, dict)
            or not required_approval_gates.issubset(approval_gates)
            or not all(value is True for value in approval_gates.values())
            or challenge.get('unique_registrable_domains', 0) < 15
            or challenge.get('false_positives', 1) != 0
            or challenge.get('exact_training_overlap', 1) != 0
            or challenge.get('domain_training_overlap', 1) != 0
            or challenge.get('observed_at_coverage', 0) < 0.8
            or challenge.get('provenance_sha256_match') is not True
        ):
            raise RuntimeError('ML blocking approval does not match the active model and required challenge gates')
    email_path = MODEL_DIR / pointer / 'email_model.joblib'
    email_model = None
    if email_path.exists():
        if settings.require_model_manifest:
            verify_artifact_integrity(f'{pointer}:email_model', email_path)
        email_model = joblib.load(email_path)

    shadow_model = None
    shadow_version = None
    configured_shadow = settings.shadow_model_version
    shadow_file = MODEL_DIR / 'SHADOW'
    if not configured_shadow and shadow_file.exists():
        configured_shadow = shadow_file.read_text(encoding='utf-8').strip() or None
    if configured_shadow and configured_shadow != pointer and settings.shadow_sample_rate > 0:
        shadow_path = MODEL_DIR / configured_shadow / 'model.joblib'
        if shadow_path.exists():
            try:
                shadow_model = load_bundle(shadow_path)
                shadow_version = configured_shadow
            except Exception:
                shadow_model, shadow_version = None, None

    _swap_state(model, pointer, email_model, shadow_model, shadow_version)
    return pointer


async def _run_inference(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(INFERENCE_EXECUTOR, functools.partial(fn, *args))


@asynccontextmanager
async def lifespan(_app):
    intel = ThreatIntel()
    await intel.start()
    global _intel
    _intel = intel
    task = asyncio.create_task(_intel_watcher())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await intel.close()


async def _intel_watcher():
    while True:
        await asyncio.sleep(max(5, settings.local_intel_reload_seconds))
        try:
            maybe_reload_local_intel()
        except Exception:
            pass


async def _bounded_local_enrichment(url: str):
    """Run bounded DNS/TLS enrichment without leaking capacity on timeout.

    `asyncio.to_thread` cannot stop an already-running OS resolver call. The
    semaphore therefore remains acquired until the underlying analysis task
    actually finishes, even if the HTTP request times out or is cancelled.
    """
    await DEEP_ANALYSIS_SEMAPHORE.acquire()
    task = None
    release_in_callback = False
    try:
        task = asyncio.create_task(_intel.enrich_url(url, include_remote=False))
        return await asyncio.wait_for(
            asyncio.shield(task), timeout=settings.deep_analysis_timeout_seconds
        )
    except (asyncio.TimeoutError, asyncio.CancelledError):
        if task is not None:
            release_in_callback = True
            def _finish_background_task(done_task):
                try:
                    done_task.result()
                except BaseException:
                    # The background error is surfaced as an unavailable result
                    # to the request already timed out; consume it to avoid an
                    # unhandled-task warning while releasing only when finished.
                    pass
                finally:
                    DEEP_ANALYSIS_SEMAPHORE.release()
            task.add_done_callback(_finish_background_task)
        raise
    finally:
        if not release_in_callback:
            DEEP_ANALYSIS_SEMAPHORE.release()


app = FastAPI(
    title='PhishGuard Inference API',
    version=settings.app_version,
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)
_intel = ThreatIntel()
settings.validate_startup()
load_current()


def _check_key(provided: str | None, expected: str | None, label: str):
    if not settings.auth_required:
        return
    if not expected:
        raise HTTPException(status_code=503, detail=f'{label} authentication is not configured')
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail='invalid API key')


def require_inference_key(x_api_key: str | None = Header(default=None, alias='X-API-Key')):
    _check_key(x_api_key, settings.api_key, 'Inference API')


def require_admin_key(x_admin_key: str | None = Header(default=None, alias='X-Admin-Key')):
    _check_key(x_admin_key, settings.admin_api_key, 'Admin API')


class URLRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    url: str = Field(min_length=8, max_length=8192)
    metadata: Optional[dict[str, Any]] = None
    enrich: bool = False

    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str):
        v = v.strip()
        if not valid_http_url(v):
            raise ValueError('url must be a valid http/https URL')
        return v


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    urls: list[Union[str, URLRequest]] = Field(min_length=1, max_length=settings.max_batch_size)

    @field_validator('urls', mode='before')
    @classmethod
    def validate_batch_urls(cls, values):
        if not isinstance(values, list):
            raise ValueError('urls must be a list')
        cleaned = []
        for item in values:
            value = item.get('url') if isinstance(item, dict) else item
            if not isinstance(value, str) or len(value) > 8192 or not valid_http_url(value.strip()):
                raise ValueError('every batch item must contain a valid http/https URL (maximum 8192 characters)')
            cleaned.append({**item, 'url': value.strip()} if isinstance(item, dict) else value.strip())
        return cleaned


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    url: str = Field(min_length=8, max_length=8192)
    label: int = Field(ge=0, le=1)
    source: str = Field(default='manual', max_length=64)
    verified: bool = False
    metadata: Optional[dict[str, Any]] = None

    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str):
        v = v.strip()
        if not valid_http_url(v):
            raise ValueError('url must be a valid http/https URL')
        return v


class EmailRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    subject: str = Field(default='', max_length=10000)
    sender: str = Field(default='', max_length=2048)
    reply_to: str = Field(default='', max_length=2048)
    return_path: str = Field(default='', max_length=2048)
    body: str = Field(default='', max_length=settings.max_email_body_chars)
    headers: dict[str, str] = Field(default_factory=dict)


class EnrichRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    url: str = Field(min_length=8, max_length=8192)

    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str):
        v = v.strip()
        if not valid_http_url(v):
            raise ValueError('url must be a valid http/https URL')
        return v


class _BodyTooLarge(Exception):
    pass


class BodyLimitMiddleware:
    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = int(max_bytes)

    async def __call__(self, scope, receive, send):
        if scope.get('type') != 'http' or scope.get('method') not in {'POST','PUT','PATCH'}:
            return await self.app(scope, receive, send)
        total = 0
        async def limited_receive():
            nonlocal total
            message = await receive()
            if message.get('type') == 'http.request':
                total += len(message.get('body', b''))
                if total > self.max_bytes:
                    raise _BodyTooLarge
            return message
        try:
            return await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            await send({'type':'http.response.start','status':413,'headers':[(b'content-type',b'application/json')]})
            await send({'type':'http.response.body','body':b'{\"detail\":\"request body too large\"}'})


app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_request_body_bytes)


@app.get('/', include_in_schema=False)
def console():
    return FileResponse(
        ROOT / 'app' / 'static' / 'index.html',
        headers={
            'Cache-Control': 'no-store',
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'Referrer-Policy': 'no-referrer',
        },
    )


@app.middleware('http')
async def metrics_middleware(request: Request, call_next):
    started = time.perf_counter()
    status = '500'
    try:
        content_length = request.headers.get('content-length')
        if content_length:
            try:
                if int(content_length) > settings.max_request_body_bytes:
                    status = '413'
                    return JSONResponse(status_code=413, content={'detail': 'request body too large'})
            except ValueError:
                status = '400'
                return JSONResponse(status_code=400, content={'detail': 'invalid content-length'})
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        if prom.REQUESTS:
            prom.REQUESTS.labels(request.url.path, status).inc()
        if prom.LATENCY:
            prom.LATENCY.labels(request.url.path).observe(time.perf_counter() - started)


def risk_band(p: float, model) -> str:
    threshold = float(model.threshold)
    configured = getattr(model, 'suspicious_threshold', None)
    suspicious = min(0.10, max(0.03, threshold * 0.45)) if configured is None else float(configured)
    if p >= 0.90:
        return 'critical'
    if p >= threshold:
        return 'high'
    if p >= suspicious:
        return 'medium'
    return 'low'


def classify_score(p: float, model):
    threshold = float(model.threshold)
    configured = getattr(model, 'suspicious_threshold', None)
    suspicious = min(0.10, max(0.03, threshold * 0.45)) if configured is None else float(configured)
    if p >= threshold:
        return 'phishing', 'high-confidence'
    if p >= suspicious:
        return 'suspicious', 'abstain-review'
    return 'benign', 'low-risk'


def classify_email_score(p: float):
    """Map email model scores to conservative advisory bands."""
    if p >= settings.email_phishing_threshold:
        return 'phishing', 'email-score-above-threshold'
    if p >= settings.email_suspicious_threshold:
        return 'suspicious', 'email-score-needs-review'
    return 'benign', 'email-score-below-review-threshold'


def classify_fast_score(p: float, model=None):
    # A real request passes its model bundle, whose bounds were selected from a
    # held-out calibration fold. The settings fallback exists only for helper
    # compatibility/tests and is not used for legacy-model inference.
    low = float(getattr(model, 'fast_low_threshold', settings.cascade_low_threshold))
    high = float(getattr(model, 'fast_high_threshold', settings.cascade_high_threshold))
    if p <= low:
        return 'benign', 'fast-high-confidence-benign'
    if p >= high:
        return 'phishing', 'fast-high-confidence-phishing'
    return 'suspicious', 'full-model-required'


def apply_official_homepage_policy(result: dict, url: str, stage: str) -> bool:
    """Return a cautious benign decision for exact, verified HTTPS homepages.

    Exact feedback and threat-intelligence matches always take precedence. The
    raw model score stays visible so this policy correction is auditable.
    """
    if stage in {'verified_feedback', 'local_intel'} or not is_trusted_official_homepage(url):
        return False
    raw_score = result.get('risk_score')
    result.update({
        'model_verdict': result.get('verdict'),
        'model_risk_score': raw_score,
        'verdict': 'benign',
        'binary_verdict': 'benign',
        'decision': 'verified_official_homepage_root',
        'risk_band': 'low',
        'risk_score': 0.0,
        'score_type': 'policy_adjusted',
        'policy_source': 'data/realworld/trusted_homepages.csv',
        'blocking_eligible': False,
    })
    return True


def confirmed_intel(enrichment: dict) -> bool:
    wr = enrichment.get('web_risk', {}) or {}
    vt = enrichment.get('virustotal', {}) or {}
    pt = enrichment.get('phishtank', {}) or {}
    sb = enrichment.get('safe_browsing', {}) or {}
    return bool(
        (wr.get('match') or {})
        or int(vt.get('malicious', 0) or 0) > 0
        or bool(pt.get('valid'))
        or bool(pt.get('verified'))
        or bool(sb.get('matches'))
    )


def cache_ttl_for_score(score: float, model, stage: str) -> int:
    """Cache high-margin decisions longer and near-threshold scores briefly."""
    if stage == 'fast':
        low = float(getattr(model, 'fast_low_threshold', 0.0))
        high = float(getattr(model, 'fast_high_threshold', 1.0))
        confident = (low > 0 and score <= low) or (high < 1 and score >= high)
    else:
        threshold = float(getattr(model, 'threshold', 0.5))
        configured = getattr(model, 'suspicious_threshold', None)
        suspicious = min(0.10, max(0.03, threshold * 0.45)) if configured is None else float(configured)
        # Use the nearest decision boundary: scores near either the suspicious
        # or phishing threshold should expire quickly so changing evidence can
        # correct stale cached decisions promptly.
        margin = min(abs(float(score) - threshold), abs(float(score) - suspicious))
        confident = margin >= 0.05
    return max(1, settings.cache_confident_ttl_seconds if confident else settings.cache_uncertain_ttl_seconds)



def predict_urls(urls: list[str], model):
    if not urls:
        return [], 0, 0, [], []
    unique = list(dict.fromkeys(urls))
    index = {u: i for i, u in enumerate(unique)}
    vals: list[float | None] = [None] * len(unique)
    stages = ['unknown'] * len(unique)
    hits = 0
    sources: dict[str, str] = {}
    missing: list[tuple[int, str]] = []
    known = verified_verdicts(unique)

    for i, u in enumerate(unique):
        norm = normalize_url(u)
        if known.get(norm) is not None:
            vals[i] = 1.0 if known[norm] == 1 else 0.0
            stages[i] = 'verified_feedback'
            sources[u] = 'verified_feedback'
            continue
        if settings.local_intel_enabled and local_phishing(u):
            vals[i] = 1.0
            stages[i] = 'local_intel'
            sources[u] = 'local_phishing_feed'
            continue
        if settings.cache_enabled:
            k = cache_key('url:' + getattr(model, 'version', 'model'), norm)
            cached = cache_get(k)
            if cached is not None:
                if isinstance(cached, dict) and 'probability' in cached:
                    vals[i] = float(cached['probability'])
                    cached_stage = str(cached.get('stage', 'model'))
                    stages[i] = 'fast' if cached_stage == 'fast' and getattr(model, 'fast_path_enabled', False) else 'cache'
                else:  # Backward-compatible values written by earlier releases.
                    vals[i] = float(cached)
                    stages[i] = 'cache'
                sources[u] = 'model_cache'
                hits += 1
                continue
        missing.append((i, u))

    # Fast structured cascade. Only the XGBoost first-stage score is evaluated
    # for clear cases; ambiguous inputs still use the full calibrated ensemble.
    cache_miss_count = 0
    if missing:
        for start in range(0, len(missing), settings.inference_chunk_size):
            chunk = missing[start:start + settings.inference_chunk_size]
            cache_miss_count += len(chunk)
            chunk_urls = [u for _, u in chunk]
            if settings.cascade_enabled and bool(getattr(model, 'fast_path_enabled', False)):
                fast = model.fast_predict_proba(chunk_urls)
                uncertain = []
                for j, ((i, u), p0) in enumerate(zip(chunk, fast)):
                    p0 = float(p0)
                    if p0 <= float(model.fast_low_threshold) or p0 >= float(model.fast_high_threshold):
                        vals[i] = p0
                        stages[i] = 'fast'
                        sources[u] = 'calibrated_fast_structured_model'
                        if settings.cache_enabled:
                            cache_set(cache_key('url:' + getattr(model, 'version', 'model'), normalize_url(u)),
                                      {'probability': float(p0), 'stage': 'fast', 'score_type': 'calibrated_fast_stage'},
                                      ttl=cache_ttl_for_score(float(p0), model, 'fast'))
                    else:
                        uncertain.append((j, i, u))
                if uncertain:
                    full_urls = [u for _, _, u in uncertain]
                    full_probs = model.predict_proba(full_urls)
                    for p, (_, i, u) in zip(full_probs, uncertain):
                        vals[i] = float(p)
                        stages[i] = 'model'
                        sources[u] = 'model'
                        if settings.cache_enabled:
                            cache_set(cache_key('url:' + getattr(model, 'version', 'model'), normalize_url(u)),
                                      {'probability': float(p), 'stage': 'model', 'score_type': 'calibrated'},
                                      ttl=cache_ttl_for_score(float(p), model, 'model'))
            else:
                full_probs = model.predict_proba(chunk_urls)
                for (i, u), p in zip(chunk, full_probs):
                    vals[i] = float(p)
                    stages[i] = 'model'
                    sources[u] = 'model'
                    if settings.cache_enabled:
                        cache_set(cache_key('url:' + getattr(model, 'version', 'model'), normalize_url(u)),
                                  {'probability': float(p), 'stage': 'model', 'score_type': 'calibrated'},
                                  ttl=cache_ttl_for_score(float(p), model, 'model'))

    if prom.CACHE_MISSES and cache_miss_count:
        prom.CACHE_MISSES.inc(cache_miss_count)
    if prom.CACHE_HITS and hits:
        prom.CACHE_HITS.inc(hits)
    return (
        [float(vals[index[u]]) for u in urls],
        hits,
        len(unique),
        [sources.get(u, 'model') for u in urls],
        [stages[index[u]] for u in urls],
    )


@app.get('/health')
def health():
    model, version, email_model, shadow_model, shadow_version = _current_state()
    intel_status = local_intel_metadata()
    refresh = intel_status.get('refresh') or {}
    providers = refresh.get('providers') or {}
    # Freshness of one local feed must not imply that every configured provider
    # is active. A skipped or failed provider makes coverage partial.
    provider_partial = any(not isinstance(item, dict) or item.get('ok') is not True for item in providers.values())
    intel_state = (
        'stale_or_unavailable' if intel_status.get('stale', True)
        else 'partial' if provider_partial
        else 'fresh'
    )
    return {
        'status': 'ok' if model is not None else 'degraded',
        'ml_blocking_enabled': settings.ml_blocking_enabled,
        'auth_required': settings.auth_required,
        'model_version': version,
        'email_model_loaded': email_model is not None,
        'local_intel_enabled': settings.local_intel_enabled,
        'shadow_enabled': bool(shadow_model is not None),
        'fast_path_enabled': bool(getattr(model, 'fast_path_enabled', False)),
        'fast_path_thresholds': {'low': getattr(model, 'fast_low_threshold', None), 'high': getattr(model, 'fast_high_threshold', None)},
        'cache': cache_backend_status(check=False),
        'deep_analysis_on_uncertain': settings.deep_analysis_on_uncertain,
        'local_intel': intel_status,
        'threat_intel_status': intel_state,
        'threat_intel_providers': providers,
    }


@app.get('/ready')
def ready():
    model, version, *_ = _current_state()
    if model is None:
        raise HTTPException(status_code=503, detail='model unavailable')
    return {'ready': True, 'model_version': version}


@app.get('/v1/evaluation/latest', dependencies=[Depends(require_admin_key)])
def latest_evaluation():
    """Return the most recent stored real-world evaluation artifact.

    This endpoint only exposes recorded evaluation evidence; it never generates
    simulated metrics or claims that a threat-feed-only sample estimates precision.
    """
    root = ROOT / 'artifacts'
    candidates = []
    for path in root.rglob('metrics.json') if root.exists() else []:
        if 'realworld' in str(path).lower():
            try:
                candidates.append(path)
            except Exception:
                pass
    if not candidates:
        raise HTTPException(status_code=404, detail='no stored real-world evaluation found')
    latest = max(candidates, key=lambda x: x.stat().st_mtime_ns)
    try:
        data = json.loads(latest.read_text(encoding='utf-8'))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'evaluation artifact unreadable: {type(exc).__name__}')
    data['artifact'] = str(latest.relative_to(ROOT))
    return data


@app.get('/metrics', dependencies=[] if settings.metrics_public else [Depends(require_admin_key)])
def prometheus_metrics():
    if prom.REQUESTS is None:
        return Response(content=b'phishguard_metrics_unavailable 1\n', media_type='text/plain')
    return Response(content=prom.generate_latest(), media_type=prom.CONTENT_TYPE_LATEST)


@app.get('/v1/providers', dependencies=[Depends(require_admin_key)])
def providers():
    return {
        'safe_browsing': bool(settings.safe_browsing_api_key),
        'web_risk': bool(settings.web_risk_api_key),
        'virustotal': bool(settings.vt_api_key),
        'urlscan': bool(settings.urlscan_api_key),
        'phishtank': bool(settings.phishtank_app_key),
    }


@app.post('/v1/predict', dependencies=[Depends(require_inference_key)])
async def predict(req: URLRequest, x_admin_key: str | None = Header(default=None, alias='X-Admin-Key')):
    model, model_version, *_ = _current_state()
    if model is None:
        raise HTTPException(status_code=503, detail='model unavailable')
    started = time.perf_counter()
    probs, cache_hit, _, sources, stages = await _run_inference(predict_urls, [req.url], model)
    p = float(probs[0]); source = sources[0]; stage = stages[0]

    if stage in {'verified_feedback', 'local_intel'}:
        # These scores are exact label/feed outcomes, not calibrated model
        # probabilities. Keep their provenance explicit to avoid reporting a
        # feed match as an ML probability.
        verdict = 'phishing' if p >= 0.5 else 'benign'
        is_feedback = stage == 'verified_feedback'
        result = {
            'request_id': str(uuid.uuid4()), 'url': req.url, 'verdict': verdict,
            'binary_verdict': verdict, 'decision': 'verified_exact_url_label' if is_feedback else 'exact_local_threat_feed_match',
            'risk_band': ('critical' if verdict == 'phishing' else 'low'),
            'risk_score': round(p, 6),
            'score_type': 'verified_label' if is_feedback else 'local_threat_intelligence',
            'model_version': model_version, 'cache_hit': bool(cache_hit),
            'source': source, 'inference_stage': stage,
        }
    elif stage == 'fast':
        verdict, decision = classify_fast_score(p, model)
        result = {
            'request_id': str(uuid.uuid4()), 'url': req.url, 'verdict': verdict,
            'binary_verdict': verdict if verdict in {'phishing','benign'} else 'unknown',
            'decision': decision, 'risk_band': 'high' if verdict == 'phishing' else ('medium' if verdict == 'suspicious' else 'low'),
            'fast_score': round(p, 6), 'risk_score': round(p, 6), 'score_type': 'calibrated_fast_stage',
            'model_threshold': model.threshold, 'model_version': model_version,
            'cache_hit': bool(cache_hit), 'source': source, 'inference_stage': stage,
        }
    else:
        verdict, decision = classify_score(p, model)
        result = {
            'request_id': str(uuid.uuid4()), 'url': req.url, 'verdict': verdict,
            'binary_verdict': verdict if verdict in {'phishing', 'benign'} else 'unknown',
            'decision': decision, 'risk_band': risk_band(p, model),
            'risk_score': round(p, 6), 'score_type': 'calibrated',
            'model_threshold': model.threshold, 'model_version': model_version,
            'cache_hit': bool(cache_hit), 'source': source, 'inference_stage': stage,
        }

    trusted_homepage = apply_official_homepage_policy(result, req.url, stage)
    if stage not in {'verified_feedback', 'local_intel'} and not settings.ml_blocking_enabled and not trusted_homepage:
        # Model scores can be useful for triage while still being unsafe as an
        # automatic block decision. Keep the score and expose the abstention.
        result.update({
            'verdict': 'suspicious', 'binary_verdict': 'unknown',
            'decision': 'ml_advisory_only_realworld_gate_required',
            'risk_band': 'medium', 'blocking_eligible': False,
        })
    elif stage not in {'verified_feedback', 'local_intel'}:
        result['blocking_eligible'] = True

    # Shadow models are sampled, never executed for every request, and their
    # details are kept out of the public response.
    _, _, _, shadow_model, shadow_version = _current_state()
    if shadow_model is not None and random.random() < settings.shadow_sample_rate and stage in {'model', 'fast'}:
        try:
            sp = float((await _run_inference(shadow_model.predict_proba, [req.url]))[0])
            if prom.SHADOW_SAMPLES: prom.SHADOW_SAMPLES.inc()
            if (sp >= shadow_model.threshold) != (p >= model.threshold):
                if prom.SHADOW_DISAGREEMENTS: prom.SHADOW_DISAGREEMENTS.inc()
        except Exception:
            if prom.SHADOW_ERRORS: prom.SHADOW_ERRORS.inc()

    unicode_diag = unicode_host_diagnostics(req.url)
    if unicode_diag.get('unicode_present') or unicode_diag.get('punycode_present') or unicode_diag.get('bidi_controls'):
        result['url_diagnostics'] = {'unicode_host': unicode_diag}

    # Deep analysis runs only for uncertain single-URL results, when explicitly
    # enabled by deployment policy. It is local DNS/TLS evidence only: failures
    # or missing evidence never convert a result to benign. Remote provider calls
    # remain separately authenticated/admin-gated below.
    if settings.deep_analysis_on_uncertain and result.get('verdict') == 'suspicious':
        try:
            deep = await _bounded_local_enrichment(req.url)
            result['deep_analysis'] = {'status': 'completed', 'mode': 'local_dns_tls', 'evidence': deep}
        except asyncio.TimeoutError:
            result['deep_analysis'] = {'status': 'timeout', 'mode': 'local_dns_tls'}
        except Exception as exc:
            result['deep_analysis'] = {'status': 'unavailable', 'mode': 'local_dns_tls', 'error_type': type(exc).__name__}

    if req.enrich:
        if settings.remote_enrichment_admin_only:
            _check_key(x_admin_key, settings.admin_api_key, 'Enrichment API')
        if not settings.enrichment_enabled:
            raise HTTPException(status_code=403, detail='enrichment is disabled')
        enrichment = await _intel.enrich_url(req.url, include_remote=True)
        result['enrichment'] = enrichment
        if confirmed_intel(enrichment):
            result.update({'verdict': 'phishing', 'binary_verdict': 'phishing', 'decision': 'intel-confirmed', 'risk_band': 'critical', 'risk_score': 1.0, 'score_type': 'threat_intelligence'})

    result['latency_ms'] = round((time.perf_counter() - started) * 1000, 3)
    if prom.PREDICTIONS:
        prom.PREDICTIONS.labels(result['binary_verdict']).inc()
    return result


@app.post('/v1/predict/batch', dependencies=[Depends(require_inference_key)])
async def batch(req: BatchRequest):
    model, model_version, *_ = _current_state()
    if model is None:
        raise HTTPException(status_code=503, detail='model unavailable')
    urls = [x if isinstance(x, str) else x.url for x in req.urls]
    if sum(len(u) for u in urls) > settings.max_batch_chars:
        raise HTTPException(status_code=413, detail='batch is too large')
    started = time.perf_counter()
    probs, hits, _, sources, stages = await _run_inference(predict_urls, urls, model)
    results = []
    for u, p, src, stage in zip(urls, probs, sources, stages):
        p = float(p)
        if stage in {'verified_feedback', 'local_intel'}:
            verdict = 'phishing' if p >= 0.5 else 'benign'
            is_feedback = stage == 'verified_feedback'
            item = {
                'url': u, 'verdict': verdict, 'binary_verdict': verdict,
                'decision': 'verified_exact_url_label' if is_feedback else 'exact_local_threat_feed_match',
                'risk_band': 'critical' if verdict == 'phishing' else 'low',
                'risk_score': round(p, 6),
                'score_type': 'verified_label' if is_feedback else 'local_threat_intelligence',
                'source': src, 'inference_stage': stage,
            }
        elif stage == 'fast':
            verdict, decision = classify_fast_score(p, model)
            item = {'url': u, 'verdict': verdict, 'binary_verdict': verdict if verdict in {'phishing','benign'} else 'unknown', 'decision': decision, 'risk_band': 'high' if verdict == 'phishing' else ('medium' if verdict == 'suspicious' else 'low'), 'fast_score': round(p,6), 'risk_score': round(p,6), 'score_type': 'calibrated_fast_stage', 'source': src, 'inference_stage': stage}
        else:
            verdict, decision = classify_score(p, model)
            item = {'url': u, 'verdict': verdict, 'binary_verdict': verdict if verdict in {'phishing','benign'} else 'unknown', 'decision': decision, 'risk_band': risk_band(p, model), 'risk_score': round(p,6), 'score_type': 'calibrated', 'source': src, 'inference_stage': stage}
        trusted_homepage = apply_official_homepage_policy(item, u, stage)
        if stage not in {'verified_feedback', 'local_intel'} and not settings.ml_blocking_enabled and not trusted_homepage:
            item.update({'verdict': 'suspicious', 'binary_verdict': 'unknown', 'decision': 'ml_advisory_only_realworld_gate_required', 'risk_band': 'medium', 'blocking_eligible': False})
        elif stage not in {'verified_feedback', 'local_intel'}:
            item['blocking_eligible'] = True
        results.append(item)
        if prom.PREDICTIONS: prom.PREDICTIONS.labels(item['binary_verdict']).inc()
    return {'model_version': model_version, 'count': len(urls), 'cache_hits': hits, 'latency_ms': round((time.perf_counter()-started)*1000,3), 'results': results}


@app.post('/v1/enrich', dependencies=[Depends(require_admin_key)])
async def enrich(req: EnrichRequest):
    if not settings.enrichment_enabled:
        raise HTTPException(status_code=403, detail='enrichment is disabled')
    return {'url': req.url, 'enrichment': await _intel.enrich_url(req.url, include_remote=True)}


@app.post('/v1/predict/email', dependencies=[Depends(require_inference_key)])
async def predict_email(req: EmailRequest):
    model, model_version, email_model, *_ = _current_state()
    if model is None:
        raise HTTPException(status_code=503, detail='model unavailable')
    payload = req.model_dump()
    total_hdr = sum(len(str(k)) + len(str(v)) for k, v in payload['headers'].items())
    if total_hdr > 50000:
        raise HTTPException(status_code=413, detail='email headers too large')
    urls = [
        candidate for candidate in (
            raw.rstrip('.,;:!?)]}')
            for raw in re.findall(r'''https?://[^\s<>"']+''', payload['body'] + ' ' + payload['subject'], re.I)
        )
        if valid_http_url(candidate)
    ][:settings.max_email_urls]
    url_probs = []
    if urls:
        url_probs = (await _run_inference(predict_urls, urls, model))[0]
    rules = extract_email_features(**payload)
    if email_model is not None:
        ep = float((await _run_inference(email_model_predict, email_model, [payload]))[0])
        score = max(ep, max(url_probs, default=0.0)); source = 'email_model+url_model'; calibrated = True
    else:
        # Do not manufacture an email probability or a benign/phishing verdict when
        # an email model has not been trained. URL findings are still returned for
        # actionable enrichment, but the email verdict is explicitly unknown.
        return {
            'verdict': 'unknown',
            'binary_verdict': 'unknown',
            'decision': 'email_model_not_configured',
            'risk_score': None,
            'email_ml_score': None,
            'email_threshold': settings.email_phishing_threshold,
            'email_calibrated': False,
            'url_scores': [round(float(x), 6) for x in url_probs],
            'source': 'email_features+url_model',
            'blocking_eligible': False,
            'features': rules,
            'model_version': model_version,
        }
    ev, edecision = classify_email_score(float(score))
    eb = 'phishing' if score >= settings.email_phishing_threshold else 'benign'
    if not settings.ml_blocking_enabled:
        ev, eb, edecision = 'suspicious', 'unknown', 'ml_advisory_only_realworld_gate_required'
    return {'verdict': ev, 'binary_verdict': eb, 'decision': edecision, 'risk_score': round(float(score),6), 'email_ml_score': ep, 'email_threshold': settings.email_phishing_threshold, 'email_calibrated': calibrated, 'blocking_eligible': settings.ml_blocking_enabled, 'url_scores': [round(float(x),6) for x in url_probs], 'source': source, 'features': rules, 'model_version': model_version}


@app.post('/v1/feedback', status_code=202, dependencies=[Depends(require_admin_key)])
def feedback(req: FeedbackRequest):
    model, model_version, *_ = _current_state()
    add_feedback(req.url, req.label, req.source, req.verified, model_version, None, req.metadata)
    if req.verified:
        reload_verified_feedback()
    return {'accepted': True, 'verified_for_training': req.verified, 'queued_for_training': bool(req.verified), 'note': 'Only admin-authenticated verified labels enter retraining candidates.'}


@app.post('/v1/model/reload', dependencies=[Depends(require_admin_key)])
def reload_model():
    new_version = load_current()
    intel_result = reload_local_intel()
    feedback_count = reload_verified_feedback()
    return {'reloaded': True, 'model_version': new_version, 'local_intel': intel_result, 'verified_feedback_entries': feedback_count}
