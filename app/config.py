from __future__ import annotations
import os
from dataclasses import dataclass

def _bool(name: str, default: bool=False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {'1','true','yes','on'}

@dataclass(frozen=True)
class Settings:
    environment: str=os.getenv('ENVIRONMENT','development').strip().lower()
    redis_url: str|None=os.getenv('REDIS_URL') or None
    cache_ttl_seconds: int=int(os.getenv('CACHE_TTL_SECONDS','300'))
    cache_confident_ttl_seconds: int=int(os.getenv('CACHE_CONFIDENT_TTL_SECONDS','900'))
    cache_uncertain_ttl_seconds: int=int(os.getenv('CACHE_UNCERTAIN_TTL_SECONDS','60'))
    cache_maxsize: int=int(os.getenv('CACHE_MAXSIZE','100000'))
    enrichment_enabled: bool=_bool('ENRICHMENT_ENABLED',False)
    web_risk_api_key: str|None=os.getenv('GOOGLE_WEB_RISK_API_KEY') or None
    safe_browsing_api_key: str|None=os.getenv('GOOGLE_SAFE_BROWSING_API_KEY') or None
    vt_api_key: str|None=os.getenv('VIRUSTOTAL_API_KEY') or None
    urlscan_api_key: str|None=os.getenv('URLSCAN_API_KEY') or None
    phishtank_app_key: str|None=os.getenv('PHISHTANK_APP_KEY') or None
    max_batch_size: int=int(os.getenv('MAX_BATCH_SIZE','10000'))
    max_batch_chars: int=int(os.getenv('MAX_BATCH_CHARS','1000000'))
    inference_chunk_size: int=int(os.getenv('INFERENCE_CHUNK_SIZE','2048'))
    shadow_model_version: str|None=os.getenv('SHADOW_MODEL_VERSION') or None
    api_timeout_seconds: float=float(os.getenv('API_TIMEOUT_SECONDS','3.0'))
    api_key: str|None=os.getenv('PHISHGUARD_API_KEY') or None
    admin_api_key: str|None=os.getenv('PHISHGUARD_ADMIN_KEY') or None
    auth_required: bool=_bool('AUTH_REQUIRED',True)
    # ML-only classifications remain advisory until a release has passed an
    # independently collected real-world legitimate-domain challenge set.
    ml_blocking_enabled: bool=_bool('ML_BLOCKING_ENABLED',False)
    max_email_body_chars: int=int(os.getenv('MAX_EMAIL_BODY_CHARS','200000'))
    max_request_body_bytes: int=int(os.getenv('MAX_REQUEST_BODY_BYTES','2000000'))
    email_phishing_threshold: float=float(os.getenv('EMAIL_PHISHING_THRESHOLD','0.75'))
    email_suspicious_threshold: float=float(os.getenv('EMAIL_SUSPICIOUS_THRESHOLD','0.35'))
    require_model_manifest: bool=_bool('REQUIRE_MODEL_MANIFEST',True)
    app_version: str=os.getenv('PHISHGUARD_APP_VERSION','2.2.0')
    inference_threads: int=int(os.getenv('INFERENCE_THREADS','1'))
    inference_executor_workers: int=int(os.getenv('INFERENCE_EXECUTOR_WORKERS','2'))
    batch_response_limit: int=int(os.getenv('BATCH_RESPONSE_LIMIT','10000'))
    local_intel_reload_seconds: int=int(os.getenv('LOCAL_INTEL_RELOAD_SECONDS','15'))
    local_intel_max_age_seconds: int=int(os.getenv('LOCAL_INTEL_MAX_AGE_SECONDS','86400'))
    max_email_urls: int=int(os.getenv('MAX_EMAIL_URLS','100'))
    remote_enrichment_admin_only: bool=_bool('REMOTE_ENRICHMENT_ADMIN_ONLY',True)
    metrics_public: bool=_bool('METRICS_PUBLIC',False)
    cache_enabled: bool=_bool('CACHE_ENABLED',True)
    local_intel_enabled: bool=_bool('LOCAL_INTEL_ENABLED',True)
    cascade_enabled: bool=_bool('CASCADE_ENABLED',True)
    cascade_low_threshold: float=float(os.getenv('CASCADE_LOW_THRESHOLD','0.10'))
    cascade_high_threshold: float=float(os.getenv('CASCADE_HIGH_THRESHOLD','0.90'))
    deep_analysis_on_uncertain: bool=_bool('DEEP_ANALYSIS_ON_UNCERTAIN',False)
    deep_analysis_timeout_seconds: float=float(os.getenv('DEEP_ANALYSIS_TIMEOUT_SECONDS','2.0'))
    deep_analysis_max_concurrency: int=int(os.getenv('DEEP_ANALYSIS_MAX_CONCURRENCY','32'))
    shadow_sample_rate: float=float(os.getenv('SHADOW_SAMPLE_RATE','0.0'))
    feed_refresher_enabled: bool=_bool('FEED_REFRESHER_ENABLED',False)
    feed_refresh_seconds: int=int(os.getenv('FEED_REFRESH_SECONDS','600'))
    openphish_feed_url: str=os.getenv('OPENPHISH_FEED_URL','https://www.openphish.com/feed.txt')
    feed_dir: str=os.getenv('PHISHGUARD_FEED_DIR','')
    urlhaus_export_url_template: str=os.getenv('URLHAUS_EXPORT_URL_TEMPLATE','https://urlhaus-api.abuse.ch/v2/files/exports/{key}/recent.csv')
    def validate_startup(self) -> None:
        if self.environment not in {'development','staging','production'}:
            raise RuntimeError(f'Unsupported ENVIRONMENT={self.environment!r}')
        if self.max_batch_size < 1 or self.inference_chunk_size < 1:
            raise RuntimeError('batch and inference chunk sizes must be positive')
        if self.cache_ttl_seconds < 1 or self.cache_maxsize < 1:
            raise RuntimeError('cache TTL and maxsize must be positive')
        if not (0.0 <= self.cascade_low_threshold < self.cascade_high_threshold <= 1.0):
            raise RuntimeError('cascade thresholds must satisfy 0 <= low < high <= 1')
        if self.deep_analysis_timeout_seconds <= 0 or self.deep_analysis_max_concurrency < 1:
            raise RuntimeError('deep analysis timeout/concurrency settings must be positive')
        if self.environment == 'production':
            if not self.auth_required:
                raise RuntimeError('AUTH_REQUIRED must be true in production')
            if not self.api_key or not self.admin_api_key:
                raise RuntimeError('PHISHGUARD_API_KEY and PHISHGUARD_ADMIN_KEY are required in production')
            if not self.require_model_manifest:
                raise RuntimeError('REQUIRE_MODEL_MANIFEST must be true in production')
settings=Settings()
