from __future__ import annotations
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    REQUESTS = Counter('phishguard_requests_total','Inference requests',['endpoint','status'])
    PREDICTIONS = Counter('phishguard_predictions_total','URL predictions',['verdict'])
    LATENCY = Histogram('phishguard_inference_latency_seconds','Inference latency',['endpoint'], buckets=(.001,.002,.005,.01,.02,.05,.1,.25,.5,1,2,5,10))
    CACHE_HITS = Counter('phishguard_cache_hits_total','URL cache hits')
    CACHE_MISSES = Counter('phishguard_cache_misses_total','URL cache misses')
    SHADOW_SAMPLES = Counter('phishguard_shadow_samples_total','Shadow predictions sampled')
    SHADOW_DISAGREEMENTS = Counter('phishguard_shadow_disagreements_total','Shadow/current verdict disagreements')
    SHADOW_ERRORS = Counter('phishguard_shadow_errors_total','Shadow inference errors')
except Exception:
    REQUESTS=PREDICTIONS=LATENCY=CACHE_HITS=CACHE_MISSES=SHADOW_SAMPLES=SHADOW_DISAGREEMENTS=SHADOW_ERRORS=None
    generate_latest=lambda: b''
    CONTENT_TYPE_LATEST='text/plain'
