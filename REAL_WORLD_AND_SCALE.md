# PhishGuard: real-world evaluation + crore-scale capacity gate

## 1. Real-world detection gate

`python scripts/realworld_eval.py` downloads current OpenPhish, URLhaus and Tranco data and evaluates the current promoted model without visiting any listed URL.

- OpenPhish: positive threat feed.
- URLhaus: positive malware-URL feed.
- Tranco: benign **proxy**, not ground-truth benign. It is a popularity ranking, so absence from threat feeds must not be treated as proof of safety.
- Exact source bytes, SHA-256, retrieval time and model version are stored with the result.

For strict precision/accuracy, set up a verified benign stream using an external reputation provider (for example Google Web Risk/VirusTotal where permitted) or organization-owned labeled traffic. The evaluator intentionally refuses to call a proxy score "Internet-wide accuracy".

## 2. Capacity gate

Use k6 with the constant-arrival-rate executor. Grafana documents this executor for request-rate testing and recommends distributed load generation for very high rates. A single k6 instance is generally in the 100K–300K RPS range depending on hardware and SUT latency, so 667K–1M RPS requires multiple generators.

Example for 1,000,000 URL decisions/s with 8 load generators:

```bash
export PHISHGUARD_API_KEY='...'
export PHISHGUARD_ADMIN_KEY='...'
export TARGET_RPS=125000
export DURATION=60s
export MODE=inference
docker compose -f loadtest/docker-compose.loadtest.yml up --build --scale loadgen=8
```

For 10,000,000 URL decisions in 10 seconds, the required sustained decision rate is 1,000,000/s. For 15 seconds it is 666,667/s.

Run two tests:

1. `MODE=inference`: unique URLs/queries; measures feature extraction + model inference.
2. `MODE=batch BATCH_SIZE=100`: measures the high-throughput batch path and is the recommended production ingestion pattern when callers can batch.

Do **not** count cache hits as proof of ML throughput. The supplied k6 script disables the cache and varies the URL query string for the capacity test.

## 3. Promotion criteria

A capacity result is valid only when the report contains:

- total HTTP requests
- achieved RPS
- p50/p95/p99 latency
- error rate
- CPU/RAM per API replica
- number of API replicas
- number of load generators
- batch size
- cache state
- model version
- exact hardware/cloud instance types

The result must be recorded as `measured_RPS_on_environment`, never as a universal RPS guarantee.

## 4. Dependency security

Run:

```bash
python -m pip install -r requirements.txt -r requirements-audit.txt
python -m pip_audit -r requirements.txt
```

CI runs the same audit. This makes vulnerability status reproducible in an Internet-enabled build environment instead of depending on whether the development sandbox has network/package-index access.

The current model artifact was built under the sandbox's scikit-learn 1.8.0. A dependency upgrade should therefore be accompanied by a clean rebuild/retrain and regression gate before the artifact is promoted.


## Phase 8–17 notes

The fast-path implementation is release-gated. A candidate fast stage is disabled unless its calibration-fold purity constraints and independent outer-test non-regression constraints pass. The current active v20261001_104517 artifact does not carry a dedicated fast calibrator, so inference uses the fully calibrated ensemble. This trades some raw throughput for not making a final verdict from an uncalibrated score.

The 2026-10-05 OpenPhish sample is stored under `data/realworld/archive/` because it is a historical capture. It remains useful for exact reproducible regression and is not described as fresh live data.


## Actual local HTTP measurement (2026-10-12)

Using `scripts/http_benchmark.py` against a local FastAPI process with cache/local-intelligence disabled and the full calibrated ensemble (active model `v20261001_104517`, fast path disabled):

- Single URL: 200/200 successful, 73.52 requests/s, p95 412.23 ms, p99 499.96 ms.
- 64-URL batch: 100/100 batch requests successful, 6,400 decisions, 667.02 decisions/s, p95 2,065.64 ms, p99 2,074.33 ms.

This is a real HTTP benchmark against captured/supplied URL strings but only on the sandbox runtime. It demonstrates current performance is not close to the 666,667–1,000,000 decisions/s target. The target remains unverified until a distributed cloud run sustains it under defined latency and error-rate objectives.
