# PhishGuard Production — audited realtime release

Production-oriented phishing URL/email detection pipeline evolved from the original `train_new_model.py`.

The original Django API and React client remain under `backend/` and `frontend/`.
This root-level service adds the audited FastAPI runtime, URL/email model
training and evaluation, feed refresh, and release gates. Both code paths are
retained; this guide documents the current production-oriented service.

## Current release

```text
Production model: v20261001_104517
Shadow model:     disabled by default
Serving release:  clean-realtime-v1
```

The November 25 investigation trained two additional challengers and refreshed
OpenPhish, but neither challenger passed the release checks. The active model's
ordinary-site false positives remain unresolved. Measurements and their limits
are in [REAL_WORLD_RESULTS_2026-11-25.md](REAL_WORLD_RESULTS_2026-11-25.md).

### Blocking policy

`ML_BLOCKING_ENABLED` defaults to `false`. Model-only URL and email predictions
are returned as `suspicious` with `binary_verdict: unknown` while retaining the
model score for analyst review. Exact verified feedback and fresh exact-match
threat-feed hits remain actionable. Do not enable automated blocking until a
candidate passes an independently sourced legitimate-domain challenge set and
the release gates. The bundled evaluation scores are dataset-specific and do
not establish real-world accuracy.

The URL model's probability calibration reflects its training distribution;
it is not validated for arbitrary current websites. Responses expose this as
`model_calibration_scope: training_distribution_only`. Email remains advisory
even if URL blocking is later approved, because the email model has a separate
evaluation and has not passed an email-specific blocking gate.

The promotion command also checks `data/realworld/legitimate_challenge.csv`.
It contains 21 unique organization homepages, including the six named in the
October 12 report. The other nine URLs from that original 15-site test were
not supplied, so this does not recreate the original test. Actual attempt times
and fetch verification status are recorded; six HTTPS requests failed TLS
verification. These capture times are not source publication times. Never add
this challenge data to model training.

An externally sourced candidate (`v20261117_182148`) corrected the previous
model's inverted predictions on a 20,950-row domain-disjoint holdout, but it
still fails frozen regression and raw homepage challenge gates. It was not
promoted. See [REAL_WORLD_RESULTS_2026-11-18.md](REAL_WORLD_RESULTS_2026-11-18.md)
for the measured comparison, data provenance, and remaining release blockers.

The URL timestamp-data gate now passes on PhishVN v5, a source-attested,
CC BY 4.0 dataset. Its official group-aware URL folds are filtered to gold and
silver labels, and `collected_at` (not retrieval-only `scraped_at`) supplies
the event date. At a strict 2023-01-01 UTC cutoff, the split has 3,183
pre-cutoff training URLs and 845 post-cutoff holdout URLs, with both labels,
full timestamp coverage, and zero exact-URL or registrable-domain overlap.
Reacquire the attributed source with `scripts/download_phishvn_temporal.py`,
rebuild with `scripts/prepare_phishvn_temporal.py`, and check with
`scripts/temporal_gate.py`.

A challenger (`v20261118_103146`) trained only on the 3,183 pre-cutoff rows achieved 99.76%
accuracy, 100% recall, and 0.56% false-positive rate on the 843 valid-HTTP
post-cutoff holdout rows, and flagged 0/21 legitimate homepage roots. It still
fails the frozen regression set badly (26.29% accuracy, 95.33% FPR), so it is
not promoted. `models/CURRENT` remains unchanged and model-only blocking stays
disabled. The temporal data gate now passes; the active production model's
temporal qualification and release gates remain unresolved. DeepURLBench is
retained as a separate CC BY-NC research-only cross-check and is never used to
train or ship the production model. See the dated report for the evaluation
details and limits.

That temporal holdout is heavily confounded by URL scheme: an HTTP-only rule
matches all 843 of that challenger's decisions. It does not provide adequate
evidence of general phishing detection. The release gate requires both labels
to have HTTP and HTTPS examples, and the present holdout fails this check.

Cross-source challengers trained with broader public data improved on the
LegitPhish v2 holdout and homepage corpus, but exposed a substantial conflict
with the frozen regression population. Candidate `v20261118_143528` reached
99.79% accuracy on the 20,950-row LegitPhish holdout and 0/21 homepage false
positives, while producing 100% FPR on the 107 frozen benign rows. Candidate
`v20261118_144004` had 0 frozen false positives but only 92.45% frozen recall
and flagged 17/21 homepage roots. Neither is releasable. `models/CURRENT` was
not changed; full cross-source results are in the dated report.
The reproducible comparison is in
`data/realworld/evaluations/crosssource_candidate_metrics.json`, generated by
`scripts/evaluate_crosssource_candidates.py`.

The latest mixed-source candidate (`v20261119_064457`) reduces the raw
homepage false-positive count to 2/21, with 99.84% accuracy on the independent
20,950-row LegitPhish holdout and 51/51 detections on the captured OpenPhish
sample. It still has only 88.68% recall on the frozen regression set, so it
remains unpromoted. It includes undated and 2026 source rows and is not a
strictly pre-cutoff model. Metrics are in
`data/realworld/evaluations/candidate_v20261119_064457_metrics.json`.

Candidate `v20261119_083010` uses the full eligible balanced LegitPhish v2
training split (58,754 rows) plus the original production corpus, rather than
the earlier 20,000-row sample. It flags 1/21 homepage roots, reaches 91.51%
frozen-regression recall (9 phishing misses, 1 benign false positive), and
records 99.78% accuracy on the separate 20,950-row LegitPhish domain holdout.
The remaining homepage false positive is the European Union root. This
candidate still fails the frozen recall release gate (98%), is not strictly
temporally qualified, and is not promoted. Its full measurements are in
`data/realworld/evaluations/candidate_v20261119_083010_metrics.json`.

If all promotion gates pass, the command writes
`models/<version>/BLOCKING_APPROVED.json`, bound to that exact model SHA-256.
Setting `ML_BLOCKING_ENABLED=true` makes API startup verify the approval and
challenge coverage. A hand-edited environment flag cannot enable blocking for
an unapproved model. Build the serving image after promotion so the approval
record is included with the model artifact.

## Phase 1–17 upgrades

### Accuracy/precision/recall

- Seven heterogeneous URL learners instead of one XGBoost model.
- New URL `char_wb` TF-IDF learner adds an independent character-boundary view.
- Bounded threshold selection prevents pathological near-1.0 thresholds on small calibration folds.
- Separate precision and recall constraints.
- OOF-derived hard-negative weighting, including additional pressure on difficult benign URLs.
- Immutable regression gate before model promotion.

### URL intelligence

- 104 validated serving-compatible structured features in the promoted model.
- Unicode/homograph features are part of the promoted 104-feature schema.
- Punycode, Unicode-script and confusable-character signals.
- Brand-position signals separating registered-domain, subdomain and path occurrences.
- Suspicious-port, host/path similarity and host-randomness features.

### Runtime

- FastAPI single and batch inference.
- Three-zone verdict: `benign`, `suspicious`, `phishing`.
- Compatibility `binary_verdict`.
- Local exact phishing-feed lookup.
- Reserved/special-use domain policy.
- Verified feedback lookup.
- Optional shadow-model disagreement output (disabled by default).
- In-process and optional Redis caching.
- Prometheus metrics.
- Optional DNS/TLS and external threat-intelligence enrichment.

## Current model measurements

### Unseen-domain outer holdout (207 rows across 95 held-out domains)

```text
accuracy:  100.00%
precision: 100.00%
recall:    100.00%
f1:        100.00%
roc_auc:   100.00%
pr_auc:    100.00%
brier:       0.06%
fp:              0
fn:              0
```

### Immutable v2 regression

```text
accuracy:   98.59%
precision: 99.05%
recall:     98.11%
f1:         98.58%
fp:              1
fn:              2
```

The outer holdout contains 207 test rows (130 phishing, 77 legitimate) across 95 held-out domains. A perfect score on this small holdout is not evidence of perfect performance in the wild. These are dataset-specific results, not live-Internet guarantees.

## Train

The promoted model is trained only from the two labeled URL datasets. Manual benign seeds and synthetic examples are rejected by the training pipeline.

```bash
python -m app.train \
  --phishing data/source/extracted_phishing_dataset.csv \
  --legitimate data/source/extracted_legitmate_dataset.csv \
  --folds 3 \
  --trees 70 \
  --lgb-trees 70 \
  --extra-trees 70 \
  --char-features 5000 \
  --word-features 2500 \
  --host-features 3000 \
  --svc-features 3500 \
  --charwb-features 4000 \
  --min-precision 0.98 \
  --min-recall 0.95 \
  --threshold-min 0.12 \
  --threshold-max 0.45 \
  --exclude-csv data/realworld/archive/openphish_20261005_unseen.csv \
  --exclude-domains-csv data/release_gate_v1.csv \
  --exclude-frozen
```

For options:

```bash
python -m app.train --help
```

For `--require-temporal`, provide `--temporal-source-metadata` as JSON with
`timestamp_provenance` (`source_attested` or `source_provided`),
`timestamp_semantics`, `timestamp_column`, and `source_files` entries
containing each input's `path` and SHA-256. Every retained row must have the
specified event time; later rows sharing a training domain are excluded from
the temporal holdout. The metadata must describe source event times, not the
date the file was downloaded. This training split alone is not release
approval; `scripts/promote_if_gates.py` checks separate model-bound evidence.

## Promotion gates

```bash
python -m scripts.promote_if_gates vYYYYMMDD_HHMMSS
```

Default gates:

```text
outer accuracy   >= 98%
outer precision  >= 98%
outer recall     >= 97%
frozen accuracy  >= 98%
frozen precision >= 99%
frozen recall    >= 98%
temporal precision >= 98%, recall >= 97%, false-positive rate <= 2%
```

Promotion also requires a candidate-specific, hash-verified training-domain
record, a disjoint homepage challenge, a source-attested chronological test
bound to the model's recorded training inputs, and non-regression against the
active model. Training never promotes a model directly. Domain grouping uses a
deterministic bundled public-suffix snapshot, including private hosting suffixes.
Previous candidates without these evidence files cannot pass the new gate.

Rollback:

```bash
python -m scripts.model_registry rollback
```

Rollback requires the target model to have a valid current release approval.

## Run the API

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --workers 4
```

Endpoints:

```text
GET  /                         (same-origin PhishGuard Console; entered key is not persisted)
GET  /health
GET  /ready
GET  /metrics
GET  /v1/providers
GET  /v1/evaluation/latest (admin)
POST /v1/predict
POST /v1/predict/batch
POST /v1/predict/email
POST /v1/enrich
POST /v1/feedback
POST /v1/model/reload
```

## Continuous learning

Submit verified labels to `/v1/feedback`. The feedback queue is deliberately separated from the live model.

Retrain only after sufficient verified data:

```bash
python -m app.retrain \
  --phishing /path/to/phishing.csv \
  --legitimate /path/to/legitimate.csv \
  --min-feedback 25
```

## Email detection

The email endpoint analyzes sender/reply-to/return-path consistency, authentication headers, urgency/financial language and embedded URLs. It can load an email ML artifact when a labeled email dataset is provided.

Expected labeled columns:

```text
subject,body,sender,reply_to,return_path,label
```

The bundled `v20261001_104517` API package now has an optional email classifier trained from the historically timestamped MeAJOR public corpus. Its chronological holdout is source-shifted and has a 17.07% false-positive rate at threshold 0.5 (12.25% at the API's default threshold 0.75); treat its score as advisory only. `ML_BLOCKING_ENABLED` remains false. Rebuild inputs with `scripts/download_meajor_email.py` and `scripts/prepare_meajor_email.py`; provenance, source hashes, labels, date semantics, and split metadata are stored beside the files.

## Real-world feeds and enrichment

The project includes ingestion adapters for OpenPhish, URLhaus and PhishTank plus optional Web Risk, VirusTotal, urlscan.io and DNS/TLS enrichment.

The API reads the last good OpenPhish and optional URLhaus snapshots from
`data/feeds/live/`. The separate refresher polls OpenPhish every 12 hours by
default, retries failures, preserves the last good snapshot, and archives
timestamped 90-day snapshots without visiting listed URLs. `/health` reports
provider-level state as `fresh`, `partial`, or `stale_or_unavailable`; it does
not call a stale or missing provider fresh. URLhaus requires `URLHAUS_AUTH_KEY`.
The local environment fetched 300 OpenPhish entries; URLhaus remained skipped
without credentials. See the dated evaluation report for exact sample metrics.

OpenPhish describes its intelligence as focused on live phishing and documents an academic-use program for eligible research users. citeturn925626search1turn925626search3

Cloudflare Radar publishes domain rankings but warns that its popularity lists can include domains with emerging malicious behavior; popularity should therefore not be treated as a benign label. citeturn954824search12

## Scale target

Your original target of 1 crore requests in 10–15 seconds corresponds to approximately 667K–1M requests/second.

That requires a distributed serving architecture rather than a single Python process:

```text
Load Balancer
     |
API replicas
     |
local cache / Redis
     |
batched inference workers
     |
model registry + telemetry
```

Use the bundled benchmark only for relative local regression testing:

```bash
python benchmark.py --input data/realworld_load_corpus.json --batch 1000
```

## References

- OpenPhish: https://www.openphish.com/
- OpenPhish knowledge base: https://www.openphish.com/kb.html
- OpenPhish academic use: https://www.openphish.com/academic_use.html
- Cloudflare Radar Domain Rankings: https://radar.cloudflare.com/domains
- IANA special-use domains: https://www.iana.org/assignments/special-use-domain-names/special-use-domain-names.xhtml

## Security/data-integrity audit — 2026-10-01

The v7 package was audited before this archive was produced. Findings fixed in this release include:

- unauthenticated admin/feedback endpoints
- exact-URL feedback scope errors that could otherwise relabel sibling URLs
- DNS-rebinding/SSRF exposure in TLS enrichment
- legacy SQLite schema incompatibility and feedback migration failures
- incompatible rollback candidates
- non-atomic production model pointer writes
- unsynchronized in-process cache mutation
- missing request/cache Prometheus counters
- inconsistent batch application of verified feedback
- oversized batch/email/request inputs
- incomplete continuous-retraining argument propagation
- stale local-intelligence state after feed refresh
- synthetic benchmark traffic being presented as throughput evidence

### Real-world result now included

The 2026-10-05 captured OpenPhish evaluation contains **51 unseen-domain phishing URLs** after leakage exclusion. The exact promoted model detected **51/51 (100% positive-feed recall)**. The capture was replayed without visiting any listed URL. This is a phishing-positive-only measurement and does not estimate precision or overall Internet-wide accuracy.

The deployment refreshes the external feed independently. The bundled October 5 capture is historical; it is not described as live on October 12. A current feed download into this build runtime was not verified because its container environment could not resolve the feed host.

### Refresh the real-world feed

```bash
python scripts/refresh_feeds.py --once --feed-dir data/feeds/live
```

The script fetches feed text only, never visits the listed URLs, writes atomically, and records source hash/provenance. After refresh, call the protected model-reload endpoint or restart the service so the in-memory local-intelligence index is refreshed.

### Dependency audit status

`pip-audit` is not installed in the build runtime, and the runtime has no package-index/network access, so this release does **not** claim a zero-CVE dependency status. `pip check` reports one unrelated environment conflict (`moviepy` requires `pillow<12`, while the shared runtime has Pillow 12.3.0). The PhishGuard dependencies themselves import and the test suite passes. The installed scikit-learn is 1.8.0; the current scikit-learn security policy lists 1.9.1 as the supported version, so a production deployment should rebuild/retrain the model under the supported line before treating the environment as security-maintained. citeturn538493search4

### Benchmark policy

`benchmark.py` replays actual URLs from the supplied/retrieved corpus and labels the result as replay evidence. It is useful for regression and local capacity comparisons, but it is not a substitute for a distributed production load test. No generated or synthetic URL is used in the release corpus.

## Real-world validation and 1M decisions/s capacity gate

See `REAL_WORLD_AND_SCALE.md` and `REAL_WORLD_RESULTS_2026-10-01.md`.

### Live evaluation

```bash
python scripts/realworld_eval.py --limit-positive 1000 --limit-benign 1000
```

The evaluator records exact feed hashes and timestamps and never visits the listed URLs.

### Distributed load test

```bash
export PHISHGUARD_API_KEY='...'
export PHISHGUARD_ADMIN_KEY='...'
export TARGET_RPS=125000
export DURATION=60s
export MODE=inference
docker compose -f loadtest/docker-compose.loadtest.yml up --build --scale loadgen=8
```

This configuration targets 1,000,000 decisions/s across eight generators. The measured result must be taken from the actual k6 run; it is never hard-coded as a claim.

For 10 million decisions:
- 10 seconds → 1,000,000 decisions/s
- 15 seconds → 666,667 decisions/s

### Actual HTTP service benchmark (sandbox evidence, 2026-10-12)

The actual FastAPI endpoint was replay-tested with `scripts/http_benchmark.py` using the captured/supplied URL corpus in `data/realworld_load_corpus.json` (2,031 URL strings). The model version was checked in each response. Cache and local-intelligence shortcuts were disabled; the active model has no dedicated calibrated fast-stage artifact, so it used the full calibrated ensemble. The benchmark did not visit any submitted URL.

```text
Single-URL HTTP:
  successful:            200 / 200
  throughput:             73.52 requests/sec
  p95 latency:            412.23 ms
  p99 latency:            499.96 ms

64-URL batch HTTP:
  successful batches:     100 / 100
  decisions:              6,400
  throughput:             667.02 decisions/sec
  p95 batch latency:      2,065.64 ms
  p99 batch latency:      2,074.33 ms
```

These are actual measurements of this sandbox configuration, not a production-capacity claim. They demonstrate that the current full-ensemble HTTP path is far below the 666,667–1,000,000 decisions/sec target. The next performance work must use a larger real cluster and can only enable a lower-cost fast path after it passes independent precision/recall and frozen-regression gates.

### Dependency audit

```bash
python -m pip install -r requirements.txt -r requirements-audit.txt
python -m pip_audit -r requirements.txt
```

CI runs the same audit. The current serialized model is not silently upgraded across scikit-learn versions; use `requirements.secure-build.txt` for a clean retrain/rebuild and then pass the model gates.


## Current promoted model: v20261001_104517

This is the only model artifact in the final release tree. Its training provenance is exactly:

```text
data/source/extracted_phishing_dataset.csv
data/source/extracted_legitmate_dataset.csv
```

No manual benign seed, synthetic training fallback, or embedded benign test list is used in training or inference. The promoted model has **104 serving-compatible structured features** plus the text learners described above.

### Real-world evidence
Stored OpenPhish captures were replayed through the exact promoted model without visiting the listed URLs. After registrable-domain exclusion against the production training-domain list, **51/51 unseen-domain phishing feed entries were detected (100% positive-feed recall)** in the dated 2026-10-05 capture.

### Scale evidence
The package includes:

- a 2,031-URL replay corpus containing actual URLs from the supplied datasets and a captured OpenPhish sample;
- an HTTP benchmark against the actual FastAPI endpoint;
- a distributed k6 constant-arrival-rate test;
- a capacity planner for the 667K-1M requests/second target.

The 1M RPS value is a **target**, not a hardcoded measurement. The actual achieved value must come from a real distributed deployment run.


## Phase 1–7 real-world implementation (2026-10-05)

The original Phase 1–7 release passed 32 tests. The combined Phase 1–17 release was revalidated on 2026-10-12 with **59/59 automated tests passing** at the time of this package build. Dependency CVE status remains explicitly unverified because security scanners require an Internet-enabled CI environment.


Phases 1–7 are implemented as enforceable project behavior:

1. Production code has no synthetic/fallback model path. Missing or invalid models fail closed.
2. Development, staging, and production example configurations are separated under `config/`.
3. `scripts/ingest_realworld.py` retrieves real OpenPhish, optional URLhaus, and Tranco data. Listed URLs are never visited. Tranco is retained as a benign candidate only and never auto-labeled as benign.
4. `scripts/build_dataset_manifest.py` records SHA-256, size, row counts, normalized-URL counts, labels, source counts, and genuine per-row timestamp coverage where present. `data/PROVENANCE.md` defines source semantics.
5. `scripts/audit_dataset_leakage.py` checks exact URL and registrable-domain overlap between training and evaluation sets.
6. `scripts/temporal_gate.py` validates real timestamp coverage. `app.train --require-temporal` refuses to fabricate dates and fails when temporal evidence is insufficient.
7. The 104-feature URL extractor remains the serving schema; all model features are reproducible from the submitted URL and use no network-only values.

### Real-world refresh

```bash
python scripts/ingest_realworld.py
```

This command may fail in an offline environment. That is intentional: the pipeline refuses to invent a replacement dataset.

### Provenance manifest

```bash
python scripts/build_dataset_manifest.py \
  data/source/extracted_phishing_dataset.csv \
  data/source/extracted_legitmate_dataset.csv \
  data/training/production_training.csv \
  data/realworld/archive/openphish_20261005_unseen.csv \
  data/realworld_load_corpus.json
```

### Leakage audit

```bash
python scripts/audit_dataset_leakage.py \
  --train data/training/production_training.csv \
  --test data/realworld/archive/openphish_20261005_unseen.csv
```

For final model release, run the audit separately for the combined training corpus against each validation/test corpus.


## Data lifecycle

Historical threat-feed snapshots under `data/history/` and timestamped snapshots
under `data/feeds/live/snapshots/` are retained for reproducibility and are never
treated as current. The production local intelligence index reads the current
feed files and freshness metadata only.


## Phases 8–17: URL detection, risk policy, live intelligence and caching

### Phase 8 — Character-level URL models
The existing promoted model uses URL character TF-IDF, word TF-IDF, host-character TF-IDF, a LinearSVC character model and a character-boundary model alongside structured learners. These components are loaded with the model; they are not rebuilt per request.

### Phase 9 — Unicode/Punycode diagnostics
The 104-feature extractor already supplies Punycode, non-ASCII, mixed-script and confusable-character signals. `app/unicode_url.py` adds human-readable IDNA round-trip, mixed-script, bidi-control and invisible-format diagnostics. These diagnostics are evidence only, not automatic phishing verdicts.

### Phase 10 — Complementary ensemble
The seven existing component scores feed the trained meta-model. The currently promoted artifact remains `v20261001_104517`; this update does not claim that it was retrained on a larger corpus. The later Phase 8–17 challenger failed independent promotion gates and is not active.

### Phase 11 — Probability calibration
The ensemble's Platt/logistic calibration remains in the production bundle. New training runs additionally calibrate the fast XGBoost score on out-of-fold calibration data.

### Phase 12 — Threshold selection
Training chooses the main operating threshold from the calibration fold under explicit precision/recall constraints. It separately chooses a suspicious/abstention threshold targeting a configurable recall. Metrics are emitted for the calibration and outer holdout; a tiny calibration sample cannot prove universal accuracy.

### Phase 13 — Multi-zone verdicts
URL responses use `benign`, `suspicious`, or `phishing`. `binary_verdict` is `unknown` for suspicious results instead of silently calling them benign. Email ML scores remain advisory while the blocking release gate is disabled. `/health` distinguishes complete fresh feeds from partial provider coverage and stale/unavailable intelligence.

### Phase 14 — Validated fast path
The trainer now learns a separate fast-stage calibrator and shortcut bounds using calibration-fold data. A shortcut is allowed only if the outer unseen-domain test shows no more than the configured recall/precision degradation. **The currently promoted artifact does not include this calibrator, so its fast path remains disabled and requests use the full calibrated ensemble.** A newly trained candidate is never promoted automatically and must also pass the frozen-regression gate.

### Phase 15 — Local threat intelligence
The feed refresher and inference index share the same default directory
(`data/feeds/live`). OpenPhish and optionally URLhaus feed rows are ingested by
the refresher; listed URLs are never visited. Local lookup uses SHA-256 digests
of canonical exact URLs (path/query remain specific; no domain-wide verdict).
Freshness metadata is required before a feed contributes to decisions. Health
shows `partial` if any provider attempted in the last refresh is skipped or
failed, even when another feed is fresh.

Run a one-time feed refresh in an Internet-enabled environment:

```bash
python scripts/refresh_feeds.py --once --feed-dir data/feeds/live
```

For production, run the feed refresher as the separate container in `docker-compose.yml`, with outbound Internet access restricted to approved feed endpoints. A stale or missing feed is reported as stale; it is not represented as live.

### Phase 16 — Deep analysis only for uncertain URLs
Set `DEEP_ANALYSIS_ON_UNCERTAIN=true` to perform bounded, local DNS/TLS enrichment only when the single-URL result is `suspicious`. It is off by default to protect latency and cost. DNS/TLS work runs off the event loop, has a concurrency semaphore and timeout, and never turns missing evidence into `benign`. Third-party enrichment remains explicitly requested, admin-gated, and disabled unless configured.

### Phase 17 — L1/L2 cache
The cache now honors per-entry TTLs in the in-process bounded LRU and Redis. Model scores are namespaced by model version; high-margin scores can use a longer TTL, while near-threshold scores expire sooner. Provider responses and origin-level DNS/TLS evidence are cached. Cache failure is non-fatal and the detector falls back to the model, not a benign verdict.

### Phase 8–17 verification
Run:

```bash
python scripts/phase8_17_gate.py
pytest -q
python scripts/production_doctor.py
```

The gate records whether the active model actually has a validated fast path. No load capacity, live feed freshness, or dependency-CVE claim is inferred from unit tests; these still require the target environment and an Internet-connected audit run.
