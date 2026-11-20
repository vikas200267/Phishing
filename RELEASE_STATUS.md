# PhishGuard Phase 1–17 implementation status — 2026-10-12

## Production model

- Current: `v20261001_104517`
- 104 serving-compatible URL features plus heterogeneous URL learners.
- Training provenance: the two supplied labeled URL corpora only; no manual benign seed and no synthetic fallback.
- Model integrity manifest enforced at startup.

## Phase 1 — Real production path

Production inference fails closed if the model, schema, or integrity manifest is invalid. There is no generated fallback model.

## Phase 2 — Environment separation

Development, staging and production configuration examples are separate. Production requires API/admin keys and model-manifest verification.

## Phase 3 — Real datasets

- `scripts/ingest_realworld.py` fetches OpenPhish, optional URLhaus, and Tranco.
- The OpenPhish capture from 2026-10-05 is stored under `data/realworld/archive/`; it is historical, not current. Listed URLs were not visited.
- Tranco is a benign candidate population only; it is never automatically labeled as ground truth.
- Historical 2026-10-01 feed material is under `data/history/`.

## Phase 4 — Provenance

SHA-256, byte counts, row counts, normalized URL counts, source URLs and capture metadata are recorded.

## Phase 5 — Leakage prevention

The production training corpus passes exact-URL and registrable-domain exclusion against the 2026-10-05 unseen OpenPhish evaluation capture: 0 exact overlap and 0 domain overlap.

## Phase 6 — Temporal validation

The temporal gate is implemented and refuses to fabricate timestamps. The current training corpus does not yet have sufficient real timestamps for a full temporal precision/recall benchmark; the gate reports `not_ready`. This is an evidence gap, not a simulated result.

## Phase 7 — URL feature extraction

The production model uses 104 features reproducible from raw URL input. Train/serve parity is tested. No network-only feature is required by the ML path.

## Real-world evaluation evidence

The historical 2026-10-05 OpenPhish capture contains 51 unseen-domain phishing URLs after leakage exclusion. The promoted model detected 51/51 on that archived capture. This is a positive-feed recall measurement only; it is not current feed coverage, overall accuracy, or precision.

A real-data challenger (`v20261005_160802`) was trained but rejected by the release gate because its frozen-regression recall was below the current production model. It remains outside the production registry.

## Current automated verification (2026-10-12)

- Phase 8–17 gate: passing for runnable code/model/schema/leakage checks; evidence-dependent gates are separate.
- 59/59 automated tests pass.
- Python compilation: passing.
- Model SHA-256 integrity: passing; only `v20261001_104517` is in the production registry.
- Exact URL and domain overlap against the October 5 historical holdout: 0 / 0.
- Temporal status: `NOT_READY` without adequate authentic per-row timestamps.
- Fresh live feed status: `NOT_VERIFIED_IN_THIS_RUNTIME`; the Oct 5 capture is archived historical evidence.
- Dependency CVE status remains unverified until dependency/container scanners run in Internet-enabled CI.
- Distributed cloud capacity: not tested; the latest sandbox HTTP result is recorded in `artifacts/benchmarks/phase8_17_local_http_replay_2026-10-12.json`.

## Important release rule

No performance or accuracy figure is treated as real-world evidence unless it comes from actual captured/verified data and an identifiable execution environment. No 667K–1M RPS claim is made by this release.


## Phase 8–17 extension

- Character-level URL learners, word/host TF-IDF, structured learners and ensemble calibration were retained.
- Unicode/Punycode diagnostics are explanatory evidence only.
- A distinct fast-stage calibration/training path was added, but the active production model has no compatible fast calibrator; it therefore uses the full ensemble.
- A newer fast-path candidate failed the frozen-regression gate and recall non-regression gate. It was not promoted and is preserved only as evaluation metadata under `artifacts/phase8_17/candidate_evaluations/`.
- Local exact URL threat indexing now uses SHA-256 digests and canonical path/query-specific matching. The live refresh directory matches the API feed directory.
- Uncertain-only local DNS/TLS enrichment is opt-in, bounded, asynchronous and does not turn missing evidence into benign.
- L1/L2 cache supports per-entry TTL and provider/local-origin caching.
- The historical 2026-10-05 OpenPhish capture is now explicitly archived; it is not called current.

## Phase 8–17 finalization — 2026-10-12

- Seven-component URL ensemble retained; active model `v20261001_104517` is unchanged and SHA-256 verified.
- Candidate fast path remains disabled because it failed the recall non-regression threshold and frozen-regression gate.
- Added four-way verdict policy, IDN/Punycode diagnostics, exact-URL SHA-256 local threat-index, opt-in bounded DNS/TLS enrichment, and bounded L1/optional Redis L2 caching.
- Fixed provider-cache hit handling and held deep-analysis concurrency slots until background DNS/TLS work actually ends after timeout.
- Live feed refresh rejects empty/invalid provider responses and preserves the last-known-good feed.
- Current real-world feed freshness remains unverified in this build runtime. The October 5 feed is a historical capture.
- Temporal validation remains `NOT_READY` until sufficient authentic per-row timestamps exist. Dependency CVE status remains `UNVERIFIED` until Internet-enabled scanning runs.
- Real HTTP sandbox benchmark (Oct 12): 200/200 single requests at 73.52 req/s, p95 412 ms / p99 500 ms; batch 100/100 requests, 6,400 URL decisions at 667.02 decisions/s, p95 2,066 ms / p99 2,074 ms. These numbers are configuration-specific and are not evidence of 1M-RPS capacity.
