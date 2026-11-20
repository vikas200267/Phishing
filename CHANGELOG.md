## Final audited realtime release — v20261001_104517

- Removed stale/duplicate captured evaluation artifacts that could be confused with current evidence.
- Added data provenance documentation and `scripts/verify_release.sh`.
- Added a separately calibrated fast-stage candidate, but the active production model has no compatible fast-stage calibrator; final decisions stay on the full calibrated ensemble.
- Added Kubernetes horizontal-scaling baseline and a recorded local HTTP replay benchmark; the 667K–1M RPS target remains unverified until a distributed cloud run.
- Added exact artifact integrity checking for optional email models.

# PhishGuard Upgrade Changelog

## v7 — 2026-10-01

### Production model
- Historical v7 model: `v20261001_063527`
- Historical shadow challenger: `v20261001_062742` (not shipped in the final package).

### Accuracy / generalization
- Added URL character-boundary (`char_wb`) TF-IDF learner for patterns spanning host/path/token boundaries.
- Kept the serving-compatible 88-feature structured schema so live URLs can reproduce every structured feature.
- Retained XGBoost + LightGBM + ExtraTrees + URL/host TF-IDF + LinearSVC + char_wb diversity with a logistic meta-stack and Platt calibration.
- Added hard-negative weighting and bounded threshold optimization to avoid pathological near-1.0 decision thresholds on small calibration sets.

### Validation
- Unseen-domain holdout: 99.58% accuracy, 100.00% precision, 98.97% recall, 99.48% F1.
- Immutable v2 regression set: 99.06% accuracy, 100.00% precision, 98.11% recall, 99.05% F1.
- 0 false positives on the v2 regression set.

### Runtime / safety
- Added three-zone verdicts: `benign`, `suspicious`, `phishing`, while retaining `binary_verdict`.
- Local exact phishing-feed and verified-feedback paths can bypass ML when an exact match is known.
- Reserved/special-use domain policy is applied before ML.
- Batch serving avoids pandas row iteration and supports cache-backed horizontal deployment.
- Continuous-learning remains feedback-gated rather than auto-training directly from untrusted observations.

### Release QA
- Runtime test suite: 8/8 passed.
- Batch API accepts both string URLs and legacy `{"url": ...}` objects.
- Standalone frozen-set evaluator no longer depends on importing the FastAPI module.
- Local benchmark after release packaging: 5,000 predictions at ~1,173 predictions/sec in this environment.

These are dataset-specific measurements and are not Internet-wide accuracy guarantees.


## v8 audit details — 2026-10-01

- Corrected live-serving security issues and a legacy feedback-database migration failure discovered during audit.
- Corrected the continuous-retraining wrapper so it uses the complete production training argument schema.
- Added real OpenPhish current-feed sample evaluation and provenance.
- 50 current-feed phishing URLs, zero exact overlap with bundled source/reference URLs, 49 ML detections: **98.0% positive-only recall**. One miss is retained for analysis.
- Invalidated the earlier synthetic throughput figure; the current benchmark is dataset replay only.
- Dependency CVE status remains unclaimed because a package vulnerability scanner was unavailable in the isolated runtime.

## v9 — 2026-10-01

### Release correction and runtime hardening
- Re-promoted `v20261001_063527` after candidate `v20261001_102206` failed the immutable regression gate.
- Removed synthetic/default-benign behavior from the production decision path.
- Added a protected `/v1/evaluation/latest` endpoint exposing only stored real-world evaluation artifacts.
- Fixed standalone script imports so the documented real-world evaluator runs from the project root.
- Fixed the cascade compatibility field so `suspicious` is not represented as a false `benign` binary verdict.
- A clean retrain excluding all manually seeded benign URLs was attempted but resource-killed in this sandbox; no unverified candidate was promoted.

## 2026-10-01 — real-world + scale release gate
- Added live OpenPhish/URLhaus/Tranco evaluator with source hashes and provenance.
- Added k6 constant-arrival-rate distributed load test targeting 667K–1M decisions/s.
- Added capacity planning utility and non-cache benchmark mode.
- Added CI pip-audit gate and dependency-audit requirements.
- Added configurable API worker count.
- Added clean-build requirements for scikit-learn 1.9.1+; current artifact remains pinned by its build provenance until retrained.


## Final clean release — v20261001_104517
- Production training provenance contains only the two supplied labeled URL datasets.
- Removed manual benign-seed and legacy model artifacts from the final package.
- Promoted only after unseen-domain and immutable regression gates.
- Added protected `/v1/evaluation/latest` endpoint and corrected benchmark/evaluator entry points.
