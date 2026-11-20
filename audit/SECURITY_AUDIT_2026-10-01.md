# PhishGuard Security, Logic, and Result-Integrity Audit — 2026-10-01

## Final release audited

- Production model: `v20261001_104517`
- Model schema: 104 serving-compatible URL features
- Training sources: the two supplied labeled URL datasets only
- Manual benign seed in production training: none
- Synthetic training fallback: none
- Shadow model: disabled in the final package

## Findings fixed

| ID | Finding | Resolution |
|---|---|---|
| PG-01 | Feedback endpoint accepted verified labels without admin authentication | Admin key required |
| PG-02 | Model reload endpoint was unprotected | Admin key required |
| PG-03 | Verified feedback had unsafe domain-wide semantics | Exact normalized URL only |
| PG-04 | DNS/TLS enrichment could be abused for SSRF/DNS rebinding | Resolve/filter public IPs first; connect to validated IP with SNI |
| PG-05 | Batch path could diverge from trusted-feedback decisions | Shared decision path |
| PG-06 | Request and batch size limits were incomplete | Body/field/batch limits added |
| PG-07 | Legacy SQLite schemas could break verified feedback | Forward migration added |
| PG-08 | Rollback could expose incompatible model bundles | Schema/component compatibility gate |
| PG-09 | CURRENT pointer writes were non-atomic | fsync + atomic replace |
| PG-10 | Cache mutation was not synchronized | Locking added |
| PG-11 | Retraining wrapper used an incomplete argument schema | Full production parser used |
| PG-12 | Local threat-feed state could remain stale | Periodic reload added |
| PG-13 | Enrichment disable switch could be bypassed | Explicit 403 guard |
| PG-14 | Prometheus request/cache metrics were incomplete | Metrics wired into middleware/inference |
| PG-15 | Synthetic throughput traffic could be mistaken for production evidence | Removed from release evidence |
| PG-16 | Live-feed evaluation could include training-source overlap | Registrable-domain exclusion added |
| PG-17 | Model artifacts lacked integrity verification | SHA-256 manifest + fail-closed load |
| PG-18 | Suspicious cascade results could be represented as binary benign | Binary verdict becomes `unknown` for suspicious |
| PG-19 | Standalone real-world evaluator failed from the documented root command | Project-root import path fixed |
| PG-20 | Benchmark used an outdated `predict_urls()` signature | Current model object is passed explicitly |
| PG-21 | Production package contained obsolete seed/legacy artifacts | Removed from final release |

## Validation results

### Unseen-domain outer holdout

- Accuracy: **100.00%**
- Precision: **100.00%**
- Recall: **100.00%**
- F1: **100.00%**
- ROC-AUC: **100.00%**
- PR-AUC: **100.00%**
- Brier score: **0.055%**
- False positives: 0
- False negatives: 0

### Immutable historical regression

- Accuracy: **98.59%**
- Precision: **99.05%**
- Recall: **98.11%**
- F1: **98.58%**
- False positives: 1
- False negatives: 2

These are dataset measurements. The holdout is small and does not establish Internet-wide performance.

## Real-world feed replay

Three stored OpenPhish captures were replayed through the exact promoted model. Listed URLs were not visited. After excluding registrable domains present in model training, the evaluated unseen-domain rows were:

- 43/43 detected
- 45/45 detected
- 163/163 detected

Combined unseen-domain positive-feed recall: **251/251 = 100%**.

This is positive-class recall only. It is not a precision or overall accuracy measurement because the population is phishing-only.

## Throughput evidence

A 2,031-URL corpus composed of actual URLs from the supplied datasets plus a captured OpenPhish sample is included for replay. A direct model replay measured approximately **1,538 URL predictions/second** in the final isolated replay benchmark. The final FastAPI path, with four workers, one inference thread per worker and the validated cascade enabled, measured **157.96 single-URL requests/second** with 800/800 successful requests; a 64-URL batch path measured **2,525 decisions/second** with 200/200 successful requests. These measurements are hardware/runtime-specific and are not a claim of production capacity. A separate higher-concurrency stress run produced 5 failures and p99 ~3.82 s and was rejected as a release-capacity result.

The package includes a distributed k6 constant-arrival-rate load test for the 667K–1M requests/second target. An actual 1M RPS result is intentionally not embedded as a precomputed value.

## Dependency-security status

`pip-audit` was unavailable in this sandbox, so the dependency CVE status is **UNVERIFIED**. CI includes a reproducible `pip-audit` gate for an Internet-enabled build environment. The current model artifact was trained under the installed scikit-learn 1.8.0 runtime; scikit-learn's current documentation is 1.9.1, and its model-persistence documentation says loading models across scikit-learn versions is unsupported. A secure production rebuild should therefore retrain under the supported dependency line before deployment.

## Remaining limitations

- The labeled corpus is small relative to Internet-scale traffic.
- No labeled email corpus was supplied; therefore no email ML accuracy is claimed.
- External threat-intelligence providers have separate coverage, quotas, privacy implications and error modes.
- 667K–1M RPS remains an infrastructure target requiring a real distributed deployment test.
