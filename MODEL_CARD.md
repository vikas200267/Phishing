# PhishGuard Model Card — v20261001_104517

## Production model

November 25, 2026: The model's ordinary-site false positives remain unresolved.
Two new challengers failed release checks; see `REAL_WORLD_RESULTS_2026-11-25.md`.
Its scores should not be interpreted as calibrated probabilities for the
current Internet population, and URL model-only blocking remains disabled.

`v20261001_104517` is the only promoted artifact in the final release package. It was trained from the two supplied labeled URL datasets only. Manual `benign_seed` rows and synthetic examples are not part of its training input.

## Model architecture

- 104 serving-compatible URL security features
- XGBoost + LightGBM + ExtraTrees structured learners
- URL character/token TF-IDF learner
- Host character TF-IDF learner
- LinearSVC URL-character learner
- URL `char_wb` learner
- Logistic meta-stack
- Platt/sigmoid probability calibration

## Validation

Unseen-domain outer holdout (207 rows: 130 phishing, 77 legitimate, across 95 held-out domains):

| Metric | Result |
|---|---:|
| Accuracy | **100.00%** |
| Precision | **100.00%** |
| Recall | **100.00%** |
| F1 | **100.00%** |
| ROC-AUC | **100.00%** |
| PR-AUC | **100.00%** |
| Brier score | **0.055%** |

Immutable historical regression set:

| Metric | Result |
|---|---:|
| Accuracy | **98.59%** |
| Precision | **99.05%** |
| Recall | **98.11%** |
| F1 | **98.58%** |
| False positives | **1** |
| False negatives | **2** |

These are dataset measurements only; the outer holdout is small (207 samples) and cannot establish Internet-wide performance. A perfect result on this holdout must not be extrapolated to all current URLs.

## Real-world replay evidence

Three **historical** OpenPhish captures were evaluated without visiting the listed URLs and with registrable domains overlapping model training excluded. Unseen-domain rows: **43, 45, and 163**; detections: **43, 45, and 163**. Combined positive-feed recall: **100% (251/251)** for those dated captures only. These archives do not represent the feed as of today. Because the sample is phishing-only, this does not estimate precision or accuracy.

## Security / operations

- No synthetic model fallback.
- No automatic retraining from unverified observations.
- Verified feedback is exact-URL scoped and admin authenticated.
- Model artifact is SHA-256 verified before load.
- DNS/TLS enrichment blocks private/local resolutions and runs off the hot path.
- Batch requests are bounded.
- Model pointer updates are atomic.
- Production defaults disable remote enrichment and shadow inference.
- Dependency CVE status is **not claimed** unless `pip-audit` has actually run in an Internet-enabled CI environment.

## Email status

The API includes email/header and embedded-URL analysis, but no email ML accuracy is claimed because the supplied project data does not include a labeled email corpus.

## External audit update — 2026-11-18

The active artifact `v20261001_104517` was evaluated against a separately sourced,
registrable-domain-disjoint LegitPhish v2 holdout (20,950 rows, 11,434 domains).
It labeled 7,365 of 7,366 legitimate holdout URLs as phishing (ROC-AUC 0.0055),
so the earlier small internal evaluation figures do not describe performance
on this independent corpus.

Candidate `v20261117_182148` was trained from a deterministic 20,000-row sample
of that corpus plus existing project sources. On the untouched holdout it scored
99.79% accuracy, 99.99% precision, 99.75% recall, and 0.9999 ROC-AUC, with 9
false positives and 34 false negatives. The candidate was not promoted: frozen
regression accuracy/recall are below release requirements, and it flags three
of 21 reviewed official homepage roots. These are dataset-specific results and
need further independent validation.

`ML_BLOCKING_ENABLED` remains false. A narrow exact-root homepage policy was
added to reduce the known homepage false positives; it preserves the raw model
score separately and does not apply to subdomains, paths, query strings, or
exact threat-intelligence/verified-feedback matches. It is not a substitute
for general model accuracy. The external corpus has no reliable per-URL
timestamps, and the temporal evaluation gate remains not ready. See
`REAL_WORLD_RESULTS_2026-11-18.md` for full details.
