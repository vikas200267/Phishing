# PhishGuard Phases 1–7 Release Record — 2026-10-05

## Phase 1 — Real production path
- Production API has no synthetic/fake model training fallback.
- Model checksum verification is required by default.
- Invalid/missing model artifacts fail closed.
- Test fixtures may contain examples, but production inference contains no generated test corpus.

## Phase 2 — Environment separation
- `config/dev.env.example` is development-only.
- `config/staging.env.example` uses the production code path with staging secrets.
- `config/prod.env.example` requires authentication and model-manifest verification.
- Production startup rejects unsafe authentication/manifest settings.

## Phase 3 — Real datasets
- Original labeled URL corpus remains the baseline.
- `data/history/openphish_20261001_snapshot.csv` is a real threat-feed snapshot.
- `scripts/ingest_realworld.py` fetches current OpenPhish, optional URLhaus, and current Tranco data.
- Tranco is stored as a candidate population only; it is never automatically labeled benign.
- Listed threat URLs are never visited by ingestion/evaluation tooling.

## Phase 4 — Provenance
- SHA-256 and row/URL counts are recorded in `data/manifests/dataset_manifest.json`.
- Real-world source metadata includes source URL, retrieval timestamp, source digest, and label semantics.

## Phase 5 — Leakage prevention
- Exact normalized URL overlap is checked.
- Registrable-domain overlap is checked for unseen-domain evaluations.
- `data/realworld/archive/openphish_20261005_unseen.csv` has 0 exact and 0 domain overlap with the promoted model's training-domain list.

## Phase 6 — Temporal validation
- `scripts/temporal_gate.py` requires real timestamps and more than one observation time.
- Current combined training corpus is correctly marked `not_ready` for temporal evaluation because most historical rows have no real timestamp.
- The system never fabricates timestamps. Repeated real feed snapshots can be accumulated under `data/temporal/` and evaluated once coverage is sufficient.

## Phase 7 — URL feature extraction
- The promoted model uses 104 features reproducible from the raw URL. The current release is `v20261001_104517`.
- No network-only feature is required on the ML path.
- Feature train/serve parity is tested explicitly.

## Phase gate result

The phase 1–7 gate passes all required checks. The temporal gate is recorded separately as `not_ready` because the production training corpus does not contain sufficient real observation timestamps; no timestamps are fabricated.

Current automated verification: **33 tests passed**.

Real-world capture used for the release gate: `data/realworld/archive/openphish_20261005_unseen.csv` — 51 source-positive URLs, exact/domain leakage against production training: 0/0, listed URLs visited: false.
