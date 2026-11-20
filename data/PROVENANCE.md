# PhishGuard Phase 1–7 Data Provenance

Production training/evaluation data must be traceable to a source file or an Internet-accessible feed. Every acquired dataset must retain retrieval time, source URL, SHA-256, label semantics, and whether listed URLs were visited. The production training pipeline rejects the `benign_seed` source and must never generate synthetic URLs.

## Source semantics

- OpenPhish / URLhaus: source-positive threat intelligence; suitable as positive evidence subject to source terms.
- Tranco: popularity candidate only. It is **not** a verified benign label and must not be silently converted into label 0.
- User-supplied verified benign data: permitted when its labeling method is documented and provenance is retained.

## Split rules

Exact normalized URLs must not cross train/test boundaries. Registrable domains should not cross an unseen-domain boundary. Temporal evaluation requires real observation timestamps; dates must never be fabricated.

## Release 2026-10-05
- Required phases 1–7 gate: PASS.
- Required gate checks: compile, tests, production doctor, production-corpus leakage audit.
- Temporal gate: NOT_READY because the baseline corpus lacks sufficient real observation timestamps; no timestamps are fabricated.
- Archived OpenPhish capture from 2026-10-05: `data/realworld/archive/openphish_20261005_unseen.csv` (source-positive, listed URLs not visited). It is a historical snapshot, not a current feed. Use `scripts/refresh_feeds.py` to fetch current data in an Internet-enabled environment.
