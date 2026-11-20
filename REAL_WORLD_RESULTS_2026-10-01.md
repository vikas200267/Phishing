# Real-world evaluation record — 2026-10-01

## Model

`v20261001_104517`

## OpenPhish stored-capture replay

The final production model was evaluated against three captured OpenPhish samples without visiting the listed URLs. Registrable domains appearing in the model training set were excluded before scoring.

| Capture | Rows | Unseen-domain rows | Detected | Positive recall |
|---|---:|---:|---:|---:|
| clean48 | 48 | 43 | 43 | 100.00% |
| unseen50 | 50 | 45 | 45 | 100.00% |
| late slice | 171 | 163 | 163 | 100.00% |
| **Total** | **269** | **251** | **251** | **100.00%** |

All labels in these samples are phishing-feed labels, so this record estimates positive-feed recall only; it does not estimate precision or Internet-wide accuracy.

## Realtime deployment path

For live deployment, a separate feed refresher downloads current feed text and atomically updates the local exact-threat index. The inference API can use that index, the trained model, caching and optional external enrichment. Listed threat URLs are not fetched merely to classify them.

## Throughput

The package contains:

- a 2,031-URL real-world replay corpus;
- an HTTP benchmark against the actual FastAPI endpoint;
- a distributed k6 constant-arrival-rate test;
- a capacity planner for the 667K–1M requests/second target.

No synthetic URL generator or hardcoded 1M-RPS measurement is used as evidence.

## Final HTTP endpoint evidence

The promoted model `v20261001_104517` was served through the actual FastAPI endpoint using the real replay corpus `data/realworld_load_corpus.json` (2,031 URLs, no generated URLs). With four workers, one inference thread per worker, cascade enabled (`0.10/0.90`), cache disabled and local intelligence disabled for an isolated ML benchmark:

- Single-URL HTTP: **800/800 successful**, 157.96 requests/s, p95 1,277.6 ms, p99 2,127.4 ms.
- 64-URL batch HTTP: **12,800 decisions**, 200/200 requests successful, 2,525 decisions/s, p95 1,192.7 ms, p99 1,243.0 ms.

A separate four-worker/concurrency-100 stress run produced 5 failed requests and p99 ~3.82 s, so that run is **rejected** as a production-capacity result. It demonstrates that this sandbox saturates under higher concurrency and that the distributed cloud test remains necessary.
