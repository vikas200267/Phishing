# Real-world retraining and release status — 2026-11-18

## Actions completed

- Acquired the manually reviewed LegitPhish v2 URL dataset (Mendeley DOI 10.17632/hx4m73v2sf.2, CC BY 4.0). The source has 101,219 rows, one unlabeled row, and no per-URL observation timestamps. Raw-data hash and mirror provenance are recorded next to the data. Source: https://data.mendeley.com/datasets/hx4m73v2sf/2.
- Normalized labels, removed 67 malformed/non-HTTP(S) records, contradictory normalized URLs (none), and 346 duplicate normalized URLs. Removed overlap with the production corpus and existing evaluation corpora before a deterministic registrable-domain split.
- Kept the full 20,950-row, 11,434-domain external holdout out of training. Its rows were not visited and have no timestamps. The candidate's `training_domains.txt` was checked against it: zero domain overlap.
- Trained candidate `v20261117_182148` using a deterministic sample of 10,000 URLs per class plus existing source corpora. The candidate was not promoted.

## Same-corpus comparison

| Model | Accuracy | ROC-AUC | Legitimate false positives | Phishing false negatives |
| --- | ---: | ---: | ---: | ---: |
| Active `v20261001_104517` | 64.68% | 0.0055 | 7,365 / 7,366 | 34 / 13,584 |
| Candidate `v20261117_182148` | 99.79% | 0.9999 | 9 / 7,366 | 34 / 13,584 |

The candidate materially repairs the class inversion seen in the active artifact. Its unusually strong score warrants additional independent datasets before any claim of general production accuracy.

## Homepage false-positive mitigation

A 21-homepage challenge contains the six names present in the supplied report plus 15 independently selected official organization homepages; it does not recreate the original 15 because the remaining nine URLs were not supplied. On raw model outputs the active model flags all 21, and the candidate flags 3 (GitHub, Ubuntu, and the EU homepage).

The API inference path now recognizes only exact HTTPS origin roots for these reviewed domains. It does not match subdomains, non-root paths, query strings, or HTTP; exact verified-feedback and threat-feed evidence takes priority. The response keeps the raw value as `model_risk_score` and marks the final result `verified_official_homepage_root`. The direct inference pipeline returned benign for all nine exercised homepage cases, including the three candidate false positives. This narrowly fixes known homepages; it does not prove general model accuracy. A direct observation run recorded attempt times for all 21 URLs; 15 requests completed and 6 were refused after TLS certificate verification failed. The remaining entries retain independently checked official-host provenance but have no verified live fetch.

## Additional data and email advisory — 2026-11-18

- Retrieved the MeAJOR cleaned email dataset from its [Zenodo record](https://zenodo.org/records/18471483), DOI 10.5281/zenodo.18471483. The CSV is 191,121,228 bytes; SHA-256 `5bf5371897ab4e4aa7ac6d8f499869e6c32e4456ed09636731e4069947b69f4f`; the Zenodo file MD5 is checked by the preparation script. The source's dates refer to historical email message dates and must not be presented as current threat-feed timestamps.
- Removed unparseable/out-of-range dates, exact normalized subject/body duplicates, and contradictory-content groups. Kept a strict date cutoff: training rows precede 2007-01-01; evaluation rows are on/after it. No exact normalized content overlaps the split.
- Trained email artifact `models/candidates/email_v20261118/email_model.joblib` and made it available to the active API as an advisory-only email scorer. Historical holdout (42,561 rows): accuracy 84.18%, precision 85.72%, recall 85.22%, ROC-AUC 0.901, FPR 17.07%, FNR 14.78% at threshold 0.50. At the API's default 0.75 threshold, recall is 71.09%, FPR 12.25%, and precision 87.47%. The evaluation is source-shifted: nearly all test messages are from TREC7 while most training data is TREC5/6. It is not suitable for blocking or a contemporary accuracy claim. `ML_BLOCKING_ENABLED` remains false; email decisions therefore stay advisory/suspicious, never a block.

## Captured threat intelligence and real-world replay

- OpenPhish was successfully fetched from its public mirror on 2026-11-18 06:58 UTC: 300 listed URLs, archived as a timestamped snapshot, with raw-feed SHA-256 and retrieval metadata. The feed refresher is configured to poll every 12 hours, retry failures, preserve the last good feed, and retain snapshots for 90 days. This is a snapshot of feed collection, not a timestamp for when each listed URL was first malicious.
- URLhaus refresh was skipped because `URLHAUS_AUTH_KEY` is not configured. The feed health remains partial, not fully fresh across providers.
- Candidate replay on that OpenPhish snapshot excluded training-domain overlaps: 269/295 detected (91.19% recall, 26 misses). Active model detected 296/296 unseen feed entries in its replay, but had previously flagged all 21 independently reviewed homepages. Candidate flagged 3/21 homepages (14.29% challenge FPR). These small, source-specific samples are not broad precision estimates.
- On the 2026-10-05 captured OpenPhish archive, candidate detected 47/51 unseen-domain entries (92.16% positive recall). There is no corresponding independently labeled benign set for that historical point, so this is not a full temporal benchmark or precision result.
- Homepage observations now record the actual attempt time and verification status. Only 15/21 direct requests completed; six failed TLS certificate validation. Attempt timestamps are not substituted for original publication/first-seen timestamps.

## URL temporal benchmark — 2026-11-18

### Commercially usable, source-attested temporal split

- Acquired PhishVN v5 from [Mendeley Data](https://data.mendeley.com/datasets/b97hxbxtpd/5), licensed CC BY 4.0 with attribution to Thai Nguyen Vu and upstream NCSC Tin Nhiem Mang, ChongLuaDao, and Tranco sources. The verified archive SHA-256 is `4c610431fd275d8348695653eeacdece861dc7dc65ab09a2068d519247988968`; acquisition details are in `data/realworld/external/phishvn/download_manifest.json`. Downloaded source rows remain ignored by Git.
- The reproducible preparation script uses only official URL train/test folds and gold/silver labels. It parses source `collected_at` dates; `scraped_at` retrieval dates and undated rows are excluded. At the strict 2023-01-01 UTC cutoff it produces 3,183 train URLs (2020-02-08 through 2022-12-27) and 845 test URLs (2023-02-07 through 2026-03-25). Both labels meet support requirements; timestamp coverage is 100%; exact-URL and registrable-domain overlap are both zero. The dataset gate now passes in `data/manifests/url_temporal_gate.json`.
- A new URL challenger (`v20261118_103146`) was trained only on those 3,183 pre-cutoff rows. On 843 valid-HTTP temporal holdout rows it scored 99.76% accuracy, 100% recall, 99.59% precision, and 0.56% FPR (487 TP, 354 TN, 2 FP, 0 FN). Two records with URL user-info syntax were rejected by the API URL policy and are explicitly excluded from inference metrics. On the 21 URL legitimate-homepage challenge it produced 0 false positives.
- The high temporal numbers are not sufficient for release: the same candidate fails the independent frozen regression badly (26.29% accuracy, 95.33% FPR; 102 FP and 55 FN). Its training pipeline's internal grouped split was also weaker than its reported OOF calibration metrics: 99.82% accuracy and 0.68% FPR on 558 rows, but that small split is not the release authority. `models/CURRENT` remains `v20261001_104517`, challenger promotion is false, and ML blocking remains disabled. The source data gate is resolved; production model release and temporal qualification of the active model are not.

Preparation and evaluation commands:

```powershell
.venv\Scripts\python.exe scripts\prepare_phishvn_temporal.py
.venv\Scripts\python.exe scripts\temporal_gate.py data\realworld\evaluations\phishvn_temporal\dataset.csv --timestamp-col event_time --cutoff 2023-01-01 --source-metadata data\realworld\evaluations\phishvn_temporal\dataset.csv.meta.json --out data\manifests\url_temporal_gate.json
.venv\Scripts\python.exe scripts\evaluate_phishvn_temporal.py --model v20261118_103146
```

### Separate noncommercial benchmark

- Acquired both checksum-verified parquet shards of [DeepURLBench](https://huggingface.co/datasets/DeepInstinct/DeepURLBench). Its published schema provides a per-record `first_seen` timestamp and labels including benign and phishing. The source timestamp is timezone-naive; it was kept as supplied and used only for chronological ordering. URLs were never visited.
- License is CC BY-NC 4.0. The corpus and sampled URL rows are ignored by Git and are for noncommercial research evaluation only; they were not used to train or package the production model.
- Scanned all 40,194,045 source rows and created a deterministic row-group sample of 63,435 unique labeled URLs with source prevalence preserved. The strict cutoff is 2022-01-01: train 28,986 rows (2010-10-21 through 2021-12-31), evaluation 34,449 rows (2022-01-01 through 2023-03-08); timestamp coverage is 100%, both classes occur on both sides, and exact URL overlap is zero. Its full sample and source checks remain in the ignored `data/realworld/evaluations/deepurlbench_temporal/` report; the canonical `data/manifests/url_temporal_gate.json` now points to the commercially usable PhishVN split above.
- A separate research baseline trained only before the cutoff and calibrated on a later pre-cutoff slice, then evaluated on 28,150 post-cutoff rows from registrable domains absent from training. At a threshold selected for calibration FPR ≤5%, test accuracy was 80.04%, precision 88.01%, recall 43.62%, ROC-AUC 0.868, and FPR 2.80%. This exposes substantial temporal recall drift. It is a character TF-IDF/logistic baseline, not the production model.
- This secondary benchmark is not used for model training because of its noncommercial license. PhishVN now supplies the deployable-license temporal training and evaluation source, but it still does not retroactively timestamp the active production corpus.

## Cross-source challenger investigation — 2026-11-19 update

To diagnose the high false-positive rate on ordinary websites, I trained additional unpromoted challengers with PhishVN and other CC BY 4.0 public sources. The [LegitPhish v2 dataset](https://data.mendeley.com/datasets/hx4m73v2sf/2) contains 101,219 labeled URL rows; its per-row event dates are blank, so it is used for cross-source training only and cannot support a strict pre-cutoff model claim.

- `v20261118_143528` combines the PhishVN training split with a deterministic 20,000-row balanced sample of LegitPhish v2. It scored 99.79% accuracy on the separate 20,950-row LegitPhish domain holdout (13,548/13,584 phishing URLs detected; 7,357/7,366 benign URLs correctly cleared), with 0/21 legitimate homepage false positives. It detected 41/51 URLs in the captured October 5 OpenPhish archive (80.39% positive recall). It failed the frozen regression population catastrophically: 100% FPR on 107 benign rows, despite strong scores on the other independent sources.
- `v20261118_144004` combines the PhishVN training split with the original production corpus. It scored 96.24% frozen accuracy with 0 false positives, but recall was 92.45% (8/106 phishing misses) and it flagged 17/21 legitimate homepage roots. On the PhishVN post-cutoff holdout it scored 99.05% accuracy with 2.25% FPR; on the 20,950-row LegitPhish holdout its 0.0545 threshold produced 84.24% accuracy and 44.57% FPR.
- Thresholds for these models were selected using internal training calibration folds. The cross-source contradictions mean a single threshold does not transfer across the frozen, homepage, and larger reference populations. The PhishVN time split remains a valid **dataset** gate, but candidates that include undated or post-cutoff sources are not temporally qualified.
- Neither cross-source challenger was promoted. `models/CURRENT` remains the original model; blocking remains disabled. The existing homepage API policy clears only exact roots listed in `data/realworld/trusted_homepages.csv`; it does not validate arbitrary sites or deep links.
- Per-candidate/per-source metrics and input hashes are reproducibly recorded in `data/realworld/evaluations/crosssource_candidate_metrics.json`, generated by `scripts/evaluate_crosssource_candidates.py`.

### Latest mixed-source candidate

Candidate `v20261119_064457` extends the cross-source training mixture with the original production corpus and uses its frozen artifact threshold (0.28543). On the 21-site legitimate homepage challenge, it cleared 19 sites and falsely flagged 2 (9.52% FPR), compared with 21/21 raw false positives from the active model. It scored 99.84% accuracy on the 20,950-row LegitPhish domain holdout (19 benign false positives, 15 phishing misses), and detected all 51 URLs in the captured October 5 OpenPhish archive. That archive is positive-only and cannot measure precision.

The candidate still fails the frozen regression gate: 93.90% accuracy, 88.68% recall, 0.93% FPR, with 12 phishing misses. It is not strictly temporally qualified because the mixture includes sources without authentic per-URL timestamps and may include post-cutoff rows. These cross-source results are diagnostic; they do not qualify the candidate for production. Reproduce the comparison with `scripts/evaluate_crosssource_candidates.py`; full metrics, source hashes, and the qualification reason are recorded in `data/realworld/evaluations/candidate_v20261119_064457_metrics.json`.

### Expanded LegitPhish training coverage — 2026-11-19

To test whether broader legitimate URL coverage improves the model, I built a deterministic balanced mix from the full eligible LegitPhish v2 training split (29,377 legitimate and 29,377 phishing rows) plus the original 1,308-row production corpus. Conflicting URLs were removed; the mix contains 60,062 rows total. The source has no authentic per-URL timestamps, so the resulting model is not temporally qualified. Preparation is reproducible with `scripts/build_full_legitphish_training_mix.py`; the generated local dataset is intentionally excluded from version control, and its input/output hashes and counts are in its adjacent manifest.

Candidate `v20261119_083010` was trained with the 3,183-row pre-cutoff PhishVN train set plus this expanded mix. Its internal unseen-domain split scored 99.82% accuracy, 99.85% recall, and 0.22% FPR across 13,140 rows (zero exact-URL/domain overlap). On the independent challenge sets at its frozen threshold (0.32273):

- Frozen regression: 95.31% accuracy, 91.51% recall, 0.93% FPR; 9 phishing misses and 1 benign false positive. The frozen recall release requirement is 98%.
- Legitimate homepage challenge: 1/21 raw false positives (the European Union root), down from 2/21 for `v20261119_064457`; the active model still scores all 21 as phishing at its raw threshold.
- LegitPhish v2 domain holdout: 99.78% accuracy; 34/7,366 benign false positives and 13/13,584 phishing misses (0.46% FPR).
- PhishVN post-cutoff holdout: 99.76% accuracy, 100% recall, 0.56% FPR, excluding two URLs rejected by the API URL policy.
- Captured October 5 OpenPhish archive: 50/51 detections; the archive is positive-only and does not measure precision.

Raising the operating threshold enough to clear the remaining homepage false positive would sharply reduce frozen phishing recall (to 76.42% at threshold 0.99); a threshold change does not solve the generalization problem. Full fixed-threshold metrics and source hashes are in `data/realworld/evaluations/candidate_v20261119_083010_metrics.json`. The candidate remains unpromoted.

## Release decision and remaining blockers

`models/CURRENT` remains `v20261001_104517`; URL candidates `v20261117_182148`, `v20261118_103146`, `v20261118_143528`, `v20261118_144004`, `v20261119_063145`, `v20261119_063844`, `v20261119_064457`, and `v20261119_083010` were not promoted. The expanded-data candidate reduces homepage false positives to 1/21 and raises frozen recall to 91.51%, but still misses 9 phishing URLs against the 98% release requirement and lacks strict temporal qualification. No candidate passes all release gates. The timestamp-data gate is resolved, but the active model has no authentic per-row training timestamps and is not temporally qualified. Email has a separate historical temporal split, with the source-shift and elevated false-positive limitations above.

The local OpenPhish feed was refreshed from its configured public mirror on 2026-11-19 08:36 UTC: 300 URLs were received, and `/health` reported the feed fresh with an approximately 24-hour freshness limit. Snapshot and raw-feed hashes are stored in the ignored `data/feeds/live/` directory; the refresher archives snapshots and retries failures. URLhaus remains skipped because `URLHAUS_AUTH_KEY` is not configured. This verifies a local feed refresh, not continuous operation of a cloud deployment. Email detection remains advisory, no production deployment or cloud-scale capacity is verified, and production ML blocking remains disabled.

The email artifact loads with its manifest hash verified, direct email-model inference returns a score, and the API health function reports the model loaded, blocking disabled, and feed coverage partial. A missing email score-to-verdict mapping was found and added. Python compilation passes. A local ASGI HTTP smoke check could not connect because this runtime forbids loopback sockets; pytest and a complete HTTP integration suite were not run.
