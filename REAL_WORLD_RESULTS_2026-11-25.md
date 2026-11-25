# Model investigation — 2026-11-25

The active URL model remains `v20261001_104517`, and `ML_BLOCKING_ENABLED`
remains false. Its scores are calibrated on its training distribution and are
not validated as probabilities for arbitrary websites. The active artifact
still scores all 21 reviewed homepage roots above its phishing threshold.

## New challengers

Both challengers used the existing PhishVN and LegitPhish inputs plus the
original production training corpus. The two legacy extracted sources were
weighted by 12 in out-of-fold fitting, stacking, calibration and final model
fitting. Frozen-regression, homepage, PhishVN test, LegitPhish holdout and
captured OpenPhish domains were reserved before training. They include undated
training data, so neither model qualifies for the pre-2023 temporal test.

| Measurement | `v20261125_062523` | `v20261125_063617` |
| --- | ---: | ---: |
| Frozen regression: benign false positives / 107 | 0 | 5 |
| Frozen regression: phishing misses / 106 | 17 | 7 |
| Homepage challenge: false positives / 21 | 1 | 1 |
| PhishVN test: false positives / 356 valid benign URLs | 2 | 24 |
| PhishVN test: phishing misses / 487 | 2 | 0 |
| LegitPhish holdout: false positives / 7,366 | 11 | 55 |
| LegitPhish holdout: phishing misses / 13,584 | 30 | 18 |
| October 5 OpenPhish archive: detections / 51 | 47 | 47 |

Two invalid HTTP input rows were excluded from the PhishVN test. The second
model's higher recall target was selected from its calibration fold, but the
target was not attained on that fold and the fallback threshold was recorded.
Neither candidate passes the frozen, homepage and temporal release checks.
The frozen and homepage sets have been used repeatedly during development;
their scores must not be represented as a fresh final acceptance test.

### Scheme shortcut audit

On the 843 valid PhishVN temporal holdout rows, the pre-cutoff model
`v20261118_103146` made exactly the same binary decisions as the simple rule
“HTTP is phishing, HTTPS is benign.” Both achieved 99.76% accuracy on this
sample. The holdout contains 487 HTTP and zero HTTPS phishing rows, and two
HTTP and 354 HTTPS benign rows. This confounding means the headline temporal
accuracy does not demonstrate general phishing detection. Replacing only the
scheme changed that model's decision on 615/843 URLs; the altered URLs were
not visited or relabeled, since the two schemes can serve different content.
Run `scripts/evaluate_shortcuts.py` to reproduce this diagnostic. The release
gate now requires at least ten valid holdout examples in each class/scheme
combination; this PhishVN holdout does not meet that requirement.

Only 15 of the 21 homepage observations completed as verified HTTP HTML;
the other six have recorded failed attempts. The release gate now requires at
least 15 distinct domains with successful recorded observations, in addition to
checking all challenge scores.

The training pipeline now reserves separate domain groups for the base and
stack fitting, calibrator fitting, and threshold selection. Earlier candidates
used a calibration fold that indirectly influenced the stacker; their
calibration results must be treated as diagnostic. A small corrected-split
smoke challenger (`v20261125_070600`, five trees per structured model) scored
0/21 homepage false positives but produced 102/107 benign false positives and
55/106 phishing misses on the reused frozen set. It is not promoted and is not
a production-quality training run.

A substantive mixed-source challenger using the corrected split and
120/100/120 trees (`v20261125_071843`) also failed release: on the reused
frozen set it had 2/107 benign false positives and 9/106 phishing misses; on
the 21-homepage challenge it had two false positives. It detected 48/51 URLs
in the October 5 OpenPhish archive and 267/276 input-domain-disjoint URLs in
the November 25 positive-only feed snapshot. On the LegitPhish holdout it had 25/7,366
benign false positives and 6/13,584 phishing misses. These datasets have been
repeatedly used during development and are not fresh independent acceptance
tests. The model includes undated inputs and is not temporally qualified.

## Current feed replay and registered-domain proxy

OpenPhish was refreshed at 06:25 UTC on November 25, receiving 300 entries.
The active model detected 284/284 entries remaining after exclusion of domains
from all recorded model inputs. The first and second challengers detected
257/272 and 268/276, respectively, after the same kind of exclusion. The
denominators differ because each model has different input domains. These are
positive-only feed replays: they do not establish accuracy or precision. The
URLs were not visited. URLhaus was skipped because its authentication key was
not configured, so local threat-intelligence coverage remains partial.

For an additional population check, a deterministic set of 500 registered U.S.
federal `.gov` domains was drawn from the [CISA registry](https://github.com/cisagov/dotgov-data),
after excluding the recorded input domains of all three models and the earlier
evaluation domains. The active model scored 500/500 roots at or above its
threshold, the first challenger 7/500, and the second 306/500. Registry
membership does not prove that a website exists or is safe; these figures are
not false-positive rates. The websites were not visited. The exact selected
sample and source hashes are recorded locally under
`artifacts/realworld/gov_registry_probe_20261125/`.

## Release and operational changes

- Release approval now checks the candidate's own hashed domain records,
  recomputes calibration and outer metrics, verifies homepage disjointness, and
  requires model-bound source-attested temporal evidence and fixed minimums.
- URL approval cannot enable email blocking. Email scores remain advisory.
- An unchanged feed expires when its freshness window ends, and the health
  endpoint recalculates its age. The API identifies model calibration as
  training-distribution-only.
- Domain grouping uses a pinned offline public-suffix snapshot. Live evaluation
  excludes domains from all available hashed model inputs and labels
  single-class feed measurements as recall only.
- The full test suite was run after these changes. The status and remaining
  limitations should be rechecked after future changes.
