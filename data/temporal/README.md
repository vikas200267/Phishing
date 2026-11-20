# Real temporal evidence

The project must accumulate timestamped snapshots from real sources. Never assign invented dates to old rows.

Recommended process:

```text
2026-10-05 feed snapshot -> data/feeds/live/<date>
2026-10-06 feed snapshot -> data/feeds/live/<date>
2026-10-07 feed snapshot -> data/feeds/live/<date>
...
```

Once the labeled corpus contains sufficiently complete `observed_at`/`collected_at` coverage across multiple observation times, run `scripts/temporal_gate.py`. Until then, the project must report temporal evaluation as `not_ready` rather than fabricating a temporal score.
