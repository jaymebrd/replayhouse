# Scaling probe: does sampling need a sub-linear structure? (2026-08-06)

Environment: embedded chdb on an Apple Silicon laptop (pessimistic bound —
a server has more cores and RAM). `k = 8192` per draw; medians.
Run: `.venv/bin/python benchmarks/bench_scaling.py`.

| rows | phase-1 ms | sample ms | sample+where ms | by=main-col ms | update-8k ms | bloat-3x sample ms | compact s | post-compact ms |
|------|-----------|-----------|-----------------|----------------|--------------|--------------------|-----------|-----------------|
| 1M   | 23        | 96        | 146             | 103            | 27           | 111                | 0.2       | 97              |
| 10M  | 109       | 318       | 658             | 179            | 23           | 396                | 1.6       | 235             |
| 50M  | 755       | 1142      | 2732            | 575            | 104          | 2642               | 8.3       | 1272            |

## Findings

1. **The two-phase SQL design holds at agentic-post-training scale.** A full
   weighted 8k-draw from 50M experiences is ~1.1s on a laptop — inside a
   typical multi-second training-step budget. At ≤10M rows everything is
   sub-second. Priority updates are effectively free (inserts, ~100ms for 8k).
2. **The bottleneck is not the weighted top-k.** Cost ranks: (a) the `where`
   semi-join (2.4× the unfiltered draw at 50M), (b) sidecar bloat from
   uncompacted priority versions (2.3× at 3× bloat; `compact` restores it),
   (c) the `argMax` dedup itself (hash aggregation over all sidecar ids —
   the bulk of phase-1).
3. **This run found a real bug before any numbers came out**: `sample(8192)`
   exceeded ClickHouse's default 256KB `max_query_size` by inlining 8192
   quoted UUIDs (fixed: chunked fetch, commit `6eb00b6`).

## Verdict on "special data structures"

Not needed at 10–100M experiences. If/when scale demands one, the data says
the *first* structure to build is not the Reverb-style sum-tree analogue
(per-granule weight sums) — it's making the dedup incremental: an
`AggregatingMergeTree` sidecar with `argMaxState(priority, version)`, so
last-writer-wins resolution happens at merge time instead of per-query.
Linear extrapolation puts the plain design at ~20s/draw at 1B rows; the
granule-weight hierarchical sampling idea earns its complexity only past
~500M rows or under high concurrent sampling QPS. Until then: compact on a
cadence, prefer `by=` main-column mode when priorities live in the payload,
and keep filters narrow.
