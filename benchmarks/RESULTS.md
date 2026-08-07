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

# Sidecar dedup variants: argMaxState vs FINAL (2026-08-07)

Same environment. `k = 8192` weighted draws over a priority sidecar at
10M/50M ids, compacted and 3x-bloated (two extra versions per id).
Run: `.venv/bin/python benchmarks/bench_sidecar_variants.py`.

| variant | 10M compact | 50M compact | 10M bloat3x | 50M bloat3x |
|---|---|---|---|---|
| A `argMax` GROUP BY (current) | 83ms | 428ms | 196ms | 2488ms |
| B `AggregatingMergeTree` + `argMaxMerge` | 89ms | 531ms | 261ms | 3321ms |
| C `ReplacingMergeTree` + `FINAL` | 18ms | 62ms | 89ms | 404ms |
| D raw scan, compacted (floor) | 21ms | 76ms | — | — |

## Findings

1. **The `argMaxState` hypothesis is falsified.** `argMaxMerge` still runs a
   hash aggregation over every id at read time; the aggregate states only
   add per-row overhead (~1.2x slower than the current design in every
   cell). No schema change to `AggregatingMergeTree` is warranted.
2. **`FINAL` is the winner, by a lot.** Modern ClickHouse executes `FINAL`
   on `ReplacingMergeTree` as a parallel merge-sorted dedup over the
   `ORDER BY id` key — no hash table — landing at the raw-scan floor when
   compacted (62ms vs 76ms at 50M) and ~6-7x faster than the current
   `argMax` read in every state (428→62ms compacted, 2488→404ms bloated
   at 50M).
3. **Consequence:** the improvement is a *query* change, not a schema
   change — phase-1 sampling should read `FROM <sidecar> FINAL` instead of
   the `argMax(priority, version) GROUP BY id` CTE. Semantics are identical
   (`ReplacingMergeTree(version)` keeps the max-version row). Planned as a
   follow-up library change with the full test suite as the semantic gate.
