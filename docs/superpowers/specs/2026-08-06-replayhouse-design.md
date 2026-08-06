# ReplayHouse: an agentic experience store with native weighted sampling

Date: 2026-08-06
Status: draft for review

## Summary

ReplayHouse is a Python package (`replayhouse` on PyPI, module
`replayhouse`) that turns ClickHouse into an experience replay store for
LLM/agent post-training, in the spirit of DeepMind Reverb but built for
agentic workloads: large variable-length trajectories, batch-shaped
prioritized sampling fused with analytical filters, and a single store that
serves both training and observability.

v1 requires **no ClickHouse core changes**. All behavior is implemented as
SQL patterns over stock ClickHouse (server via `clickhouse-connect`, or
embedded via `chdb`). Native core features (`SAMPLE k BY <weight>` syntax,
per-granule weight statistics, a `ReplayMergeTree` engine) are explicitly out
of scope for v1 and will be justified later by benchmarks from this package.

## Goals

- Prioritized (weighted, without-replacement) sampling of experience
  batches, combinable with arbitrary SQL filters and stratification.
- Append-heavy trajectory ingestion from many concurrent rollout workers.
- Cheap batch priority updates from the trainer (no mutations in the hot loop).
- Time- and capacity-based eviction (FIFO and lowest-priority policies).
- Zero-infrastructure local mode (chdb) and cluster mode (ClickHouse server)
  behind the same API.
- A PyTorch-friendly consumption path.

## Non-goals (v1)

- Reverb-parity low-latency RL: kHz sample/update loops, actor/learner rate
  limiting. Classic small-transition DQN-style replay is not the target.
- ClickHouse core/C++ changes of any kind.
- Backend-agnosticism (DuckDB/Postgres backends). The storage interface is
  kept internally clean ("run SQL, return Arrow") but only ClickHouse
  backends ship.
- A hosted service, auth, or multi-tenancy beyond what ClickHouse provides.

## Primary workload

LLM/agent post-training: trajectories with tool calls, model outputs,
rewards, judge scores. Sampling is batch-shaped (e.g. 8192 trajectories per
training step, seconds of latency acceptable), priorities are
staleness-tolerant, datasets exceed RAM. Bandit-style decisioning (Thompson /
UCB draws over an arms table) falls out of the same primitive with `k = 1`
and aggregate weights.

## Architecture

```
rollout workers ──insert()──▶ ┌─────────────────────────────┐
                              │ ClickHouse                  │
trainer ── sample() ────────▶ │  <name>            (MergeTree, fat rows)
      └─ update_priorities() ▶│  <name>__priorities (ReplacingMergeTree,
                              │                      narrow sidecar)
dashboards / SQL ───────────▶ └─────────────────────────────┘
```

The package is a thin client (~1–2 kLOC): connection handling, DDL helpers,
the two-phase sampling query, batch priority writes, eviction queries, and an
Arrow/PyTorch adapter. ClickHouse does all the heavy lifting.

### Backends

- `ReplayHouse.connect("clickhouse://host:8123/db")` — server mode via
  `clickhouse-connect` (required dependency).
- `ReplayHouse.connect("chdb:///path/to/dir")` — embedded mode via `chdb`
  (optional extra: `pip install replayhouse[embedded]`). Single-process
  only; documented as the laptop/prototyping mode and used for CI tests.

Both backends speak identical SQL through an internal `Backend` protocol:
`query(sql, params) -> pyarrow.Table`, `insert(table, arrow_table)`,
`command(sql)`.

Minimum ClickHouse version: 25.3 (for the production `JSON` column type);
chdb pinned to a release embedding ≥ 25.3.

## Data model

Each logical store `<name>` is two physical tables.

### Main table `<name>` (append-only, immutable)

Created by `store.create(...)` from a user-supplied payload schema plus
required system columns:

```sql
CREATE TABLE agent_experiences
(
    id           UUID DEFAULT generateUUIDv7(),
    inserted_at  DateTime DEFAULT now(),
    -- user payload columns, e.g.:
    task_family  LowCardinality(String),
    env_version  UInt32,
    steps        JSON,
    reward       Float32,
    advantage    Float32
)
ENGINE = MergeTree
ORDER BY id
PARTITION BY toStartOfDay(inserted_at)
TTL inserted_at + INTERVAL 30 DAY    -- if ttl is configured
```

`generateUUIDv7` keeps ids time-ordered so primary-key fetches of a sampled
batch touch few parts. Day partitions make FIFO capacity eviction a cheap
`DROP PARTITION`.

### Priority sidecar `<name>__priorities` (narrow, hot)

```sql
CREATE TABLE agent_experiences__priorities
(
    id        UUID,
    priority  Float32,
    version   UInt64        -- monotonic, client-supplied (e.g. ns timestamp)
)
ENGINE = ReplacingMergeTree(version)
ORDER BY id
```

Every `insert()` writes the initial priority here as well as the payload row,
so the sidecar always covers all ids. `update_priorities()` is a plain batch
insert of new versions — no mutations. At ~13 bytes/row uncompressed, 100M
experiences is roughly 1 GB: scannable in well under a second, forever,
regardless of how fat the main table grows.

Deduplication happens at read time via `argMax(priority, version) GROUP BY
id` (not `FINAL`), so sampling correctness never depends on merge timing.

## Sampling design

Weighted sampling without replacement uses the A-ES / Efraimidis–Spirakis
reservoir: draw the top `k` rows by key `-ln(rand()) / weight`. This is
implementable as plain `ORDER BY ... LIMIT k`, is vectorized and parallel,
and merges associatively across parts, threads, and shards — so it works
unchanged through a `Distributed` table.

Execution is two-phase to avoid ever scanning fat payload columns:

**Phase 1 — choose ids (narrow scan).**

```sql
WITH current AS
(
    SELECT id, argMax(priority, version) AS priority
    FROM agent_experiences__priorities
    GROUP BY id
)
SELECT id
FROM current
WHERE priority > 0
  AND id IN (
      SELECT id FROM agent_experiences
      WHERE env_version >= 12            -- user filter, main-table columns
  )
ORDER BY -log(rand()) / priority
LIMIT 8192
```

The user filter is applied as a semi-join against the main table, which scans
only the narrow columns the filter references. When no filter is given the
semi-join is omitted.

Stratified sampling (`stratify_by="task_family"`) replaces `LIMIT k` with
`LIMIT k/num_groups BY task_family` after joining the group column in — even
coverage across groups with the same weighted draw within each group.

**Phase 2 — fetch payloads.**

```sql
SELECT * FROM agent_experiences WHERE id IN (%(ids)s)
```

Primary-key lookup of `k` rows; returns Arrow.

The `by=` argument accepts any SQL expression over sidecar and main columns
(e.g. `by="advantage"` resolves to a main-table column and skips the sidecar
entirely; `by="priority * exp(-age_hours/24)"` mixes both). `by="priority"`
(the sidecar) is the default.

Semantics documented to users: without replacement, per-call independence,
priorities are read-committed (a concurrent `update_priorities` may or may
not be visible to an in-flight `sample`). No sampling-key `SAMPLE` clause is
used or required.

## Priority updates

`table.update_priorities(ids, priorities)` inserts `(id, priority, version)`
rows with `version = time.time_ns()`. Idempotent, batched, safe from
concurrent trainers (latest version wins). The sidecar's `ReplacingMergeTree`
compacts old versions in the background; a `table.compact()` helper issues
`OPTIMIZE ... FINAL` for users who want to bound sidecar growth explicitly.

Alternative considered: lightweight `UPDATE` (patch parts, 25.7+) on a single
table. Rejected for v1 because the sidecar needs no version gate, has
strictly predictable cost, and keeps the main table append-only; the
benchmark harness (below) will compare both and may promote lightweight
`UPDATE` later.

## Eviction

Configured at `create()`, executed by `table.evict()` (called explicitly, or
by whatever scheduler the user runs; no daemon ships in v1):

- **TTL**: delegated entirely to ClickHouse `TTL` on `inserted_at`.
- **`capacity_bytes` / `capacity_rows` with `eviction="fifo"`**: drop oldest
  day partitions until under capacity.
- **`eviction="lowest_priority"`**: `DELETE FROM <name> WHERE id IN (bottom-k
  by current priority)` via lightweight delete, sized to return under
  capacity; matching sidecar rows deleted likewise.

Eviction never runs implicitly inside `insert()` or `sample()`.

## Python API surface

```python
from replayhouse import ReplayHouse

store = ReplayHouse.connect("clickhouse://host:8123/db")   # or "chdb:///dir"

store.create(
    "agent_experiences",
    columns={
        "task_family": "LowCardinality(String)",
        "env_version": "UInt32",
        "steps": "JSON",
        "reward": "Float32",
        "advantage": "Float32",
    },
    ttl_days=30,                      # optional
    capacity_bytes="2TiB",            # optional
    eviction="lowest_priority",       # "fifo" | "lowest_priority"
)

t = store.table("agent_experiences")

t.insert(rows)          # list[dict] | pyarrow.Table | pandas.DataFrame;
                        # optional per-row "priority" key, default 1.0

batch = t.sample(
    8192,
    by="priority",                    # any SQL expression; default "priority"
    where="env_version >= 12",        # optional SQL predicate
    stratify_by="task_family",        # optional column
)
batch.ids               # list[UUID]
batch.table             # pyarrow.Table
batch.to_pandas()

t.update_priorities(batch.ids, new_priorities)
t.evict()
t.compact()

# PyTorch integration
from replayhouse.torch import ReplayIterableDataset
ds = ReplayIterableDataset(t, batch_size=8192, by="advantage",
                           where="env_version >= 12")
```

Errors surface as `ReplayHouseError` subclasses (`SchemaError`,
`BackendError`); SQL injected via `where=`/`by=` is documented as trusted
input (this is a database client, not a sandbox).

## Package layout

```
replayhouse/
  pyproject.toml            # deps: clickhouse-connect, pyarrow; extras: embedded (chdb), torch
  src/replayhouse/
    __init__.py             # ReplayHouse, ReplayTable, SampleBatch
    backend.py              # Backend protocol; ClickHouseBackend, ChdbBackend
    ddl.py                  # create/drop, schema introspection
    sampling.py             # two-phase query builder
    priorities.py           # sidecar writes, compaction
    eviction.py             # fifo / lowest_priority
    torch.py                # ReplayIterableDataset (imports torch lazily)
  tests/                    # pytest, chdb-backed (no infra in CI)
  tests_integration/        # against a ClickHouse server container
  benchmarks/               # see below
  docs/
```

## Testing

- **Unit/functional tests run against chdb** — the entire sampling, priority,
  and eviction behavior is testable in-process with no services, including
  statistical tests (e.g. chi-squared check that sample frequencies track
  weights over many draws).
- **Integration tests** run the same suite against a ClickHouse server
  (docker), plus concurrency tests: parallel inserters + sampler + priority
  updater.
- Property under test everywhere: sampling correctness (weights honored,
  without replacement, filters respected), not performance.

## Benchmark harness (validation gate for future core work)

`benchmarks/` generates synthetic stores at 1M / 100M / 1B rows (fat JSON
payloads, realistic priority skew) and measures:

1. Phase-1 sampling latency vs. store size and sidecar bloat.
2. `update_priorities` throughput; sidecar `argMax` cost vs. lightweight
   `UPDATE` on a single table.
3. End-to-end `sample()` p50/p95 at trainer-realistic cadence.

These numbers are the evidence for (or against) each future core feature:
`SAMPLE k BY <weight>` syntax, per-granule weight statistics for sub-linear
draws, and a `ReplayMergeTree` engine bundling eviction. None of those are
built until a benchmark shows the SQL layer breaking.

## Milestones

1. **M1 — core client**: backends, `create`/`insert`/`sample` (unstratified),
   chdb test suite. Usable end-to-end.
2. **M2 — priorities + eviction**: sidecar updates, `compact`, `evict`,
   stratified sampling, concurrency integration tests.
3. **M3 — training integration**: PyTorch dataset, demo notebook (bandit demo
   on chdb; agent-trajectory demo against a server).
4. **M4 — benchmarks**: harness + published numbers; decision memo on core
   features.
