# replayhouse

Experience replay on ClickHouse.

replayhouse stores experiences — agent trajectories, transitions, scored
rollouts — and samples weighted training batches from them. It is a small
Python client over ClickHouse: a server for scale, or
[chdb](https://github.com/chdb-io/chdb) embedded in-process for local work
with no infrastructure. It borrows the shape of
[Reverb](https://github.com/google-deepmind/reverb) and replaces the replay
server with a database. Because the buffer is a table, a draw can be
filtered with `WHERE`, stratified with `LIMIT BY`, and the exact rows your
trainer consumed are one `SELECT` away from a dashboard.

```python
batch = t.sample(8192, by="priority", where="env_version >= 12")
loss, new_priorities = train_step(batch.rows)
t.update_priorities(batch.ids, new_priorities)
```

**Demo: https://jaymebrd.github.io/replayhouse/** — the full ClickHouse
engine compiled to WebAssembly, running prioritized-replay training loops
in the page. Every number on it comes from a query in your tab.

## Installation

```bash
pip install replayhouse[embedded]      # embedded chdb backend
```

replayhouse is pre-PyPI; install from a checkout today:

```bash
pip install 'replayhouse[embedded] @ git+https://github.com/jaymebrd/replayhouse'
```

## Quick start

```python
import replayhouse

store = replayhouse.connect("chdb:///tmp/replay")   # or "clickhouse://host:8123/db"

t = store.create(
    "agent_experiences",
    columns={
        "task_family": "LowCardinality(String)",
        "env_version": "UInt32",
        "steps": "JSON",
        "reward": "Float32",
    },
    ttl_days=30,
    capacity_rows=10_000_000,
    eviction="lowest_priority",
)

t.insert([{"task_family": "web", "env_version": 1,
           "steps": {"tool_calls": []}, "reward": 0.7, "priority": 2.0}])

batch = t.sample(8192, by="priority", where="env_version >= 1",
                 stratify_by="task_family")
t.update_priorities(batch.ids, [0.5] * len(batch))
t.evict()
```

## How it works

**Tables.** Each buffer is two tables: an append-only `MergeTree` table
holding the experiences, and a `ReplacingMergeTree` sidecar holding
priorities. A priority update is an insert with a monotonic version, not a
mutation; readers resolve last-write-wins at query time. Both directions of
the hot loop — experiences in, priority updates back — are append-only.

**Sampling.** `sample(k, by=...)` is weighted and without replacement,
implemented as an Efraimidis–Spirakis top-k in plain SQL, so it parallelizes
across cores and distributes across shards. `by` accepts any SQL expression
over the row (`"priority"`, `"reward + 0.01"`, `"abs(advantage)"`), `where`
filters the population, `stratify_by` balances groups, and `seed` makes a
draw reproducible. Rows with non-positive weight are never drawn.

**Eviction.** TTL by insertion time, plus capacity policies: `oldest` or
`lowest_priority`, by row count or bytes. `evict()` also removes orphaned
sidecar entries.

**Backends.** `clickhouse://` speaks to a server over HTTP;
`chdb://` runs the engine in-process. The SQL contract is identical — the
test suite runs against both.

Measured on a laptop with embedded chdb: drawing 8,192 rows from a
50-million-row store takes about 1.1 s; updating 8k priorities about
100 ms ([benchmarks](benchmarks/RESULTS.md)).

## Training with PyTorch

```bash
pip install 'replayhouse[embedded,torch]'
```

```python
from replayhouse.torch import ReplayIterableDataset
from torch.utils.data import DataLoader

ds = ReplayIterableDataset(t, batch_size=8192, by="priority",
                           where="env_version >= 12")
for batch in DataLoader(ds, batch_size=None, num_workers=0):
    loss, new_priorities = train_step(batch.rows)
    t.update_priorities(batch.ids, new_priorities)
```

Each item is a whole `SampleBatch` — the store does the batching, so keep
`batch_size=None` and `num_workers=0` in the `DataLoader`. torch is an
optional extra; the core package does not depend on it.

## Examples

All examples run offline against embedded chdb.

![Prioritized replay demo](examples/demo.gif)

The recording is live output: a model trains from the store, per-example
errors flow back as priorities, and pressing `u` switches between
prioritized and uniform sampling. Re-record with `vhs examples/demo.tape`.

- [`examples/demo.py`](examples/demo.py) — the terminal animation above.
- [`examples/agent/`](examples/agent/) — record agent trajectories
  (simulated by default, `--live` runs a real tool-use agent), then curate
  a filtered, stratified, reward-weighted fine-tuning set and export
  Parquet.
- [`examples/grpo_loop.py`](examples/grpo_loop.py) — a GRPO-shaped loop:
  group-relative advantages in, advantage-weighted sampling out.
- [`examples/observability.py`](examples/observability.py) — six
  Grafana-ready queries ([`observability.sql`](examples/observability.sql))
  over the rows the trainer samples.
- [`examples/quickstart.ipynb`](examples/quickstart.ipynb) — the API as a
  notebook.
- [`examples/bandit.py`](examples/bandit.py),
  [`examples/train_reward_model.py`](examples/train_reward_model.py) —
  single-file demos.

## Browser playground

[`web/`](web/) contains the demo site: chdb compiled to WebAssembly, a JS
mirror of the SQL contract, and two training loops that draw every batch
from a `MergeTree` table in the tab. Run it locally with
`npm --prefix web install && npm --prefix web run serve`, deploy with
`web/deploy.sh`, and verify in headless Chrome with
`npm --prefix web run verify:browser`.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev,torch]'
.venv/bin/pytest tests            # full offline suite (chdb)

# integration tests against a real server:
docker run -d --rm --name rh-it -p 18123:8123 \
  -e CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1 clickhouse/clickhouse-server:25.3
REPLAYHOUSE_TEST_URL=clickhouse://localhost:18123/default \
  .venv/bin/pytest tests_integration -m integration; docker stop rh-it
```
