# replayhouse

Experience replay on ClickHouse.

replayhouse stores agent trajectories and samples weighted training batches
from them. It's a small Python client over ClickHouse — a server, or
[chdb](https://github.com/chdb-io/chdb) running in-process, so there is
nothing to deploy to try it. The idea is Reverb with a database where the
replay server used to be: because the buffer is a table, you can filter a
draw with a `WHERE` clause, stratify it with `LIMIT BY`, and point Grafana
at the exact rows your trainer consumed.

```python
batch = t.sample(8192, by="priority", where="env_version >= 12")
loss, new_priorities = train_step(batch.rows)
t.update_priorities(batch.ids, new_priorities)
```

Sampling is weighted and without replacement (an Efraimidis–Spirakis top-k
in plain SQL, so it parallelizes and even distributes across shards).
Priority updates never mutate rows — they're inserts into a small sidecar
table, resolved last-write-wins at read time — so both directions of the
hot loop are append-only. Eviction is TTL plus capacity policies (drop
oldest, or drop lowest-priority). On a laptop, drawing 8192 trajectories
from a 50M-row store takes about 1.1s ([measured](benchmarks/RESULTS.md));
updating 8k priorities takes ~100ms.

Status: early development, pre-PyPI. The
[design doc](docs/superpowers/specs/2026-08-06-replayhouse-design.md) has
the full data model.

## Quick start {#quick-start}

```bash
pip install replayhouse[embedded]      # embedded chdb, zero infrastructure
```

```python
import replayhouse

store = replayhouse.connect("chdb:///tmp/replay")      # or "clickhouse://host:8123/db"

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

## Training with PyTorch {#training-with-pytorch}

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

Each item is a whole `SampleBatch` (the store does the batching) — keep
`batch_size=None` and `num_workers=0` in the `DataLoader`. Runnable demos:
[`examples/bandit.py`](examples/bandit.py) and
[`examples/train_reward_model.py`](examples/train_reward_model.py).
