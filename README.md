# replayhouse

An agentic experience store with native weighted sampling, built on ClickHouse.

ReplayHouse is a thin Python client that turns ClickHouse (a server, or
embedded [chdb](https://github.com/chdb-io/chdb) with zero infrastructure)
into a replay buffer for LLM/agent post-training — in the spirit of DeepMind
Reverb, but designed for agentic workloads:

- **Prioritized sampling as a query**: weighted, without-replacement batch
  draws fused with arbitrary SQL filters and stratification.
- **Append-heavy trajectory ingestion** from many concurrent rollout workers.
- **Batch priority updates** with no mutations in the hot loop.
- **Time- and capacity-based eviction** (FIFO / lowest-priority).
- **One store for training and observability** — the same rows feed the
  trainer, dashboards, and debugging queries.

Status: design phase. See the
[design doc](docs/superpowers/specs/2026-08-06-replayhouse-design.md).

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
```
