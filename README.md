# replay-store

An agentic experience store with native weighted sampling, built on ClickHouse.

ReplayStore is a thin Python client that turns ClickHouse (a server, or
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
[design doc](docs/superpowers/specs/2026-08-06-replay-store-design.md).
