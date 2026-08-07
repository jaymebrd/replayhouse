# ReplayHouse Examples Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Four adoption-grade examples: a flagship agent-trajectory pipeline (record → curate → export), a GRPO-shaped training loop, an observability query pack, and an executable quick-start notebook — all runnable offline in minutes, all smoke-tested in CI.

**Architecture:** Every example runs against embedded chdb with no services and fixed seeds. The flagship (`examples/agent/`) has a deterministic simulated agent by default and an optional `--live` mode that runs a real Claude tool-use agent (manual agentic loop — we intercept every step to record the trajectory, and examples avoid beta SDK dependencies); `anthropic` is imported lazily and is never a package dependency. Curation exports Parquet via pyarrow (already a core dep). The GRPO loop uses the `torch` extra. Observability queries live in a `.sql` file (Grafana-ready) executed by a runner script. The notebook is executed in CI via nbclient.

**Tech Stack:** Python ≥ 3.10, replayhouse (main, 65 tests green), pyarrow (core dep), torch ≥ 2.0 (extra, installed in venv), anthropic SDK (optional, live mode only, lazy import), nbformat/nbclient/ipykernel (new dev deps).

## Global Constraints

- Work on branch `replayhouse-examples` off `main`; commit style `feat:`/`test:`/`docs:`; commit after each task; run `.venv/bin/pytest tests -q` before each commit — the existing 65 tests must stay green.
- Examples are **offline-by-default**: embedded chdb under a `tempfile.TemporaryDirectory()` (or a user-supplied `--db` path), deterministic via seeded `random.Random`/`torch.manual_seed`, no network, no API keys. Each prints a final summary line its smoke test greps (exact strings per task).
- `anthropic` must NOT be added to pyproject dependencies or extras; live mode imports it inside the `--live` branch and exits with `pip install anthropic` guidance if missing.
- Live mode: model string exactly `claude-opus-5`; `max_tokens=16000`; no `thinking` parameter (on by default); no `temperature`/`top_p`/`top_k` (rejected on this model); check `stop_reason == "refusal"` before reading content and end the episode gracefully; manual agentic loop with a comment explaining why (per-step trajectory recording, no beta dependency).
- Smoke tests run examples via `subprocess.run([sys.executable, ...], timeout=600, check=True)` and must NOT require torch/anthropic when absent (`pytest.importorskip` for torch-dependent tests; live mode is never exercised in tests).
- New dev deps allowed: `nbformat`, `nbclient`, `ipykernel` (dev extra only).
- The `sample(k, by=...)` weight expression must always be strictly positive for rows that should be samplable (`exp(...)` or `x + <floor>` patterns), matching the engine's `weight > 0` exclusion.
- README gets an "Examples" section (with `{#examples}` anchor) and a "Development" section (with `{#development}` anchor) documenting the test commands including the Docker integration run with `CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1`.

## File Structure

```
examples/agent/__init__.py        # empty (import as package in tests)
examples/agent/toyworld.py        # deterministic toy environment: KB + tools + task generator + answer checker
examples/agent/run_agent.py       # record episodes into ReplayHouse (simulate default, --live optional)
examples/agent/curate.py          # filtered/stratified weighted sample -> Parquet + stats
examples/grpo_loop.py             # group-relative advantage training loop (torch)
examples/observability.sql        # 6 Grafana-ready queries with comment headers
examples/observability.py         # seeds a demo store, runs the .sql queries, prints tables
examples/quickstart.ipynb         # executable quick start (chdb)
tests/test_agent_example.py       # smoke: record -> curate -> parquet assertions
tests/test_grpo_example.py        # smoke: learning-curve assertion (importorskip torch)
tests/test_observability_example.py  # smoke: output headers
tests/test_quickstart_notebook.py # executes the notebook via nbclient
pyproject.toml                    # + nbformat/nbclient/ipykernel in dev extra
README.md                         # + Examples and Development sections
```

---

### Task 1: Flagship — toy world, agent recorder, curation pipeline

**Files:**
- Create: `examples/agent/__init__.py`, `examples/agent/toyworld.py`, `examples/agent/run_agent.py`, `examples/agent/curate.py`
- Test: `tests/test_agent_example.py`

**Interfaces:**
- Consumes: public replayhouse API (`connect`, `create`, `insert`, `sample`, `SampleBatch.to_arrow()`).
- Produces:
  - `toyworld.generate_task(rng) -> Task` where `Task` is a dataclass `(family: str, question: str, answer: str, plan: list[tuple[str, str]])` — `plan` is the correct `(tool, tool_input)` sequence.
  - `toyworld.TOOLS: dict[str, Callable[[str], str]]` — `kb_lookup`, `calculator`.
  - `toyworld.simulate_episode(task, rng) -> dict` — an episode row ready for `insert` (keys: `task_family`, `env_version`, `model`, `steps`, `answer`, `reward`, `total_tokens`).
  - `run_agent.main(argv)` CLI: `--db PATH` (default `./replay_demo_db`), `--episodes N` (default 50), `--seed N` (default 0), `--live` flag. Store name: `trajectories`. Prints `recorded <N> episodes (mean reward <x.xx>)`.
  - `curate.main(argv)` CLI: `--db PATH`, `--out PATH` (default `curated.parquet`), `--k N` (default 200). Prints `exported <N> rows to <path>` last.

- [ ] **Step 1: Write `examples/agent/toyworld.py`**

```python
"""A tiny deterministic tool-use world for recording agent trajectories.

Three task families over a small knowledge base and a calculator:
- lookup:    "What is <entity>?"                      (1 tool call)
- arithmetic:"What is a + b * c?"                     (1 tool call)
- multi_hop: "What is the <relation> of <entity>?"    (2 tool calls)

`simulate_episode` plays a scripted policy with per-family skill levels, so
recorded trajectories contain both successes and realistic failures without
any LLM involved.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

KB = {
    "mercury": "the closest planet to the sun",
    "vostok": "the coldest research station",
    "sequoia": "the tallest tree species",
    "mariana": "the deepest ocean trench",
    "sahara": "the largest hot desert",
    "capital_of_atlantis": "poseidonia",
    "capital_of_lemuria": "shambala",
    "mayor_of_poseidonia": "coral the wise",
    "mayor_of_shambala": "lotus the elder",
}

SKILL = {"lookup": 0.75, "arithmetic": 0.9, "multi_hop": 0.5}
FAMILIES = tuple(SKILL)


def kb_lookup(key: str) -> str:
    return KB.get(key.strip().lower(), "NOT FOUND")


def calculator(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/ ().")
        if not set(expression) <= allowed:
            return "ERROR: invalid characters"
        return str(eval(expression, {"__builtins__": {}}, {}))  # toy world only
    except Exception as e:
        return f"ERROR: {e}"


TOOLS = {"kb_lookup": kb_lookup, "calculator": calculator}


@dataclass
class Task:
    family: str
    question: str
    answer: str
    plan: list  # [(tool_name, tool_input), ...]


def generate_task(rng: random.Random) -> Task:
    family = rng.choice(FAMILIES)
    if family == "lookup":
        key = rng.choice(["mercury", "vostok", "sequoia", "mariana", "sahara"])
        return Task(family, f"What is {key}?", KB[key], [("kb_lookup", key)])
    if family == "arithmetic":
        a, b, c = rng.randint(2, 99), rng.randint(2, 99), rng.randint(2, 9)
        expr = f"{a} + {b} * {c}"
        return Task(family, f"What is {expr}?", str(a + b * c), [("calculator", expr)])
    realm = rng.choice(["atlantis", "lemuria"])
    capital = KB[f"capital_of_{realm}"]
    return Task(
        family,
        f"Who is the mayor of the capital of {realm}?",
        KB[f"mayor_of_{capital}"],
        [("kb_lookup", f"capital_of_{realm}"), ("kb_lookup", f"mayor_of_{capital}")],
    )


def simulate_episode(task: Task, rng: random.Random) -> dict:
    steps, ok = [], rng.random() < SKILL[task.family]
    for i, (tool, tool_input) in enumerate(task.plan):
        if not ok and i == len(task.plan) - 1:
            tool_input = tool_input + "_oops"  # realistic failure: garbled final hop
        output = TOOLS[tool](tool_input)
        steps.append({"tool": tool, "input": tool_input, "output": output})
    answer = steps[-1]["output"] if steps else ""
    reward = 1.0 if answer == task.answer else 0.0
    return {
        "task_family": task.family,
        "env_version": 1,
        "model": "simulated-v0",
        "steps": {"question": task.question, "trace": steps},
        "answer": answer,
        "reward": reward,
        "total_tokens": 40 * len(steps) + rng.randint(10, 60),
    }
```

- [ ] **Step 2: Write `examples/agent/run_agent.py`**

```python
"""Record agent episodes into a ReplayHouse store.

Default: a deterministic simulated agent (no network, no keys).
--live:  a real Claude tool-use agent (requires `pip install anthropic`
         and ANTHROPIC_API_KEY). Same schema either way.

Run: python examples/agent/run_agent.py --episodes 50
"""

from __future__ import annotations

import argparse
import random
import sys

import replayhouse

from . import toyworld

COLUMNS = {
    "task_family": "LowCardinality(String)",
    "env_version": "UInt32",
    "model": "LowCardinality(String)",
    "steps": "JSON",
    "answer": "String",
    "reward": "Float32",
    "total_tokens": "UInt32",
}

TOOL_DEFS = [
    {
        "name": "kb_lookup",
        "description": "Look up an entity in the knowledge base. Call this for any "
                       "question about an entity; input is the entity key.",
        "input_schema": {"type": "object", "properties": {"key": {"type": "string"}},
                          "required": ["key"]},
    },
    {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression. Call this for any "
                       "arithmetic; input is the expression string.",
        "input_schema": {"type": "object",
                          "properties": {"expression": {"type": "string"}},
                          "required": ["expression"]},
    },
]


def run_live_episode(client, task) -> dict:
    # Manual agentic loop rather than the SDK tool runner: we intercept every
    # step to record the trajectory, and examples avoid beta dependencies.
    messages = [{"role": "user", "content": f"{task.question} Use the tools, then "
                 f"reply with ONLY the final answer text."}]
    trace, total_tokens = [], 0
    for _ in range(6):
        response = client.messages.create(
            model="claude-opus-5", max_tokens=16000,
            tools=TOOL_DEFS, messages=messages,
        )
        total_tokens += response.usage.input_tokens + response.usage.output_tokens
        if response.stop_reason == "refusal":
            return _episode(task, trace, "", total_tokens, model=response.model)
        if response.stop_reason != "tool_use":
            text = next((b.text for b in response.content if b.type == "text"), "")
            return _episode(task, trace, text.strip(), total_tokens,
                            model=response.model)
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                arg = next(iter(block.input.values()), "")
                output = toyworld.TOOLS[block.name](str(arg))
                trace.append({"tool": block.name, "input": str(arg), "output": output})
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": output})
        messages.append({"role": "user", "content": results})
    return _episode(task, trace, "", total_tokens, model="claude-opus-5")


def _episode(task, trace, answer, total_tokens, model) -> dict:
    return {
        "task_family": task.family, "env_version": 1, "model": model,
        "steps": {"question": task.question, "trace": trace},
        "answer": answer, "reward": 1.0 if answer == task.answer else 0.0,
        "total_tokens": total_tokens,
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="./replay_demo_db")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--live", action="store_true")
    args = p.parse_args(argv)

    client = None
    if args.live:
        try:
            import anthropic
        except ImportError:
            sys.exit("live mode needs the SDK: pip install anthropic")
        client = anthropic.Anthropic()

    rng = random.Random(args.seed)
    store = replayhouse.connect(f"chdb://{args.db}")
    try:
        t = store.create("trajectories", columns=COLUMNS)
    except Exception:
        t = store.table("trajectories")  # append to an existing demo store

    rows = []
    for _ in range(args.episodes):
        task = toyworld.generate_task(rng)
        row = run_live_episode(client, task) if args.live \
            else toyworld.simulate_episode(task, rng)
        row["priority"] = max(row["reward"], 0.1)  # failures stay sampleable
        rows.append(row)
    t.insert(rows)
    mean = sum(r["reward"] for r in rows) / len(rows)
    print(f"recorded {len(rows)} episodes (mean reward {mean:.2f})")
    store.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write `examples/agent/curate.py`**

```python
"""Build a curated fine-tuning set from recorded trajectories.

The pitch in one query: a weighted, without-replacement draw fused with SQL
filters and stratification, exported straight to Parquet.

Run: python examples/agent/curate.py --db ./replay_demo_db
"""

from __future__ import annotations

import argparse
from collections import Counter

import pyarrow.parquet as pq

import replayhouse


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="./replay_demo_db")
    p.add_argument("--out", default="curated.parquet")
    p.add_argument("--k", type=int, default=200)
    args = p.parse_args(argv)

    store = replayhouse.connect(f"chdb://{args.db}")
    t = store.table("trajectories")

    batch = t.sample(
        args.k,
        by="reward + 0.1",                # weight toward successes, keep failures
        where="env_version >= 1 AND total_tokens < 2000",
        stratify_by="task_family",        # even coverage across families
    )
    fams = Counter(r["task_family"] for r in batch.rows)
    mean = sum(float(r["reward"]) for r in batch.rows) / max(len(batch), 1)
    print(f"sampled {len(batch)} trajectories, mean reward {mean:.2f}")
    for fam, n in sorted(fams.items()):
        print(f"  {fam}: {n}")

    pq.write_table(batch.to_arrow(), args.out)
    print(f"exported {len(batch)} rows to {args.out}")
    store.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write the smoke test**

`tests/test_agent_example.py`:

```python
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def _run(args, cwd):
    return subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True,
                          text=True, timeout=600, check=True).stdout


def test_record_then_curate_roundtrip(tmp_path):
    db = tmp_path / "db"
    out = tmp_path / "curated.parquet"
    rec = _run(["-m", "examples.agent.run_agent", "--db", str(db),
                "--episodes", "60", "--seed", "1"], cwd=ROOT)
    assert "recorded 60 episodes" in rec

    cur = _run(["-m", "examples.agent.curate", "--db", str(db),
                "--out", str(out), "--k", "30"], cwd=ROOT)
    assert "exported 30 rows" in cur
    assert all(f"  {fam}: 10" in cur for fam in ("arithmetic", "lookup", "multi_hop"))

    table = pq.read_table(out)
    assert table.num_rows == 30
    assert {"task_family", "steps", "reward", "answer"} <= set(table.column_names)


def test_simulator_reward_signal_is_mixed(tmp_path):
    rec = _run(["-m", "examples.agent.run_agent", "--db", str(tmp_path / "db2"),
                "--episodes", "80", "--seed", "2"], cwd=ROOT)
    mean = float(rec.split("mean reward ")[1].rstrip(")\n"))
    assert 0.4 < mean < 0.95  # skill levels produce a mixed signal, not all-success
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/pytest tests/test_agent_example.py -v`
Expected: 2 passed. The stratified assertion (10 per family) holds because k=30 over exactly 3 families gives per_group=10 and 60 seeded episodes contain >10 of each family; if a seed produces fewer, bump `--episodes` in the test to 90 — do not weaken the assertion.

- [ ] **Step 6: Full suite and commit**

Run: `.venv/bin/pytest tests -q` (all green, 67 total)

```bash
git add examples/agent tests/test_agent_example.py
git commit -m "feat: flagship agent example - record trajectories, curate to parquet"
```

---

### Task 2: GRPO-shaped training loop

**Files:**
- Create: `examples/grpo_loop.py`
- Test: `tests/test_grpo_example.py`

**Interfaces:**
- Consumes: replayhouse public API; `replayhouse[torch]` venv.
- Produces: CLI `python examples/grpo_loop.py [--rounds N] [--seed N]`; prints `first mean reward: <x>` early and `final mean reward: <x>` last.

- [ ] **Step 1: Write `examples/grpo_loop.py`**

```python
"""A GRPO-shaped loop with ReplayHouse as the rollout buffer.

Shape matches LLM RL practice: G rollouts per prompt, group-relative
advantages, advantage-weighted resampling from the store, policy update,
priorities refreshed from |advantage|. The "policy" is a tiny softmax net
and the "environment" a contextual bandit, so it runs offline in seconds —
swap those two pieces for a model and rollout workers and the plumbing is
unchanged.

Run: python examples/grpo_loop.py   (needs replayhouse[torch])
"""

from __future__ import annotations

import argparse
import random
import tempfile

import torch

import replayhouse

N_PROMPTS, N_ACTIONS, GROUP, DIM = 8, 4, 8, 6


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    contexts = torch.randn(N_PROMPTS, DIM)
    true_w = torch.randn(DIM, N_ACTIONS)
    best = (contexts @ true_w).argmax(dim=1)          # hidden optimal action

    policy = torch.nn.Linear(DIM, N_ACTIONS)
    opt = torch.optim.Adam(policy.parameters(), lr=0.05)

    with tempfile.TemporaryDirectory() as tmp:
        store = replayhouse.connect(f"chdb://{tmp}/db")
        t = store.create("rollouts", columns={
            "prompt_id": "UInt32", "action": "UInt8",
            "reward": "Float32", "advantage": "Float32",
        })

        first = None
        for round_ in range(args.rounds):
            rows, round_rewards = [], []
            for pid in range(N_PROMPTS):
                logits = policy(contexts[pid])
                dist = torch.distributions.Categorical(logits=logits)
                acts = dist.sample((GROUP,))
                rewards = [1.0 if a == best[pid] else 0.1 for a in acts]
                mean_r = sum(rewards) / GROUP
                round_rewards += rewards
                for a, r in zip(acts.tolist(), rewards):
                    rows.append({"prompt_id": pid, "action": a, "reward": r,
                                 "advantage": r - mean_r,
                                 "priority": max(abs(r - mean_r), 0.01)})
            t.insert(rows)

            batch = t.sample(256, by="exp(2 * advantage)")   # favor above-group-mean
            ctx = contexts[[int(r["prompt_id"]) for r in batch.rows]]
            act = torch.tensor([int(r["action"]) for r in batch.rows])
            adv = torch.tensor([float(r["advantage"]) for r in batch.rows])
            logp = torch.distributions.Categorical(
                logits=policy(ctx)).log_prob(act)
            loss = -(logp * adv).mean()
            opt.zero_grad(); loss.backward(); opt.step()

            t.update_priorities(batch.ids,
                                [max(abs(float(r["advantage"])), 0.01)
                                 for r in batch.rows])

            mean_reward = sum(round_rewards) / len(round_rewards)
            if first is None:
                first = mean_reward
                print(f"first mean reward: {first:.3f}")
            if (round_ + 1) % 10 == 0:
                print(f"round {round_ + 1}: mean reward {mean_reward:.3f}")
        print(f"final mean reward: {mean_reward:.3f}")
        store.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the smoke test**

`tests/test_grpo_example.py`:

```python
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_grpo_loop_learns():
    pytest.importorskip("torch")
    out = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "grpo_loop.py"), "--rounds", "30"],
        capture_output=True, text=True, timeout=600, check=True,
    ).stdout
    first = float(out.split("first mean reward: ")[1].split("\n")[0])
    final = float(out.split("final mean reward: ")[1].split("\n")[0])
    assert final > first + 0.15  # policy actually improved via replayed advantages
```

- [ ] **Step 3: Run the test**

Run: `.venv/bin/pytest tests/test_grpo_example.py -v`
Expected: 1 passed. With seed 0, a linear policy on an 8-prompt bandit converges well past +0.15; if it flakes, raise `--rounds` to 50 in the test — never lower the margin.

- [ ] **Step 4: Full suite and commit**

Run: `.venv/bin/pytest tests -q`

```bash
git add examples/grpo_loop.py tests/test_grpo_example.py
git commit -m "feat: GRPO-shaped training loop example with advantage-weighted replay"
```

---

### Task 3: Observability query pack

**Files:**
- Create: `examples/observability.sql`, `examples/observability.py`
- Test: `tests/test_observability_example.py`

**Interfaces:**
- Consumes: replayhouse public API; the flagship's store schema (`trajectories` with `task_family`, `reward`, `total_tokens`, `inserted_at`).
- Produces: CLI `python examples/observability.py [--db PATH]` — seeds a 7-day demo store when `--db` is omitted, executes every query in `observability.sql`, prints each under its comment header; final line `ran 6 queries`.

- [ ] **Step 1: Write `examples/observability.sql`**

```sql
-- Reward by day and task family (Grafana: time series, one line per family)
SELECT toDate(inserted_at) AS day, task_family, round(avg(reward), 3) AS mean_reward
FROM trajectories GROUP BY day, task_family ORDER BY day, task_family;

-- Worst 10 trajectories to eyeball (Grafana: table panel)
SELECT task_family, answer, reward, total_tokens
FROM trajectories ORDER BY reward ASC, total_tokens DESC LIMIT 10;

-- Token spend per family (Grafana: bar gauge)
SELECT task_family, sum(total_tokens) AS tokens, count() AS episodes
FROM trajectories GROUP BY task_family ORDER BY tokens DESC;

-- Success rate overall and last day (Grafana: stat panels)
SELECT round(avg(reward), 3) AS overall,
       round(avgIf(reward, inserted_at >= now() - INTERVAL 1 DAY), 3) AS last_day
FROM trajectories;

-- Current priority distribution (uses the sidecar; Grafana: histogram)
SELECT round(argMax(priority, version), 1) AS p, count() AS n
FROM trajectories__priorities GROUP BY id, p
ORDER BY p;

-- Store size: rows and bytes on disk (Grafana: stat panels)
SELECT sum(rows) AS rows, formatReadableSize(sum(bytes_on_disk)) AS size
FROM system.parts WHERE active AND database = currentDatabase()
  AND `table` = 'trajectories';
```

Note for the implementer: the priority-distribution query as written groups by `id, p` then re-aggregates — that is wrong as a single statement; use this corrected form in the file instead (nested aggregation):

```sql
-- Current priority distribution (uses the sidecar; Grafana: histogram)
SELECT p, count() AS n FROM
(
    SELECT id, round(argMax(priority, version), 1) AS p
    FROM trajectories__priorities GROUP BY id
)
GROUP BY p ORDER BY p;
```

- [ ] **Step 2: Write `examples/observability.py`**

```python
"""One store for training AND observability — the same rows, queried.

Seeds a week of synthetic trajectories (unless --db points at a real store,
e.g. one made by examples/agent/run_agent.py), then runs every query in
observability.sql and prints the results. The .sql file is Grafana-ready:
point a ClickHouse datasource at your store and paste the queries.

Run: python examples/observability.py
"""

from __future__ import annotations

import argparse
import random
import tempfile
from pathlib import Path

import replayhouse

SQL_FILE = Path(__file__).with_name("observability.sql")
FAMILIES = ("browse", "code", "search", "summarize")


def seed(store) -> None:
    rng = random.Random(7)
    t = store.create("trajectories", columns={
        "task_family": "LowCardinality(String)", "env_version": "UInt32",
        "model": "LowCardinality(String)", "steps": "JSON",
        "answer": "String", "reward": "Float32", "total_tokens": "UInt32",
    })
    rows = []
    for day in range(7):
        for fam in FAMILIES:
            drift = 0.04 * day if fam != "browse" else -0.02 * day  # one regressing family
            for _ in range(18):
                r = 1.0 if rng.random() < 0.5 + drift else 0.0
                rows.append({
                    "task_family": fam, "env_version": 1, "model": "demo-v0",
                    "steps": {"trace": []}, "answer": "x", "reward": r,
                    "total_tokens": rng.randint(200, 3000),
                    "inserted_at": f"2026-08-0{day + 1} 12:00:00",
                    "priority": max(r, 0.1),
                })
    t.insert(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=None, help="existing store; default seeds a demo")
    args = p.parse_args(argv)

    queries = [q.strip() for q in SQL_FILE.read_text().split(";") if q.strip()]
    with tempfile.TemporaryDirectory() as tmp:
        store = replayhouse.connect(f"chdb://{args.db or tmp + '/db'}")
        if args.db is None:
            seed(store)
        for q in queries:
            header = q.splitlines()[0].lstrip("- ")
            print(f"\n== {header}")
            for row in store._backend.query_rows(q):
                print("  " + "  ".join(f"{k}={v}" for k, v in row.items()))
        print(f"\nran {len(queries)} queries")
        store.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write the smoke test**

`tests/test_observability_example.py`:

```python
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_observability_runs_all_queries():
    out = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "observability.py")],
        capture_output=True, text=True, timeout=600, check=True,
    ).stdout
    assert "ran 6 queries" in out
    for header in ("Reward by day", "Worst 10 trajectories", "Token spend",
                   "Success rate", "priority distribution", "Store size"):
        assert header in out
```

- [ ] **Step 4: Run the test**

Run: `.venv/bin/pytest tests/test_observability_example.py -v`
Expected: 1 passed.

- [ ] **Step 5: Full suite and commit**

Run: `.venv/bin/pytest tests -q`

```bash
git add examples/observability.sql examples/observability.py tests/test_observability_example.py
git commit -m "feat: observability query pack - Grafana-ready SQL over the same store"
```

---

### Task 4: Quick-start notebook + README sections

**Files:**
- Create: `examples/quickstart.ipynb`
- Modify: `pyproject.toml` (dev extra), `README.md`
- Test: `tests/test_quickstart_notebook.py`

**Interfaces:**
- Consumes: full public API.
- Produces: a notebook executable top-to-bottom on a clean chdb dir; README "Examples" and "Development" sections.

- [ ] **Step 1: Add notebook dev deps and install**

In `pyproject.toml`: `dev = ["pytest>=8", "pandas>=2", "chdb>=3.0", "nbformat>=5", "nbclient>=0.10", "ipykernel>=6"]`

Run: `.venv/bin/pip install -e '.[dev]' -q`

- [ ] **Step 2: Write `examples/quickstart.ipynb`**

Create with nbformat (write this generator inline in a python one-liner file or heredoc, run it once, do NOT commit the generator):

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(
        "# ReplayHouse quick start\n"
        "Experience replay on ClickHouse — running entirely in-process via chdb.\n"
        "```\npip install replayhouse[embedded]\n```"),
    nbf.v4.new_code_cell(
        "import tempfile\nimport replayhouse\n\n"
        "tmp = tempfile.mkdtemp()\n"
        "store = replayhouse.connect(f\"chdb://{tmp}/db\")"),
    nbf.v4.new_code_cell(
        "t = store.create(\n"
        "    \"experiences\",\n"
        "    columns={\"task\": \"LowCardinality(String)\", \"steps\": \"JSON\","
        " \"reward\": \"Float32\"},\n"
        "    capacity_rows=100_000, eviction=\"lowest_priority\",\n"
        ")"),
    nbf.v4.new_code_cell(
        "import random\n"
        "rng = random.Random(0)\n"
        "t.insert([\n"
        "    {\"task\": rng.choice([\"web\", \"code\"]),"
        " \"steps\": {\"n\": i}, \"reward\": rng.random(),\n"
        "     \"priority\": 1.0}\n"
        "    for i in range(1000)\n"
        "])"),
    nbf.v4.new_code_cell(
        "batch = t.sample(64, by=\"reward + 0.05\", stratify_by=\"task\")\n"
        "len(batch), batch.rows[0][\"task\"]"),
    nbf.v4.new_code_cell(
        "t.update_priorities(batch.ids, [0.5] * len(batch))\n"
        "t.compact()\n"
        "t.evict()"),
    nbf.v4.new_code_cell("batch.to_pandas().head()"),
    nbf.v4.new_code_cell("store.close()"),
]
nbf.write(nb, "examples/quickstart.ipynb")
```

- [ ] **Step 3: Write the notebook test**

`tests/test_quickstart_notebook.py`:

```python
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def test_quickstart_notebook_executes(tmp_path):
    nb = nbformat.read(ROOT / "examples" / "quickstart.ipynb", as_version=4)
    client = NotebookClient(nb, timeout=300, kernel_name="python3",
                            resources={"metadata": {"path": str(tmp_path)}})
    client.execute()  # raises CellExecutionError on any failing cell
```

- [ ] **Step 4: Run the test**

Run: `.venv/bin/pytest tests/test_quickstart_notebook.py -v`
Expected: 1 passed.

- [ ] **Step 5: Append README sections**

Append to `README.md`:

````markdown
## Examples {#examples}

All examples run offline against embedded chdb — no server, no keys.

- [`examples/agent/`](examples/agent/) — the full pipeline: record agent
  trajectories (simulated by default; `--live` runs a real Claude tool-use
  agent), then `curate.py` builds a filtered, stratified, reward-weighted
  fine-tuning set and exports Parquet.
- [`examples/grpo_loop.py`](examples/grpo_loop.py) — a GRPO-shaped training
  loop: group-relative advantages in, advantage-weighted sampling out,
  priorities refreshed from `|advantage|`.
- [`examples/observability.py`](examples/observability.py) — the same store
  feeding dashboards: six Grafana-ready queries
  ([`observability.sql`](examples/observability.sql)) over the rows the
  trainer samples.
- [`examples/quickstart.ipynb`](examples/quickstart.ipynb) — the API tour as
  an executable notebook.
- [`examples/bandit.py`](examples/bandit.py) and
  [`examples/train_reward_model.py`](examples/train_reward_model.py) — small
  single-file demos (priority-proportional bandit; prioritized-replay
  training).

## Development {#development}

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev,torch]'
.venv/bin/pytest tests            # full offline suite (chdb)

# integration tests against a real server:
docker run -d --rm --name rh-it -p 18123:8123 \
  -e CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1 clickhouse/clickhouse-server:25.3
REPLAYHOUSE_TEST_URL=clickhouse://localhost:18123/default \
  .venv/bin/pytest tests_integration -m integration; docker stop rh-it
```
````

- [ ] **Step 6: Full suite and commit**

Run: `.venv/bin/pytest tests -q` (expected: all green — 65 baseline + 5 new example tests)

```bash
git add examples/quickstart.ipynb pyproject.toml README.md tests/test_quickstart_notebook.py
git commit -m "feat: executable quickstart notebook; document examples and dev workflow"
```

---

## Self-Review Notes

- **Coverage vs the agreed scope:** flagship record→curate ✔ (Task 1, with live mode), GRPO/TRL-shaped loop ✔ (Task 2), observability SQL ✔ (Task 3), notebook ✔ (Task 4). The "ship a captured live dataset" idea from discussion is deliberately dropped — the simulator produces equivalent structure deterministically, and shipping data blobs in a pip repo is a liability (noted as a deviation, not an oversight).
- **Live-mode code follows the current Claude API surface:** `claude-opus-5`, no sampling params, no `thinking` param, `refusal` handled, parallel tool results returned in a single user message, manual-loop rationale documented.
- **Type consistency:** store name `trajectories` and its schema are identical in Task 1 (creator) and Task 3 (consumer via `--db`); the Task 3 self-seeding path creates the same schema.
- **Placeholder scan:** clean; the one intentionally-corrected SQL query is given in full corrected form.
