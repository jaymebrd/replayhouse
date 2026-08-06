# ReplayHouse M3 (PyTorch Integration + Demos) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the `torch` optional extra — a `ReplayIterableDataset` that plugs a `ReplayTable` into a PyTorch training loop — plus two runnable demos (bandit, prioritized reward-model training) and README docs.

**Architecture:** `replayhouse.torch` is a thin adapter: an `IterableDataset` whose items are whole `SampleBatch` objects (the store does the batching, so `DataLoader(ds, batch_size=None)`). Torch is imported only inside `replayhouse.torch` — the core package must never import it. Demos are plain scripts in `examples/`, smoke-tested via subprocess (deliberate deviation from the spec's "demo notebook": scripts are testable and reviewable; a notebook can be generated from them later if wanted).

**Tech Stack:** Python ≥ 3.10, torch ≥ 2.0 (optional extra only), existing replayhouse API (M1-M2, all green at 56 tests).

## Global Constraints

- `torch` is an optional extra (`pip install replayhouse[torch]`), never a core dependency; `import replayhouse` must not import torch (verified by a subprocess test).
- `replayhouse.torch` raises a helpful `ImportError` naming the extra if torch is missing.
- Dataset items are `SampleBatch` objects; document `DataLoader(ds, batch_size=None)` and `num_workers=0` (backend connections are not fork-safe; extra workers would only draw overlapping random batches).
- Empty store / exhausted filter → the iterator STOPS (StopIteration), never spins.
- Tests use `pytest.importorskip("torch")` so torch-less environments stay green; run all tests with `.venv/bin/pytest`.
- Examples run against embedded chdb in a `tempfile.TemporaryDirectory()` — no server, no fixed paths, runnable as `.venv/bin/python examples/<name>.py`.
- Work on a feature branch `replayhouse-m3` off `main`; commit style `feat:`/`test:`/`docs:`; commit after each task.
- Existing suite (56 tests) must stay green throughout.

## File Structure

```
src/replayhouse/torch.py     # ReplayIterableDataset; torch imported here only
tests/test_torch.py          # importorskip-gated dataset tests + lazy-import subprocess test
examples/bandit.py           # 3-arm bandit on chdb, no torch
examples/train_reward_model.py  # prioritized-replay training loop, torch
tests/test_examples.py       # subprocess smoke tests for both examples
pyproject.toml               # + [project.optional-dependencies] torch
README.md                    # + "Training with PyTorch" section
```

---

### Task 1: `torch` extra + `ReplayIterableDataset`

**Files:**
- Modify: `pyproject.toml` (optional-dependencies)
- Create: `src/replayhouse/torch.py`
- Test: `tests/test_torch.py`

**Interfaces:**
- Consumes: `ReplayTable.sample(k, *, by, where, stratify_by) -> SampleBatch`; `SampleBatch` with `.ids`, `.rows`, `__len__`.
- Produces: `class ReplayIterableDataset(torch.utils.data.IterableDataset)` with `__init__(table, batch_size, *, by="priority", where=None, stratify_by=None, num_batches=None)` and `__iter__() -> Iterator[SampleBatch]`. Task 2's training demo relies on exactly this signature.

- [ ] **Step 1: Add the extra and install torch**

In `pyproject.toml`, extend `[project.optional-dependencies]`:

```toml
[project.optional-dependencies]
embedded = ["chdb>=3.0"]
torch = ["torch>=2.0"]
dev = ["pytest>=8", "pandas>=2", "chdb>=3.0"]
```

Run: `cd ~/source/replayhouse && .venv/bin/pip install -e '.[dev,torch]' -q`
Expected: installs torch (large download on first run — normal). Verify: `.venv/bin/python -c "import torch; print(torch.__version__)"` prints ≥ 2.0.

- [ ] **Step 2: Write the failing tests**

`tests/test_torch.py`:

```python
import itertools
import subprocess
import sys

import pytest

from tests.conftest import make_rows

torch = pytest.importorskip("torch")


def test_importing_replayhouse_does_not_import_torch():
    # Lazy-import contract: the core package must not pull torch in.
    code = "import replayhouse, sys; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_dataset_yields_batches_and_stops_at_num_batches(table):
    from replayhouse.torch import ReplayIterableDataset

    table.insert(make_rows(500))
    ds = ReplayIterableDataset(table, 100, num_batches=3)
    batches = list(ds)
    assert len(batches) == 3
    assert all(len(b) == 100 for b in batches)


def test_dataset_infinite_mode_with_islice(table):
    from replayhouse.torch import ReplayIterableDataset

    table.insert(make_rows(50))
    ds = ReplayIterableDataset(table, 10)
    batches = list(itertools.islice(iter(ds), 5))
    assert len(batches) == 5


def test_dataset_stops_on_empty_store(table):
    from replayhouse.torch import ReplayIterableDataset

    assert list(ReplayIterableDataset(table, 10)) == []


def test_dataset_respects_where_and_by(table):
    from replayhouse.torch import ReplayIterableDataset

    table.insert(make_rows(20, env_version=1))
    table.insert(make_rows(20, env_version=2))
    ds = ReplayIterableDataset(table, 50, where="env_version = 2", num_batches=1)
    (batch,) = list(ds)
    assert len(batch) == 20
    assert all(int(r["env_version"]) == 2 for r in batch.rows)


def test_dataset_works_under_dataloader(table):
    from torch.utils.data import DataLoader

    from replayhouse.torch import ReplayIterableDataset

    table.insert(make_rows(200))
    ds = ReplayIterableDataset(table, 64, num_batches=2)
    seen = 0
    for batch in DataLoader(ds, batch_size=None, num_workers=0):
        rewards = torch.tensor([float(r["reward"]) for r in batch.rows])
        assert rewards.shape == (len(batch),)
        seen += 1
    assert seen == 2


def test_dataset_is_iterable_dataset_subclass():
    from torch.utils.data import IterableDataset

    from replayhouse.torch import ReplayIterableDataset

    assert issubclass(ReplayIterableDataset, IterableDataset)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_torch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'replayhouse.torch'` (the subprocess lazy-import test may already pass — fine).

- [ ] **Step 4: Implement `src/replayhouse/torch.py`**

```python
"""PyTorch integration. Requires the extra: pip install 'replayhouse[torch]'."""

from __future__ import annotations

from typing import Iterator

from .table import ReplayTable, SampleBatch

try:
    from torch.utils.data import IterableDataset
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "replayhouse.torch requires PyTorch; install with: pip install 'replayhouse[torch]'"
    ) from e


class ReplayIterableDataset(IterableDataset):
    """Iterates weighted sample batches drawn from a ReplayTable.

    Each item is a whole SampleBatch (the store does the batching), so use
    DataLoader(ds, batch_size=None) with num_workers=0 — backend connections
    are not fork-safe, and extra workers would only draw overlapping random
    batches. Iteration stops when the store yields an empty batch, or after
    num_batches if given.
    """

    def __init__(self, table: ReplayTable, batch_size: int, *,
                 by: str = "priority", where: str | None = None,
                 stratify_by: str | None = None,
                 num_batches: int | None = None):
        self._table = table
        self._batch_size = int(batch_size)
        self._by = by
        self._where = where
        self._stratify_by = stratify_by
        self._num_batches = num_batches

    def __iter__(self) -> Iterator[SampleBatch]:
        produced = 0
        while self._num_batches is None or produced < self._num_batches:
            batch = self._table.sample(
                self._batch_size, by=self._by, where=self._where,
                stratify_by=self._stratify_by,
            )
            if len(batch) == 0:
                return
            yield batch
            produced += 1
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_torch.py tests -v`
Expected: 7 new tests pass; full suite (63) green.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/replayhouse/torch.py tests/test_torch.py
git commit -m "feat: torch extra with ReplayIterableDataset"
```

---

### Task 2: Runnable demos + README section

**Files:**
- Create: `examples/bandit.py`, `examples/train_reward_model.py`
- Test: `tests/test_examples.py`
- Modify: `README.md` (append section)

**Interfaces:**
- Consumes: full public API + `ReplayIterableDataset(table, batch_size, *, by, where, stratify_by, num_batches)` from Task 1 (exact signature).
- Produces: two scripts runnable as `.venv/bin/python examples/<name>.py`, each printing a final summary line the smoke tests grep for (`best arm:` and `final loss:` respectively).

- [ ] **Step 1: Write `examples/bandit.py`**

```python
"""3-arm Thompson-style bandit on embedded chdb. No torch required.

Run: python examples/bandit.py
"""

import random
import tempfile

import replayhouse


def main():
    with tempfile.TemporaryDirectory() as tmp:
        store = replayhouse.connect(f"chdb://{tmp}/db")
        arms = store.create("arms", columns={"arm": "LowCardinality(String)"})

        true_reward = {"slow_tool": 0.2, "fast_tool": 0.5, "smart_tool": 0.9}
        ids = {a: arms.insert([{"arm": a}])[0] for a in true_reward}
        wins = {a: 1.0 for a in true_reward}
        pulls = {a: 1 for a in true_reward}
        rng = random.Random(0)

        for step in range(500):
            arm = arms.sample(1).rows[0]["arm"]
            pulls[arm] += 1
            wins[arm] += 1.0 if rng.random() < true_reward[arm] else 0.0
            arms.update_priorities([ids[arm]], [max(wins[arm] / pulls[arm], 0.01)])
            if (step + 1) % 100 == 0:
                print(f"step {step + 1}: pulls={pulls}")

        best = max(pulls, key=pulls.get)
        print(f"best arm: {best} ({pulls[best]} pulls)")
        store.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `examples/train_reward_model.py`**

```python
"""Prioritized experience replay demo: train a reward model on chdb.

Inserts synthetic experiences, trains a small torch model on weighted
batches, and feeds per-sample |error| back as new priorities — the classic
PER loop, with ClickHouse as the buffer.

Run: python examples/train_reward_model.py   (needs replayhouse[torch])
"""

import random
import tempfile

import torch

import replayhouse
from replayhouse.torch import ReplayIterableDataset


def main():
    rng = random.Random(0)
    with tempfile.TemporaryDirectory() as tmp:
        store = replayhouse.connect(f"chdb://{tmp}/db")
        t = store.create("exp", columns={"x1": "Float32", "x2": "Float32",
                                         "reward": "Float32"})
        rows = []
        for _ in range(2000):
            x1, x2 = rng.uniform(-1, 1), rng.uniform(-1, 1)
            rows.append({"x1": x1, "x2": x2,
                         "reward": 2.0 * x1 - 1.0 * x2 + rng.gauss(0, 0.05)})
        t.insert(rows)

        model = torch.nn.Linear(2, 1)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = torch.nn.MSELoss()

        first = last = None
        ds = ReplayIterableDataset(t, 256, num_batches=40)
        for i, batch in enumerate(ds):
            x = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in batch.rows])
            y = torch.tensor([[float(r["reward"])] for r in batch.rows])
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            # PER: resample hard examples more often.
            errors = (pred - y).abs().squeeze(1).detach()
            t.update_priorities(batch.ids, [max(float(e), 0.01) for e in errors])

            first = first if first is not None else loss.item()
            last = loss.item()
            if (i + 1) % 10 == 0:
                print(f"batch {i + 1}: loss={last:.4f}")

        print(f"final loss: {last:.4f} (first was {first:.4f})")
        store.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write the smoke tests**

`tests/test_examples.py`:

```python
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _run(script):
    return subprocess.run(
        [sys.executable, str(EXAMPLES / script)],
        capture_output=True, text=True, timeout=300, check=True,
    ).stdout


def test_bandit_example_runs_and_converges():
    out = _run("bandit.py")
    assert "best arm: smart_tool" in out


def test_train_reward_model_example_runs():
    pytest.importorskip("torch")
    out = _run("train_reward_model.py")
    assert "final loss:" in out
```

- [ ] **Step 4: Run the smoke tests**

Run: `.venv/bin/pytest tests/test_examples.py -v`
Expected: 2 passed (the bandit assertion on `smart_tool` is robust: 0.9 vs 0.5/0.2 over 500 pulls; if it ever flakes, raise pulls to 1000 — never weaken the assertion).

- [ ] **Step 5: Append README section**

Append to `README.md`:

````markdown
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
````

- [ ] **Step 6: Run the full suite and commit**

Run: `.venv/bin/pytest tests -v`
Expected: all green (65).

```bash
git add examples tests/test_examples.py README.md
git commit -m "feat: bandit and prioritized-training examples with README docs"
```

---

## Self-Review Notes

- **Spec coverage:** M3 = "PyTorch dataset, demo notebook (bandit demo on chdb; agent-trajectory demo against a server)". Dataset ✔ (Task 1). Bandit demo on chdb ✔ (Task 2). Deviations, both deliberate: scripts instead of a notebook (testability), and the training demo runs on chdb rather than against a server — the server path is exercised by the M2 integration suite, and a demo requiring infrastructure isn't runnable documentation. The torch extra stays optional-only per the user's explicit decision.
- **Type consistency:** `ReplayIterableDataset(table, batch_size, *, by, where, stratify_by, num_batches)` identical in Task 1 implementation, Task 1 tests, Task 2 demo, README.
- **Placeholder scan:** clean.
