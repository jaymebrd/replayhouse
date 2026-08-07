# ReplayHouse Library Follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the three library gaps the examples suite exposed: a public `store.query(sql)`, a typed create-conflict error with `exists_ok`, and seedable (reproducible) sampling.

**Architecture:** All three are small additive API changes. `query` is a thin public delegate to the backend's JSONEachRow path. `create` gains a pre-check against `system.tables` raising a new `TableExistsError` (or returning the existing table with `exists_ok=True`); the check-then-create race is documented as accepted for a client-side convenience. Seedable sampling swaps the A-ES randomness source per draw: `seed=None` keeps `randCanonical()`; an integer seed uses `cityHash64(id, seed)` scaled into (0,1], making draws a pure function of (id set, weights, seed). Tasks 1–2 also update the example code that previously worked around each gap.

**Tech Stack:** Python ≥ 3.10, existing replayhouse internals (`backend.query_rows`, `ddl`, `sampling`), no new dependencies.

## Global Constraints

- Branch `replayhouse-library-followups` off `main`; commit style `feat:`/`fix:`/`docs:`; run `.venv/bin/pytest tests -q` before each commit — the existing 72 tests stay green.
- No new dependencies; no changes to storage schemas or SQL contracts other than the sampling key's randomness source.
- `query(sql)` is documented **trusted input** (same contract as `by=`/`where=`); it must NOT add escaping/validation beyond what the backend does.
- Identifier-validation posture unchanged: every new SQL construction path that interpolates an identifier validates it (`validate_name`); the seed is validated as an `int` (`SchemaError` otherwise) before interpolation.
- Weight semantics unchanged: rows with weight ≤ 0 remain excluded regardless of seed; seeded and unseeded draws are both without replacement.
- The torch `ReplayIterableDataset` does NOT gain a `seed` parameter (a fixed seed would make every batch identical — an easy footgun); `sample`'s docstring notes that a seeded draw is fully deterministic for a given store state.
- Error taxonomy: `TableExistsError` subclasses `SchemaError`; exported from `replayhouse` alongside the other errors.

## File Structure

```
src/replayhouse/errors.py      # + TableExistsError(SchemaError)
src/replayhouse/store.py       # + query(); create() gains exists_ok + pre-check
src/replayhouse/sampling.py    # sample_key/phase1_sql/stratified_sql gain seed (+ id_expr)
src/replayhouse/table.py       # sample() gains seed=None, validates + threads it
src/replayhouse/__init__.py    # export TableExistsError
examples/observability.py      # use store.query(); drop private-access comment
examples/agent/run_agent.py    # use create(..., exists_ok=True); drop bare except
tests/test_store_query.py      # new
tests/test_create_conflict.py  # new
tests/test_seeded_sampling.py  # new
```

---

### Task 1: Public `store.query(sql)`

**Files:**
- Modify: `src/replayhouse/store.py`, `examples/observability.py`
- Test: `tests/test_store_query.py`

**Interfaces:**
- Consumes: `Backend.query_rows(sql) -> list[dict]` (raises `BackendError` on driver failure).
- Produces: `ReplayHouse.query(self, sql: str) -> list[dict]` — public, trusted-input, returns JSONEachRow-parsed rows (64-bit ints arrive as strings, same as everywhere else).

- [ ] **Step 1: Write the failing tests**

`tests/test_store_query.py`:

```python
import pytest

from replayhouse.errors import BackendError
from tests.conftest import make_rows


def test_query_returns_rows(store, table):
    table.insert(make_rows(5))
    rows = store.query("SELECT count() AS c FROM exp")
    assert int(rows[0]["c"]) == 5


def test_query_arbitrary_sql(store, table):
    table.insert(make_rows(10, task_family="a"))
    table.insert(make_rows(20, task_family="b"))
    rows = store.query(
        "SELECT task_family, count() AS c FROM exp GROUP BY task_family ORDER BY c"
    )
    assert [(r["task_family"], int(r["c"])) for r in rows] == [("a", 10), ("b", 20)]


def test_query_bad_sql_raises_backend_error(store):
    with pytest.raises(BackendError):
        store.query("SELECT nonsense FROM nowhere")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_store_query.py -v`
Expected: FAIL with `AttributeError: 'ReplayHouse' object has no attribute 'query'`

- [ ] **Step 3: Implement `query` on `ReplayHouse`**

Add to the `ReplayHouse` class in `src/replayhouse/store.py`:

```python
    def query(self, sql: str) -> list[dict]:
        """Run arbitrary SQL against the store's backend and return rows.

        Trusted input (same contract as sample's by=/where=): this is a
        database client, not a sandbox. 64-bit integers arrive as strings
        (JSONEachRow); coerce with int() as needed.
        """
        return self._backend.query_rows(sql)
```

- [ ] **Step 4: Update `examples/observability.py`**

Replace the private-access lines (the two-line "Demo shortcut" comment and the `store._backend.query_rows(q)` call) with:

```python
            for row in store.query(q):
```

(The `.sql` file remains the Grafana deliverable; no other change.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_store_query.py tests/test_observability_example.py -v`
Expected: all pass (the observability smoke test re-exercises the example end-to-end).

- [ ] **Step 6: Full suite and commit**

Run: `.venv/bin/pytest tests -q` (75 expected)

```bash
git add src/replayhouse/store.py examples/observability.py tests/test_store_query.py
git commit -m "feat: public store.query for raw SQL over the store backend"
```

---

### Task 2: `TableExistsError` + `create(exists_ok=True)`

**Files:**
- Modify: `src/replayhouse/errors.py`, `src/replayhouse/store.py`, `src/replayhouse/__init__.py`, `examples/agent/run_agent.py`
- Test: `tests/test_create_conflict.py`

**Interfaces:**
- Consumes: `ddl.validate_name`, `backend.query_rows`, existing `create` flow (DDL + sidecar + rollback).
- Produces: `class TableExistsError(SchemaError)` with `.table_name` attribute; `ReplayHouse.create(..., exists_ok: bool = False)` — raises `TableExistsError` when the main table already exists, or returns `self.table(name)` when `exists_ok=True`. Exported from `replayhouse`.

- [ ] **Step 1: Write the failing tests**

`tests/test_create_conflict.py`:

```python
import pytest

from replayhouse import TableExistsError
from tests.conftest import make_rows


def test_create_twice_raises_typed_error(store):
    store.create("dup", columns={"x": "UInt32"})
    with pytest.raises(TableExistsError) as exc:
        store.create("dup", columns={"x": "UInt32"})
    assert exc.value.table_name == "dup"


def test_create_exists_ok_returns_usable_table(store):
    t1 = store.create("keep", columns={"x": "UInt32"})
    t1.insert([{"x": 1}])
    t2 = store.create("keep", columns={"x": "UInt32"}, exists_ok=True)
    t2.insert([{"x": 2}])
    assert int(store.query("SELECT count() AS c FROM keep")[0]["c"]) == 2


def test_exists_ok_false_is_default(store):
    store.create("strict", columns={"x": "UInt32"})
    with pytest.raises(TableExistsError):
        store.create("strict", columns={"x": "UInt32"}, exists_ok=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_create_conflict.py -v`
Expected: FAIL with `ImportError: cannot import name 'TableExistsError'`

- [ ] **Step 3: Implement**

`src/replayhouse/errors.py` — add:

```python
class TableExistsError(SchemaError):
    """create() found an existing table with the same name."""

    def __init__(self, table_name: str):
        super().__init__(f"table {table_name!r} already exists "
                         f"(pass exists_ok=True to open it instead)")
        self.table_name = table_name
```

`src/replayhouse/store.py` — at the top of `create(...)` (signature gains `exists_ok: bool = False` as the last keyword), after config validation but before any DDL executes:

```python
        ddl.validate_name(name)
        existing = self._backend.query_rows(
            f"SELECT count() AS c FROM system.tables "
            f"WHERE database = currentDatabase() AND name = '{name}'"
        )
        if int(existing[0]["c"]) > 0:
            if exists_ok:
                return self.table(name)
            raise TableExistsError(name)
        # Note: check-then-create is racy under concurrent creators; the
        # loser gets the backend's own already-exists error, which is fine
        # for a client-side convenience.
```

(with `from .errors import TableExistsError` imported; keep the existing rollback logic unchanged.)

`src/replayhouse/__init__.py` — add `TableExistsError` to the errors import and `__all__`.

- [ ] **Step 4: Update `examples/agent/run_agent.py`**

Replace the try/except create-or-append block:

```python
    store = replayhouse.connect(f"chdb://{args.db}")
    t = store.create("trajectories", columns=COLUMNS, exists_ok=True)
```

(Remove the `try:`/`except Exception:` and the `store.table("trajectories")` fallback entirely.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_create_conflict.py tests/test_agent_example.py tests/test_ddl.py -v`
Expected: all pass — including the existing sidecar-rollback test, which pre-creates only a conflicting `conflicted__priorities` (the main-table pre-check doesn't fire, the sidecar create still fails, rollback still runs).

- [ ] **Step 6: Full suite and commit**

Run: `.venv/bin/pytest tests -q` (78 expected)

```bash
git add src/replayhouse tests/test_create_conflict.py examples/agent/run_agent.py
git commit -m "feat: typed TableExistsError and create(exists_ok=True)"
```

---

### Task 3: Seedable sampling

**Files:**
- Modify: `src/replayhouse/sampling.py`, `src/replayhouse/table.py`
- Test: `tests/test_seeded_sampling.py`

**Interfaces:**
- Consumes: existing `sample_key(by)`, `phase1_sql(name, k, by, where)`, `stratified_sql(name, k, per_group, stratify_by, by, where)`, `ReplayTable.sample(...)`.
- Produces:
  - `sampling.sample_key(by: str, seed: int | None = None, id_expr: str = "id") -> str` — seed `None` keeps `-log(1 - randCanonical()) / (by)`; an int seed yields `-log((cityHash64({id_expr}, {seed}) + 1) / 18446744073709551616.) / (by)` (a value in (0,1], so the log is always defined; the draw becomes a pure function of id, weights, and seed).
  - `phase1_sql(..., seed: int | None = None)` and `stratified_sql(..., seed: int | None = None)` — the join-form branches pass `id_expr="m.id"` (bare `id` is ambiguous in the join); all other branches pass the default `"id"`.
  - `ReplayTable.sample(k, *, by="priority", where=None, stratify_by=None, seed: int | None = None)` — raises `SchemaError` if `seed` is not `None` and not an `int` (`bool` counts as int; accept it); threads seed through both paths.

- [ ] **Step 1: Write the failing tests**

`tests/test_seeded_sampling.py`:

```python
import pytest

from replayhouse.errors import SchemaError
from replayhouse.sampling import phase1_sql, sample_key
from tests.conftest import make_rows


def test_sample_key_seeded_form():
    key = sample_key("priority", seed=42)
    assert "cityHash64(id, 42)" in key and "randCanonical" not in key


def test_sample_key_unseeded_form_unchanged():
    key = sample_key("priority")
    assert "randCanonical()" in key and "cityHash64" not in key


def test_phase1_join_mode_qualifies_id():
    sql = phase1_sql("exp", 10, by="priority * reward", seed=7)
    assert "cityHash64(m.id, 7)" in sql


def test_same_seed_same_batch(table):
    table.insert(make_rows(100))
    a = table.sample(20, seed=42)
    b = table.sample(20, seed=42)
    assert sorted(a.ids) == sorted(b.ids)


def test_different_seeds_differ(table):
    table.insert(make_rows(100))
    a = table.sample(20, seed=1)
    b = table.sample(20, seed=2)
    assert sorted(a.ids) != sorted(b.ids)


def test_seeded_with_filter_and_stratify(table):
    table.insert(make_rows(30, task_family="a", env_version=1))
    table.insert(make_rows(30, task_family="b", env_version=1))
    table.insert(make_rows(30, task_family="c", env_version=2))
    a = table.sample(10, where="env_version = 1", stratify_by="task_family", seed=5)
    b = table.sample(10, where="env_version = 1", stratify_by="task_family", seed=5)
    assert sorted(a.ids) == sorted(b.ids)
    assert {r["task_family"] for r in a.rows} == {"a", "b"}


def test_seeded_weights_still_honored(table):
    table.insert(make_rows(200, task_family="heavy", priority=9.0))
    table.insert(make_rows(200, task_family="light", priority=1.0))
    batch = table.sample(200, seed=3)
    heavy = sum(1 for r in batch.rows if r["task_family"] == "heavy")
    # Same analytic expectation (~164) as the unseeded statistical test;
    # deterministic given the seed, so this can never flake once green.
    assert 130 < heavy < 195


def test_seeded_excludes_nonpositive_weights(table):
    table.insert(make_rows(5, task_family="dead", priority=0.0))
    table.insert(make_rows(5, task_family="live"))
    batch = table.sample(10, seed=11)
    assert len(batch) == 5
    assert {r["task_family"] for r in batch.rows} == {"live"}


def test_non_int_seed_raises(table):
    table.insert(make_rows(5))
    with pytest.raises(SchemaError):
        table.sample(5, seed="42; DROP TABLE exp")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_seeded_sampling.py -v`
Expected: FAIL — `sample_key() got an unexpected keyword argument 'seed'` (and TypeErrors on `sample(seed=...)`).

- [ ] **Step 3: Implement in `sampling.py`**

Replace `sample_key` with:

```python
def sample_key(by: str, seed: int | None = None, id_expr: str = "id") -> str:
    if seed is None:
        u = "1 - randCanonical()"
    else:
        u = f"(cityHash64({id_expr}, {int(seed)}) + 1) / 18446744073709551616."
    return f"-log({u}) / ({by})"
```

Thread `seed: int | None = None` through `phase1_sql` and `stratified_sql`, replacing each `sample_key(...)` call site: branches that already qualify columns with `m.`/`c.` (the INNER JOIN forms in both functions) pass `id_expr="m.id"`; the sidecar-only and main-only branches pass the default. No other SQL changes.

- [ ] **Step 4: Implement in `table.py`**

`sample` signature becomes:

```python
    def sample(self, k: int, *, by: str = "priority", where: str | None = None,
               stratify_by: str | None = None, seed: int | None = None) -> SampleBatch:
        if seed is not None and not isinstance(seed, int):
            raise SchemaError("seed must be an int")
```

and both query-builder calls gain `seed=seed`. Extend the method docstring with one line: "A seeded draw is a pure function of the store's current rows, weights, and the seed — identical until the data changes."

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_seeded_sampling.py tests/test_sampling.py tests/test_stratify.py -v`
Expected: all pass (existing unseeded behavior byte-identical).

- [ ] **Step 6: Full suite and commit**

Run: `.venv/bin/pytest tests -q` (87 expected)

```bash
git add src/replayhouse/sampling.py src/replayhouse/table.py tests/test_seeded_sampling.py
git commit -m "feat: seedable sampling via cityHash64 for reproducible draws"
```

---

## Self-Review Notes

- **Coverage:** all three ledgered follow-ups have tasks; Tasks 1–2 also retire the example workarounds they caused (private `_backend` access; bare `except`). The notebook/README are untouched — `seed`/`query`/`exists_ok` are additive and existing docs stay correct.
- **Type consistency:** `query(sql) -> list[dict]`; `TableExistsError(table_name)` with `.table_name`; `sample_key(by, seed, id_expr)`; `sample(..., seed: int | None = None)` — used identically across tasks.
- **Race honesty:** the create pre-check TOCTOU is documented in-code rather than hidden.
- **Placeholder scan:** clean.
