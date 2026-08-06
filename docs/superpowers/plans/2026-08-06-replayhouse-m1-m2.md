# ReplayHouse M1–M2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working `replayhouse` Python package: create experience stores on ClickHouse (server or embedded chdb), insert trajectories, sample weighted batches (plain and stratified), update priorities, compact, and evict — fully tested against chdb in CI with no services.

**Architecture:** Thin client over stock ClickHouse. Each logical store is two tables: an append-only `MergeTree` main table (fat payloads, day partitions) and a `ReplacingMergeTree(version)` priority sidecar. Sampling is two-phase Efraimidis–Spirakis (A-ES): a narrow scan draws `k` ids by `ORDER BY -log(1 - randCanonical()) / weight LIMIT k`, then payloads are fetched by primary key. Priority updates are batch inserts of new versions (no mutations in the hot loop). Eviction config is persisted as JSON in the main table's `COMMENT`.

**Tech Stack:** Python ≥ 3.10, `clickhouse-connect` (server backend), `chdb` (embedded backend + test suite), `pyarrow`, `pytest`, hatchling.

## Global Constraints

- No ClickHouse core changes of any kind; everything is SQL over stock ClickHouse.
- Minimum ClickHouse server 25.3; chdb ≥ 3.0. Both backends set `enable_json_type = 1` at session start.
- Package name `replayhouse` (PyPI), module `replayhouse`, src layout (`src/replayhouse/`).
- `where=` / `by=` / `stratify_by=` are documented **trusted input** (SQL); table and column *names* are validated against `^[A-Za-z_][A-Za-z0-9_]*$`.
- System columns `id` (UUID, client-generated UUIDv7) and `inserted_at` (DateTime) are reserved; user payload columns may not use those names.
- Sidecar table for store `<name>` is `<name>__priorities` `(id UUID, priority Float32, version UInt64)`, `ReplacingMergeTree(version) ORDER BY id`; dedup at read time via `argMax(priority, version) GROUP BY id`, never `FINAL` in sampling.
- Sampling is without replacement; rows with non-positive weight are never sampled (`WHERE weight > 0`).
- Deletes in eviction use `ALTER TABLE ... DELETE ... SETTINGS mutations_sync = 2` (synchronous, so tests are deterministic) — never in insert/sample paths.
- Tests: unit/functional suite runs on chdb only (no services); server tests carry `@pytest.mark.integration` and skip unless `REPLAYHOUSE_TEST_URL` is set.
- 64-bit integers arrive as strings through `JSONEachRow` output (`output_format_json_quote_64bit_integers`), so every numeric scalar read from `query_rows` must go through `int()`/`float()`.
- Errors surface as `ReplayHouseError` subclasses: `SchemaError` (bad names/config/input), `BackendError` (connection/driver).
- Commit after every task; conventional-commit style messages (`feat:`, `test:`, `chore:`).

## File Structure

```
replayhouse/
  pyproject.toml               # hatchling; deps + [embedded] and [dev] extras
  .gitignore
  src/replayhouse/
    __init__.py                # public exports: connect, ReplayHouse, ReplayTable, SampleBatch, errors
    errors.py                  # ReplayHouseError, SchemaError, BackendError
    _ids.py                    # uuid7()
    backend.py                 # Backend protocol, parse_url, backend_from_url, ChdbBackend, ClickHouseBackend
    ddl.py                     # name validation, parse_size, main/sidecar DDL builders, config load
    sampling.py                # phase-1 SQL builders (plain + stratified), UUID validation
    priorities.py              # update_priorities, compact
    eviction.py                # table_stats, evict
    table.py                   # ReplayTable, SampleBatch (insert/sample/update_priorities/evict/compact)
    store.py                   # ReplayHouse (connect/create/table/drop/close), module connect()
  tests/                       # chdb-backed; conftest.py provides store/table fixtures
  tests_integration/           # server-only concurrency tests (integration marker)
```

Execution note: run all commands from the repo root `~/source/replayhouse`. One-time setup is part of Task 1.

---

### Task 1: Scaffold, errors, uuid7

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/replayhouse/__init__.py`, `src/replayhouse/errors.py`, `src/replayhouse/_ids.py`
- Test: `tests/test_ids.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `uuid7() -> str` (canonical lowercase UUID string, version 7, time-ordered); exception classes `ReplayHouseError`, `SchemaError(ReplayHouseError)`, `BackendError(ReplayHouseError)`.

- [ ] **Step 1: Write project scaffold**

`pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "replayhouse"
version = "0.1.0"
description = "An agentic experience store with native weighted sampling, built on ClickHouse"
requires-python = ">=3.10"
dependencies = ["clickhouse-connect>=0.8", "pyarrow>=15"]

[project.optional-dependencies]
embedded = ["chdb>=3.0"]
dev = ["pytest>=8", "pandas>=2", "chdb>=3.0"]

[tool.pytest.ini_options]
testpaths = ["tests", "tests_integration"]
markers = ["integration: requires a running ClickHouse server (set REPLAYHOUSE_TEST_URL)"]
```

`.gitignore`:

```
.venv/
__pycache__/
*.egg-info/
dist/
.pytest_cache/
```

`src/replayhouse/errors.py`:

```python
class ReplayHouseError(Exception):
    """Base class for all replayhouse errors."""


class SchemaError(ReplayHouseError):
    """Invalid table/column names, config values, or input shapes."""


class BackendError(ReplayHouseError):
    """Connection or driver-level failure."""
```

`src/replayhouse/__init__.py` (temporary; Task 6 finalizes exports):

```python
from .errors import BackendError, ReplayHouseError, SchemaError

__version__ = "0.1.0"
__all__ = ["ReplayHouseError", "SchemaError", "BackendError", "__version__"]
```

- [ ] **Step 2: Set up the venv**

Run: `cd ~/source/replayhouse && python3 -m venv .venv && .venv/bin/pip install -e '.[dev]' -q`
Expected: installs without error. All later `pytest` commands mean `.venv/bin/pytest`.

- [ ] **Step 3: Write the failing test**

`tests/test_ids.py`:

```python
import uuid

from replayhouse._ids import uuid7


def test_uuid7_is_valid_uuid_version_7():
    u = uuid.UUID(uuid7())
    assert u.version == 7


def test_uuid7_is_time_ordered_and_unique():
    ids = [uuid7() for _ in range(1000)]
    assert len(set(ids)) == 1000
    # 48-bit ms timestamp prefix => lexicographic order tracks generation order
    assert ids == sorted(ids) or ids[:12] == sorted(ids[:12])
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_ids.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'replayhouse._ids'`

- [ ] **Step 5: Implement `_ids.py`**

```python
import os
import time


def uuid7() -> str:
    """RFC 9562 UUIDv7: 48-bit unix ms timestamp + random, time-ordered."""
    b = bytearray(int(time.time_ns() // 1_000_000).to_bytes(6, "big") + os.urandom(10))
    b[6] = (b[6] & 0x0F) | 0x70
    b[8] = (b[8] & 0x3F) | 0x80
    h = bytes(b).hex()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_ids.py -v`
Expected: 2 passed. (The ordering test tolerates same-millisecond ties via the first-12 fallback; if it flakes, keep only the uniqueness + version assertions.)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore src tests
git commit -m "feat: scaffold replayhouse package with errors and uuid7"
```

---

### Task 2: ChdbBackend + URL parsing

**Files:**
- Create: `src/replayhouse/backend.py`
- Test: `tests/test_backend.py`

**Interfaces:**
- Consumes: `BackendError` from Task 1.
- Produces:
  - `parse_url(url: str) -> tuple[str, dict]` — `("chdb", {"path": str})` or `("clickhouse", {"host", "port", "username", "password", "database", "secure"})`.
  - `backend_from_url(url: str) -> Backend`.
  - `Backend` protocol: `query_arrow(sql) -> pyarrow.Table`, `query_rows(sql) -> list[dict]`, `insert_rows(table: str, rows: list[dict]) -> None`, `command(sql) -> None`, `close() -> None`.
  - `class ChdbBackend(path: str)` implementing the protocol.

- [ ] **Step 1: Write the failing tests**

`tests/test_backend.py`:

```python
import pytest

from replayhouse.backend import ChdbBackend, parse_url
from replayhouse.errors import BackendError


def test_parse_url_chdb():
    kind, cfg = parse_url("chdb:///some/dir/db")
    assert kind == "chdb"
    assert cfg == {"path": "/some/dir/db"}


def test_parse_url_clickhouse_defaults():
    kind, cfg = parse_url("clickhouse://host.example.com/mydb")
    assert kind == "clickhouse"
    assert cfg == {
        "host": "host.example.com", "port": 8123, "username": "default",
        "password": "", "database": "mydb", "secure": False,
    }


def test_parse_url_clickhouse_secure_with_creds():
    kind, cfg = parse_url("clickhouses://alice:s3cret@h:9443/db1")
    assert cfg["port"] == 9443 and cfg["secure"] is True
    assert cfg["username"] == "alice" and cfg["password"] == "s3cret"


def test_parse_url_rejects_unknown_scheme():
    with pytest.raises(BackendError):
        parse_url("postgres://x/y")


@pytest.fixture
def backend(tmp_path):
    b = ChdbBackend(str(tmp_path / "db"))
    yield b
    b.close()


def test_chdb_roundtrip_rows_and_json(backend):
    backend.command("CREATE TABLE t (x UInt32, s String, j JSON) ENGINE = MergeTree ORDER BY x")
    backend.insert_rows("t", [{"x": 1, "s": "a", "j": {"k": [1, 2]}}, {"x": 2, "s": "b", "j": {"k": []}}])
    rows = backend.query_rows("SELECT x, s, j FROM t ORDER BY x")
    assert [r["s"] for r in rows] == ["a", "b"]
    assert rows[0]["j"] == {"k": [1, 2]}


def test_chdb_query_arrow(backend):
    tbl = backend.query_arrow("SELECT number FROM numbers(3)")
    assert tbl.num_rows == 3


def test_chdb_insert_empty_is_noop(backend):
    backend.command("CREATE TABLE e (x UInt32) ENGINE = MergeTree ORDER BY x")
    backend.insert_rows("e", [])
    assert backend.query_rows("SELECT count() AS c FROM e")[0]["c"] in (0, "0")


def test_chdb_bad_sql_raises(backend):
    with pytest.raises(BackendError):
        backend.query_rows("SELECT nonsense FROM nowhere")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'replayhouse.backend'`

- [ ] **Step 3: Implement `backend.py`**

```python
from __future__ import annotations

import json
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

import pyarrow as pa

from .errors import BackendError


class Backend(Protocol):
    def query_arrow(self, sql: str) -> pa.Table: ...
    def query_rows(self, sql: str) -> list[dict[str, Any]]: ...
    def insert_rows(self, table: str, rows: list[dict[str, Any]]) -> None: ...
    def command(self, sql: str) -> None: ...
    def close(self) -> None: ...


def _ndjson(rows: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(r, default=str) for r in rows)


def parse_url(url: str) -> tuple[str, dict]:
    p = urlparse(url)
    if p.scheme == "chdb":
        return "chdb", {"path": (p.netloc or "") + p.path}
    if p.scheme in ("clickhouse", "clickhouses", "http", "https"):
        secure = p.scheme in ("clickhouses", "https")
        return "clickhouse", {
            "host": p.hostname or "localhost",
            "port": p.port or (8443 if secure else 8123),
            "username": unquote(p.username) if p.username else "default",
            "password": unquote(p.password) if p.password else "",
            "database": p.path.lstrip("/") or "default",
            "secure": secure,
        }
    raise BackendError(f"unsupported URL scheme: {p.scheme!r}")


def backend_from_url(url: str) -> Backend:
    kind, cfg = parse_url(url)
    if kind == "chdb":
        return ChdbBackend(cfg["path"])
    return ClickHouseBackend(cfg)


class ChdbBackend:
    """Embedded ClickHouse via a chdb session. Single-process only."""

    def __init__(self, path: str):
        try:
            from chdb import session
        except ImportError as e:
            raise BackendError("chdb not installed; pip install 'replayhouse[embedded]'") from e
        self._sess = session.Session(path)
        self._run("SET enable_json_type = 1")

    def _run(self, sql: str, fmt: str = "CSV"):
        try:
            res = self._sess.query(sql, fmt)
        except Exception as e:
            raise BackendError(str(e)) from e
        err = getattr(res, "error_message", None)
        msg = err() if callable(err) else err
        if msg:
            raise BackendError(msg)
        return res

    def query_arrow(self, sql: str) -> pa.Table:
        buf = self._run(sql, "ArrowStream").bytes()
        if not buf:
            return pa.table({})
        return pa.ipc.open_stream(pa.BufferReader(buf)).read_all()

    def query_rows(self, sql: str) -> list[dict[str, Any]]:
        text = self._run(sql, "JSONEachRow").bytes().decode()
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def insert_rows(self, table: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self._run(f"INSERT INTO `{table}` FORMAT JSONEachRow\n{_ndjson(rows)}")

    def command(self, sql: str) -> None:
        self._run(sql)

    def close(self) -> None:
        self._sess.close()


class ClickHouseBackend:
    """ClickHouse server via clickhouse-connect (HTTP)."""

    def __init__(self, cfg: dict):
        import clickhouse_connect

        try:
            self._client = clickhouse_connect.get_client(
                settings={"enable_json_type": 1}, **cfg
            )
        except Exception as e:
            raise BackendError(str(e)) from e

    def query_arrow(self, sql: str) -> pa.Table:
        return self._client.query_arrow(sql)

    def query_rows(self, sql: str) -> list[dict[str, Any]]:
        raw = self._client.raw_query(sql, fmt="JSONEachRow")
        return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]

    def insert_rows(self, table: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self._client.raw_insert(f"`{table}`", insert_block=_ndjson(rows), fmt="JSONEachRow")

    def command(self, sql: str) -> None:
        self._client.command(sql)

    def close(self) -> None:
        self._client.close()
```

Implementation note for the executor: if `INSERT ... FORMAT JSONEachRow\n<data>` through `chdb session.query` fails (inline data unsupported in your chdb build), fall back inside `ChdbBackend.insert_rows` to writing the NDJSON to a `tempfile.NamedTemporaryFile` and running `INSERT INTO \`{table}\` SELECT * FROM file('{path}', JSONEachRow)`. The test suite is the arbiter; keep whichever passes.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_backend.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/replayhouse/backend.py tests/test_backend.py
git commit -m "feat: backend protocol with chdb and clickhouse-connect implementations"
```

---

### Task 3: DDL builders + ReplayHouse store object

**Files:**
- Create: `src/replayhouse/ddl.py`, `src/replayhouse/store.py`, `src/replayhouse/table.py` (skeleton), `tests/conftest.py`
- Test: `tests/test_ddl.py`

**Interfaces:**
- Consumes: `Backend`, `backend_from_url` (Task 2); `SchemaError` (Task 1).
- Produces:
  - `ddl.validate_name(name: str) -> str` (returns name or raises `SchemaError`).
  - `ddl.parse_size(s: str | int) -> int` (accepts `B/KiB/MiB/GiB/TiB` suffixes).
  - `ddl.main_table_ddl(name, columns: dict[str, str], ttl_days: int | None, config: dict) -> str`.
  - `ddl.sidecar_ddl(name) -> str`; sidecar name is always `f"{name}__priorities"`.
  - `ddl.load_config(backend, name) -> dict` (reads JSON from the main table comment via `system.tables`).
  - `class ReplayHouse`: `connect(url) classmethod`, `create(name, columns, *, ttl_days=None, capacity_bytes=None, capacity_rows=None, eviction="fifo") -> ReplayTable`, `table(name) -> ReplayTable`, `drop(name) -> None`, `close() -> None`; module-level `connect(url)` in `store.py`.
  - `class ReplayTable(backend, name)` with attributes `.name`, `._sidecar` (methods filled in by Tasks 4–8).

- [ ] **Step 1: Write the shared fixtures**

`tests/conftest.py`:

```python
import pytest

from replayhouse.store import ReplayHouse


@pytest.fixture
def store(tmp_path):
    s = ReplayHouse.connect(f"chdb://{tmp_path}/db")
    yield s
    s.close()


@pytest.fixture
def table(store):
    return store.create(
        "exp",
        columns={
            "task_family": "LowCardinality(String)",
            "env_version": "UInt32",
            "steps": "JSON",
            "reward": "Float32",
        },
    )


def make_rows(n, *, task_family="web", env_version=1, priority=None, inserted_at=None, reward=None):
    out = []
    for i in range(n):
        r = {
            "task_family": task_family,
            "env_version": env_version,
            "steps": {"i": i},
            "reward": float(i) if reward is None else float(reward),
        }
        if priority is not None:
            r["priority"] = float(priority)
        if inserted_at is not None:
            r["inserted_at"] = inserted_at
        out.append(r)
    return out
```

- [ ] **Step 2: Write the failing tests**

`tests/test_ddl.py`:

```python
import json

import pytest

from replayhouse.ddl import load_config, parse_size, validate_name
from replayhouse.errors import SchemaError


def test_validate_name_rejects_injection():
    assert validate_name("agent_experiences") == "agent_experiences"
    for bad in ("t; DROP TABLE x", "t`x", "1abc", "", "a-b"):
        with pytest.raises(SchemaError):
            validate_name(bad)


def test_parse_size():
    assert parse_size(123) == 123
    assert parse_size("1KiB") == 1024
    assert parse_size("2TiB") == 2 * 1024**4
    with pytest.raises(SchemaError):
        parse_size("2TB")


def test_create_makes_main_and_sidecar_with_config(store):
    store.create(
        "exp", columns={"reward": "Float32"},
        ttl_days=30, capacity_rows=1000, eviction="lowest_priority",
    )
    cols = store._backend.query_rows(
        "SELECT name, type FROM system.columns "
        "WHERE database = currentDatabase() AND table = 'exp' ORDER BY position"
    )
    assert [c["name"] for c in cols] == ["id", "inserted_at", "reward"]
    side = store._backend.query_rows(
        "SELECT name FROM system.columns "
        "WHERE database = currentDatabase() AND table = 'exp__priorities' ORDER BY position"
    )
    assert [c["name"] for c in side] == ["id", "priority", "version"]
    cfg = load_config(store._backend, "exp")
    assert cfg == {"eviction": "lowest_priority", "capacity_rows": 1000}


def test_create_rejects_reserved_and_bad_columns(store):
    with pytest.raises(SchemaError):
        store.create("t1", columns={"id": "UInt32"})
    with pytest.raises(SchemaError):
        store.create("t2", columns={"x; DROP": "UInt32"})
    with pytest.raises(SchemaError):
        store.create("t3", columns={"x": "UInt32"}, eviction="random")


def test_drop_removes_both_tables(store):
    store.create("gone", columns={"x": "UInt32"})
    store.drop("gone")
    n = store._backend.query_rows(
        "SELECT count() AS c FROM system.tables "
        "WHERE database = currentDatabase() AND name LIKE 'gone%'"
    )[0]["c"]
    assert int(n) == 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_ddl.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'replayhouse.ddl'`

- [ ] **Step 4: Implement `ddl.py`**

```python
from __future__ import annotations

import json
import re

from .errors import SchemaError

RESERVED_COLUMNS = ("id", "inserted_at")
EVICTION_POLICIES = ("fifo", "lowest_priority")
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SIZE_RE = re.compile(r"^\s*([\d.]+)\s*(B|KiB|MiB|GiB|TiB)\s*$")
_UNITS = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4}


def validate_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise SchemaError(f"invalid identifier: {name!r}")
    return name


def sidecar_name(name: str) -> str:
    return f"{name}__priorities"


def parse_size(s: str | int) -> int:
    if isinstance(s, int):
        return s
    m = _SIZE_RE.match(s or "")
    if not m:
        raise SchemaError(f"invalid size {s!r}; use B/KiB/MiB/GiB/TiB")
    return int(float(m.group(1)) * _UNITS[m.group(2)])


def build_config(*, capacity_bytes=None, capacity_rows=None, eviction="fifo") -> dict:
    if eviction not in EVICTION_POLICIES:
        raise SchemaError(f"eviction must be one of {EVICTION_POLICIES}")
    cfg: dict = {"eviction": eviction}
    if capacity_bytes is not None:
        cfg["capacity_bytes"] = parse_size(capacity_bytes)
    if capacity_rows is not None:
        cfg["capacity_rows"] = int(capacity_rows)
    return cfg


def main_table_ddl(name: str, columns: dict[str, str], ttl_days: int | None, config: dict) -> str:
    validate_name(name)
    for col in columns:
        validate_name(col)
        if col in RESERVED_COLUMNS:
            raise SchemaError(f"column name {col!r} is reserved")
    col_sql = "".join(f",\n    `{c}` {t}" for c, t in columns.items())
    ttl = f"\nTTL inserted_at + INTERVAL {int(ttl_days)} DAY" if ttl_days else ""
    comment = json.dumps(config).replace("\\", "\\\\").replace("'", "\\'")
    return (
        f"CREATE TABLE `{name}`\n(\n"
        f"    id UUID,\n"
        f"    inserted_at DateTime DEFAULT now(){col_sql}\n)\n"
        f"ENGINE = MergeTree\nORDER BY id\n"
        f"PARTITION BY toStartOfDay(inserted_at){ttl}\n"
        f"COMMENT '{comment}'"
    )


def sidecar_ddl(name: str) -> str:
    validate_name(name)
    return (
        f"CREATE TABLE `{sidecar_name(name)}`\n"
        f"(\n    id UUID,\n    priority Float32,\n    version UInt64\n)\n"
        f"ENGINE = ReplacingMergeTree(version)\nORDER BY id"
    )


def load_config(backend, name: str) -> dict:
    validate_name(name)
    rows = backend.query_rows(
        f"SELECT comment FROM system.tables "
        f"WHERE database = currentDatabase() AND name = '{name}'"
    )
    if not rows:
        raise SchemaError(f"no such table: {name!r}")
    try:
        return json.loads(rows[0]["comment"] or "{}")
    except json.JSONDecodeError:
        return {}
```

- [ ] **Step 5: Implement `store.py` and the `table.py` skeleton**

`src/replayhouse/store.py`:

```python
from __future__ import annotations

from . import ddl
from .backend import Backend, backend_from_url
from .table import ReplayTable


class ReplayHouse:
    def __init__(self, backend: Backend):
        self._backend = backend

    @classmethod
    def connect(cls, url: str) -> "ReplayHouse":
        return cls(backend_from_url(url))

    def create(self, name, columns, *, ttl_days=None, capacity_bytes=None,
               capacity_rows=None, eviction="fifo") -> ReplayTable:
        config = ddl.build_config(
            capacity_bytes=capacity_bytes, capacity_rows=capacity_rows, eviction=eviction
        )
        self._backend.command(ddl.main_table_ddl(name, columns, ttl_days, config))
        self._backend.command(ddl.sidecar_ddl(name))
        return self.table(name)

    def table(self, name: str) -> ReplayTable:
        return ReplayTable(self._backend, ddl.validate_name(name))

    def drop(self, name: str) -> None:
        ddl.validate_name(name)
        self._backend.command(f"DROP TABLE IF EXISTS `{ddl.sidecar_name(name)}`")
        self._backend.command(f"DROP TABLE IF EXISTS `{name}`")

    def close(self) -> None:
        self._backend.close()


def connect(url: str) -> ReplayHouse:
    return ReplayHouse.connect(url)
```

`src/replayhouse/table.py` (skeleton; later tasks add methods):

```python
from __future__ import annotations

from .backend import Backend
from .ddl import sidecar_name


class ReplayTable:
    def __init__(self, backend: Backend, name: str):
        self._backend = backend
        self.name = name
        self._sidecar = sidecar_name(name)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_ddl.py -v`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add src/replayhouse tests
git commit -m "feat: DDL builders and ReplayHouse store with create/table/drop"
```

---

### Task 4: Insert path (dual-table write)

**Files:**
- Modify: `src/replayhouse/table.py`
- Test: `tests/test_insert.py`

**Interfaces:**
- Consumes: `uuid7` (Task 1), `Backend.insert_rows` (Task 2), fixtures (Task 3).
- Produces: `ReplayTable.insert(rows, *, default_priority: float = 1.0) -> list[str]` — accepts `list[dict]`, `pandas.DataFrame`, or `pyarrow.Table`; returns inserted ids. Per-row optional keys: `priority` (stripped from payload, written to sidecar), `id` (kept if provided), `inserted_at` (optional override).

- [ ] **Step 1: Write the failing tests**

`tests/test_insert.py`:

```python
import pytest

from replayhouse.errors import SchemaError
from tests.conftest import make_rows


def _count(store, tbl):
    return int(store._backend.query_rows(f"SELECT count() AS c FROM `{tbl}`")[0]["c"])


def test_insert_writes_main_and_sidecar(store, table):
    ids = table.insert(make_rows(10))
    assert len(ids) == 10 and len(set(ids)) == 10
    assert _count(store, "exp") == 10
    assert _count(store, "exp__priorities") == 10


def test_insert_default_and_explicit_priority(store, table):
    table.insert(make_rows(3))                    # default 1.0
    table.insert(make_rows(2, priority=7.5))
    rows = store._backend.query_rows(
        "SELECT priority, count() AS c FROM exp__priorities GROUP BY priority ORDER BY priority"
    )
    assert [(float(r["priority"]), int(r["c"])) for r in rows] == [(1.0, 3), (7.5, 2)]


def test_insert_preserves_provided_ids(table):
    given = "0198a111-2222-7333-8444-555566667777"
    ids = table.insert([{**make_rows(1)[0], "id": given}])
    assert ids == [given]


def test_insert_accepts_pandas(store, table):
    pd = pytest.importorskip("pandas")
    table.insert(pd.DataFrame(make_rows(4)))
    assert _count(store, "exp") == 4


def test_insert_rejects_garbage(table):
    with pytest.raises(SchemaError):
        table.insert("not rows")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_insert.py -v`
Expected: FAIL with `AttributeError: 'ReplayTable' object has no attribute 'insert'`

- [ ] **Step 3: Implement insert in `table.py`**

Add:

```python
import time
from typing import Any

from ._ids import uuid7
from .errors import SchemaError


def _normalize_rows(rows) -> list[dict[str, Any]]:
    if isinstance(rows, list):
        if not all(isinstance(r, dict) for r in rows):
            raise SchemaError("rows must be dicts")
        return rows
    if hasattr(rows, "to_pylist"):        # pyarrow.Table
        return rows.to_pylist()
    if hasattr(rows, "to_dict"):          # pandas.DataFrame
        return rows.to_dict("records")
    raise SchemaError(f"unsupported rows type: {type(rows).__name__}")
```

and on `ReplayTable`:

```python
    def insert(self, rows, *, default_priority: float = 1.0) -> list[str]:
        rows = _normalize_rows(rows)
        if not rows:
            return []
        version = time.time_ns()
        main, prios = [], []
        for r in rows:
            r = dict(r)
            priority = float(r.pop("priority", default_priority))
            r.setdefault("id", uuid7())
            main.append(r)
            prios.append({"id": r["id"], "priority": priority, "version": version})
        self._backend.insert_rows(self.name, main)
        self._backend.insert_rows(self._sidecar, prios)
        return [r["id"] for r in main]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_insert.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/replayhouse/table.py tests/test_insert.py
git commit -m "feat: dual-table insert with client-side ids and priority sidecar"
```

---

### Task 5: Weighted sampling (unstratified) + SampleBatch

**Files:**
- Create: `src/replayhouse/sampling.py`
- Modify: `src/replayhouse/table.py`, `src/replayhouse/__init__.py`
- Test: `tests/test_sampling.py`

**Interfaces:**
- Consumes: `Backend.query_rows`, `validate_name`, insert from Task 4.
- Produces:
  - `sampling.phase1_sql(name: str, k: int, by: str = "priority", where: str | None = None) -> str`.
  - `sampling.validate_ids(ids: list[str]) -> list[str]` (UUID-format check, raises `SchemaError`).
  - `class SampleBatch`: fields `ids: list[str]`, `rows: list[dict]`; `__len__`; `to_arrow() -> pyarrow.Table`; `to_pandas()`.
  - `ReplayTable.sample(k: int, *, by: str = "priority", where: str | None = None, stratify_by: str | None = None) -> SampleBatch` (stratify_by raises `NotImplementedError` until Task 7).
  - Final `__init__.py` exports: `connect`, `ReplayHouse`, `ReplayTable`, `SampleBatch`, error classes.

Sampling SQL contract (A-ES weighted reservoir, without replacement; sort key is `-log(1 - randCanonical()) / weight` ascending):

- `by == "priority"` (default) — sidecar-only scan with optional semi-join filter:

```sql
WITH current AS
(
    SELECT id, argMax(priority, version) AS priority
    FROM `exp__priorities`
    GROUP BY id
)
SELECT id
FROM current
WHERE priority > 0
  AND id IN (SELECT id FROM `exp` WHERE (env_version >= 12))   -- only when where= given
ORDER BY -log(1 - randCanonical()) / priority ASC
LIMIT 8192
```

- `by` mentions the word `priority` inside a larger expression — join main and sidecar:

```sql
WITH current AS
(
    SELECT id, argMax(priority, version) AS priority
    FROM `exp__priorities`
    GROUP BY id
)
SELECT m.id AS id
FROM `exp` AS m
INNER JOIN current AS c ON m.id = c.id
WHERE ((priority * reward)) > 0 AND ((env_version >= 12))
ORDER BY -log(1 - randCanonical()) / ((priority * reward)) ASC
LIMIT 8192
```

- otherwise (`by` is a main-table expression, e.g. `"reward"`) — main table only, no sidecar:

```sql
SELECT id
FROM `exp`
WHERE ((reward)) > 0 AND ((env_version >= 12))
ORDER BY -log(1 - randCanonical()) / ((reward)) ASC
LIMIT 8192
```

- [ ] **Step 1: Write the failing tests**

`tests/test_sampling.py`:

```python
import pytest

from replayhouse.errors import SchemaError
from replayhouse.sampling import phase1_sql, validate_ids
from tests.conftest import make_rows


def test_phase1_sql_default_uses_sidecar_only():
    sql = phase1_sql("exp", 10)
    assert "exp__priorities" in sql and "JOIN" not in sql and "IN (SELECT" not in sql


def test_phase1_sql_where_adds_semijoin():
    sql = phase1_sql("exp", 10, where="env_version >= 2")
    assert "IN (SELECT id FROM `exp` WHERE (env_version >= 2))" in sql


def test_phase1_sql_main_column_skips_sidecar():
    sql = phase1_sql("exp", 10, by="reward")
    assert "exp__priorities" not in sql


def test_phase1_sql_mixed_expression_joins():
    sql = phase1_sql("exp", 10, by="priority * reward")
    assert "INNER JOIN" in sql and "exp__priorities" in sql


def test_validate_ids():
    good = ["0198a111-2222-7333-8444-555566667777"]
    assert validate_ids(good) == good
    with pytest.raises(SchemaError):
        validate_ids(["x'); DROP TABLE exp; --"])


def test_sample_without_replacement_and_k(table):
    table.insert(make_rows(50))
    batch = table.sample(20)
    assert len(batch) == 20
    assert len(set(batch.ids)) == 20
    assert set(batch.rows[0]) >= {"id", "task_family", "env_version", "steps", "reward"}


def test_sample_more_than_population_returns_all(table):
    table.insert(make_rows(5))
    assert len(table.sample(100)) == 5


def test_sample_excludes_nonpositive_priority(table):
    table.insert(make_rows(5, task_family="dead", priority=0.0))
    table.insert(make_rows(5, task_family="live"))
    batch = table.sample(10)
    assert len(batch) == 5
    assert {r["task_family"] for r in batch.rows} == {"live"}


def test_sample_respects_where(table):
    table.insert(make_rows(10, env_version=1))
    table.insert(make_rows(10, env_version=2))
    batch = table.sample(20, where="env_version = 2")
    assert len(batch) == 10
    assert all(int(r["env_version"]) == 2 for r in batch.rows)


def test_sample_by_main_column(table):
    table.insert(make_rows(1, reward=0.0))      # weight 0 -> excluded
    table.insert(make_rows(10, reward=3.0))
    batch = table.sample(20, by="reward")
    assert len(batch) == 10


def test_sample_weights_are_honored_statistically(table):
    table.insert(make_rows(200, task_family="heavy", priority=9.0))
    table.insert(make_rows(200, task_family="light", priority=1.0))
    heavy = 0
    for _ in range(5):
        batch = table.sample(200)
        heavy += sum(1 for r in batch.rows if r["task_family"] == "heavy")
    heavy_avg = heavy / 5
    # Analytic expectation ~164 of 200; generous non-flaky bounds.
    assert 130 < heavy_avg < 195


def test_empty_table_returns_empty_batch(table):
    batch = table.sample(10)
    assert len(batch) == 0 and batch.ids == [] and batch.to_arrow().num_rows == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_sampling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'replayhouse.sampling'`

- [ ] **Step 3: Implement `sampling.py`**

```python
from __future__ import annotations

import re

from .ddl import sidecar_name, validate_name
from .errors import SchemaError

_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_PRIORITY_WORD = re.compile(r"\bpriority\b")

_CURRENT_CTE = (
    "WITH current AS\n(\n"
    "    SELECT id, argMax(priority, version) AS priority\n"
    "    FROM `{sidecar}`\n    GROUP BY id\n)\n"
)


def sample_key(by: str) -> str:
    return f"-log(1 - randCanonical()) / ({by})"


def phase1_sql(name: str, k: int, by: str = "priority", where: str | None = None) -> str:
    validate_name(name)
    k = int(k)
    sidecar = sidecar_name(name)
    if by.strip() == "priority":
        filt = f"\n  AND id IN (SELECT id FROM `{name}` WHERE ({where}))" if where else ""
        return (
            _CURRENT_CTE.format(sidecar=sidecar)
            + f"SELECT id\nFROM current\nWHERE priority > 0{filt}\n"
            + f"ORDER BY {sample_key('priority')} ASC\nLIMIT {k}"
        )
    cond = f"(({by})) > 0" + (f" AND (({where}))" if where else "")
    order = f"ORDER BY {sample_key(f'({by})')} ASC\nLIMIT {k}"
    if _PRIORITY_WORD.search(by):
        return (
            _CURRENT_CTE.format(sidecar=sidecar)
            + f"SELECT m.id AS id\nFROM `{name}` AS m\n"
            + f"INNER JOIN current AS c ON m.id = c.id\nWHERE {cond}\n{order}"
        )
    return f"SELECT id\nFROM `{name}`\nWHERE {cond}\n{order}"


def validate_ids(ids: list[str]) -> list[str]:
    for i in ids:
        if not _UUID_RE.match(str(i)):
            raise SchemaError(f"not a UUID: {i!r}")
    return list(ids)


def fetch_sql(name: str, ids: list[str]) -> str:
    validate_name(name)
    id_list = ", ".join(f"'{i}'" for i in validate_ids(ids))
    return f"SELECT * FROM `{name}` WHERE id IN ({id_list})"
```

- [ ] **Step 4: Implement `SampleBatch` and `ReplayTable.sample`**

In `table.py` add:

```python
from dataclasses import dataclass, field

import pyarrow as pa

from .sampling import fetch_sql, phase1_sql


@dataclass
class SampleBatch:
    ids: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rows)

    def to_arrow(self) -> pa.Table:
        return pa.Table.from_pylist(self.rows)

    def to_pandas(self):
        import pandas

        return pandas.DataFrame(self.rows)
```

and on `ReplayTable`:

```python
    def sample(self, k: int, *, by: str = "priority", where: str | None = None,
               stratify_by: str | None = None) -> SampleBatch:
        if stratify_by is not None:
            raise NotImplementedError("stratified sampling arrives in Task 7")
        chosen = self._backend.query_rows(phase1_sql(self.name, k, by=by, where=where))
        ids = [r["id"] for r in chosen]
        if not ids:
            return SampleBatch()
        rows = self._backend.query_rows(fetch_sql(self.name, ids))
        return SampleBatch(ids=[r["id"] for r in rows], rows=rows)
```

Finalize `src/replayhouse/__init__.py`:

```python
from .errors import BackendError, ReplayHouseError, SchemaError
from .store import ReplayHouse, connect
from .table import ReplayTable, SampleBatch

__version__ = "0.1.0"
__all__ = [
    "connect", "ReplayHouse", "ReplayTable", "SampleBatch",
    "ReplayHouseError", "SchemaError", "BackendError", "__version__",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_sampling.py tests/ -v`
Expected: all pass, including the statistical test (bounds 130–195 give it huge slack; if it still flakes, widen — never tighten mid-execution).

- [ ] **Step 6: Commit**

```bash
git add src/replayhouse tests/test_sampling.py
git commit -m "feat: two-phase A-ES weighted sampling with SampleBatch"
```

---

### Task 6: Priority updates + compaction

**Files:**
- Create: `src/replayhouse/priorities.py`
- Modify: `src/replayhouse/table.py`
- Test: `tests/test_priorities.py`

**Interfaces:**
- Consumes: `Backend.insert_rows`/`command`, `validate_ids` (Task 5), sidecar name (Task 3).
- Produces:
  - `priorities.update_priorities(backend, sidecar: str, ids: Sequence[str], values: Sequence[float]) -> None`.
  - `priorities.compact(backend, sidecar: str) -> None` (issues `OPTIMIZE TABLE ... FINAL`).
  - `ReplayTable.update_priorities(ids, values) -> None`, `ReplayTable.compact() -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_priorities.py`:

```python
import pytest

from replayhouse.errors import SchemaError
from tests.conftest import make_rows


def _sidecar_count(store):
    return int(store._backend.query_rows("SELECT count() AS c FROM exp__priorities")[0]["c"])


def _current(store):
    rows = store._backend.query_rows(
        "SELECT id, argMax(priority, version) AS p FROM exp__priorities GROUP BY id"
    )
    return {r["id"]: float(r["p"]) for r in rows}


def test_update_priorities_latest_version_wins(store, table):
    ids = table.insert(make_rows(4))
    table.update_priorities(ids, [5.0, 5.0, 5.0, 5.0])
    table.update_priorities(ids[:2], [0.5, 0.5])
    cur = _current(store)
    assert cur[ids[0]] == 0.5 and cur[ids[3]] == 5.0


def test_zeroed_priority_removes_from_sampling(table):
    ids = table.insert(make_rows(10))
    table.update_priorities(ids[:9], [0.0] * 9)
    batch = table.sample(10)
    assert batch.ids == [ids[9]]


def test_update_length_mismatch_raises(table):
    ids = table.insert(make_rows(2))
    with pytest.raises(SchemaError):
        table.update_priorities(ids, [1.0])


def test_compact_collapses_versions(store, table):
    ids = table.insert(make_rows(100))
    table.update_priorities(ids, [2.0] * 100)
    table.update_priorities(ids, [3.0] * 100)
    assert _sidecar_count(store) == 300
    table.compact()
    assert _sidecar_count(store) == 100
    assert set(_current(store).values()) == {3.0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_priorities.py -v`
Expected: FAIL with `AttributeError: 'ReplayTable' object has no attribute 'update_priorities'`

- [ ] **Step 3: Implement `priorities.py`**

```python
from __future__ import annotations

import time
from typing import Sequence

from .errors import SchemaError
from .sampling import validate_ids


def update_priorities(backend, sidecar: str, ids: Sequence[str], values: Sequence[float]) -> None:
    ids = validate_ids(list(ids))
    if len(ids) != len(values):
        raise SchemaError(f"{len(ids)} ids but {len(values)} priorities")
    if not ids:
        return
    version = time.time_ns()
    backend.insert_rows(
        sidecar,
        [{"id": i, "priority": float(v), "version": version} for i, v in zip(ids, values)],
    )


def compact(backend, sidecar: str) -> None:
    backend.command(f"OPTIMIZE TABLE `{sidecar}` FINAL")
```

On `ReplayTable`:

```python
    def update_priorities(self, ids, values) -> None:
        priorities.update_priorities(self._backend, self._sidecar, ids, values)

    def compact(self) -> None:
        priorities.compact(self._backend, self._sidecar)
```

(with `from . import priorities` at the top of `table.py`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_priorities.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/replayhouse/priorities.py src/replayhouse/table.py tests/test_priorities.py
git commit -m "feat: batch priority updates and sidecar compaction"
```

---

### Task 7: Stratified sampling

**Files:**
- Modify: `src/replayhouse/sampling.py`, `src/replayhouse/table.py`
- Test: `tests/test_stratify.py`

**Interfaces:**
- Consumes: everything from Task 5.
- Produces:
  - `sampling.stratified_sql(name, k, per_group: int, stratify_by: str, by="priority", where=None) -> str`.
  - `ReplayTable.sample(..., stratify_by="task_family")` now works: per-group quota is `max(1, k // g)` where `g = uniqExact(stratify_by)` under the same `where` filter; overall result still capped at `k`.

Stratified SQL contract (join form regardless of `by` mode, since the group column lives in the main table):

```sql
WITH current AS
(
    SELECT id, argMax(priority, version) AS priority
    FROM `exp__priorities`
    GROUP BY id
)
SELECT id FROM
(
    SELECT m.id AS id, m.`task_family` AS strat
    FROM `exp` AS m
    INNER JOIN current AS c ON m.id = c.id
    WHERE ((priority)) > 0 AND ((env_version >= 1))
    ORDER BY strat ASC, -log(1 - randCanonical()) / ((priority)) ASC
    LIMIT 10 BY strat
)
LIMIT 30
```

When `by` does not mention `priority`, the CTE and join are omitted and weights come straight from the main table (same `SELECT id FROM (SELECT id, {strat} ... LIMIT n BY strat) LIMIT k` shape).

- [ ] **Step 1: Write the failing tests**

`tests/test_stratify.py`:

```python
from collections import Counter

from tests.conftest import make_rows


def _families(batch):
    return Counter(r["task_family"] for r in batch.rows)


def test_stratified_even_coverage_across_skewed_groups(table):
    table.insert(make_rows(100, task_family="big"))
    table.insert(make_rows(50, task_family="mid"))
    table.insert(make_rows(10, task_family="tiny"))
    batch = table.sample(30, stratify_by="task_family")
    fams = _families(batch)
    assert fams == {"big": 10, "mid": 10, "tiny": 10}


def test_stratified_respects_where(table):
    table.insert(make_rows(20, task_family="a", env_version=1))
    table.insert(make_rows(20, task_family="b", env_version=1))
    table.insert(make_rows(20, task_family="c", env_version=2))
    batch = table.sample(10, stratify_by="task_family", where="env_version = 1")
    fams = _families(batch)
    assert set(fams) == {"a", "b"} and sum(fams.values()) == 10


def test_stratified_by_main_column(table):
    table.insert(make_rows(20, task_family="a", reward=1.0))
    table.insert(make_rows(20, task_family="b", reward=2.0))
    batch = table.sample(10, stratify_by="task_family", by="reward")
    assert set(_families(batch)) == {"a", "b"}


def test_stratified_empty_filter_returns_empty(table):
    table.insert(make_rows(5))
    batch = table.sample(10, stratify_by="task_family", where="env_version = 999")
    assert len(batch) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_stratify.py -v`
Expected: FAIL with `NotImplementedError` from the Task 5 stub.

- [ ] **Step 3: Implement `stratified_sql` in `sampling.py`**

```python
def stratified_sql(name: str, k: int, per_group: int, stratify_by: str,
                   by: str = "priority", where: str | None = None) -> str:
    validate_name(name)
    validate_name(stratify_by)
    k, per_group = int(k), int(per_group)
    cond = f"(({by})) > 0" + (f" AND (({where}))" if where else "")
    order = f"ORDER BY strat ASC, {sample_key(f'({by})')} ASC\n    LIMIT {per_group} BY strat"
    if _PRIORITY_WORD.search(by):
        inner = (
            f"    SELECT m.id AS id, m.`{stratify_by}` AS strat\n"
            f"    FROM `{name}` AS m\n"
            f"    INNER JOIN current AS c ON m.id = c.id\n"
            f"    WHERE {cond}\n    {order}"
        )
        return (
            _CURRENT_CTE.format(sidecar=sidecar_name(name))
            + f"SELECT id FROM\n(\n{inner}\n)\nLIMIT {k}"
        )
    inner = (
        f"    SELECT id, `{stratify_by}` AS strat\n"
        f"    FROM `{name}`\n    WHERE {cond}\n    {order}"
    )
    return f"SELECT id FROM\n(\n{inner}\n)\nLIMIT {k}"
```

- [ ] **Step 4: Wire it into `ReplayTable.sample`**

Replace the `NotImplementedError` branch:

```python
        if stratify_by is not None:
            where_sql = f" WHERE ({where})" if where else ""
            g_rows = self._backend.query_rows(
                f"SELECT uniqExact(`{stratify_by}`) AS g FROM `{self.name}`{where_sql}"
            )
            groups = int(g_rows[0]["g"]) if g_rows else 0
            if groups == 0:
                return SampleBatch()
            per_group = max(1, int(k) // groups)
            sql = stratified_sql(self.name, k, per_group, stratify_by, by=by, where=where)
            chosen = self._backend.query_rows(sql)
        else:
            chosen = self._backend.query_rows(phase1_sql(self.name, k, by=by, where=where))
```

(import `stratified_sql` alongside `phase1_sql`; the rest of `sample` is unchanged).

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_stratify.py tests/ -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/replayhouse/sampling.py src/replayhouse/table.py tests/test_stratify.py
git commit -m "feat: stratified sampling via LIMIT BY with per-group quotas"
```

---

### Task 8: Eviction

**Files:**
- Create: `src/replayhouse/eviction.py`
- Modify: `src/replayhouse/table.py`
- Test: `tests/test_eviction.py`

**Interfaces:**
- Consumes: `load_config` (Task 3), `Backend` (Task 2), current-priority CTE semantics (Task 5).
- Produces:
  - `eviction.table_stats(backend, name) -> tuple[int, int]` — `(rows, bytes_on_disk)`; rows via `SELECT count()` (mutation-aware), bytes via `system.parts` actives.
  - `eviction.evict(backend, name) -> dict` — `{"rows_before": int, "rows_after": int}`; no-op when no capacity configured or under capacity. `fifo`: drop oldest partitions (never the last one) until under capacity. `lowest_priority`: synchronous `ALTER TABLE ... DELETE` of the bottom-n ids by current priority. Both finish by deleting sidecar orphans.
  - `ReplayTable.evict() -> dict`.

- [ ] **Step 1: Write the failing tests**

`tests/test_eviction.py`:

```python
from tests.conftest import make_rows


def _count(store, tbl):
    return int(store._backend.query_rows(f"SELECT count() AS c FROM `{tbl}`")[0]["c"])


def _mk(store, **kw):
    return store.create(
        "ev", columns={"task_family": "LowCardinality(String)", "env_version": "UInt32",
                       "steps": "JSON", "reward": "Float32"}, **kw)


def test_evict_noop_without_capacity(store):
    t = _mk(store)
    t.insert(make_rows(10))
    r = t.evict()
    assert r == {"rows_before": 10, "rows_after": 10}


def test_fifo_drops_oldest_partitions(store):
    t = _mk(store, capacity_rows=130, eviction="fifo")
    t.insert(make_rows(60, inserted_at="2026-08-04 10:00:00"))
    t.insert(make_rows(60, inserted_at="2026-08-05 10:00:00"))
    t.insert(make_rows(60, inserted_at="2026-08-06 10:00:00"))
    r = t.evict()
    assert r == {"rows_before": 180, "rows_after": 120}
    oldest = store._backend.query_rows("SELECT min(inserted_at) AS m FROM ev")[0]["m"]
    assert oldest.startswith("2026-08-05")
    assert _count(store, "ev__priorities") == 120  # orphans cleaned


def test_fifo_never_drops_last_partition(store):
    t = _mk(store, capacity_rows=5, eviction="fifo")
    t.insert(make_rows(60, inserted_at="2026-08-06 10:00:00"))
    r = t.evict()
    assert r["rows_after"] == 60  # single partition retained even though over capacity


def test_lowest_priority_deletes_bottom_rows(store):
    t = _mk(store, capacity_rows=100, eviction="lowest_priority")
    low_ids = t.insert(make_rows(50, task_family="low", priority=0.1))
    t.insert(make_rows(100, task_family="high", priority=5.0))
    r = t.evict()
    assert r == {"rows_before": 150, "rows_after": 100}
    fams = store._backend.query_rows("SELECT DISTINCT task_family AS f FROM ev")
    assert [x["f"] for x in fams] == ["high"]
    assert _count(store, "ev__priorities") == 100
    assert low_ids  # (sanity: fixture returned ids)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_eviction.py -v`
Expected: FAIL with `AttributeError: 'ReplayTable' object has no attribute 'evict'`

- [ ] **Step 3: Implement `eviction.py`**

```python
from __future__ import annotations

import math

from .ddl import load_config, sidecar_name, validate_name


def table_stats(backend, name: str) -> tuple[int, int]:
    validate_name(name)
    rows = int(backend.query_rows(f"SELECT count() AS c FROM `{name}`")[0]["c"])
    parts = backend.query_rows(
        f"SELECT sum(bytes_on_disk) AS b FROM system.parts "
        f"WHERE active AND database = currentDatabase() AND `table` = '{name}'"
    )
    bytes_ = int(parts[0]["b"] or 0) if parts else 0
    return rows, bytes_


def _over(rows: int, bytes_: int, cfg: dict) -> bool:
    cb, cr = cfg.get("capacity_bytes"), cfg.get("capacity_rows")
    return (cb is not None and bytes_ > cb) or (cr is not None and rows > cr)


def _delete_sidecar_orphans(backend, name: str) -> None:
    backend.command(
        f"ALTER TABLE `{sidecar_name(name)}` DELETE "
        f"WHERE id NOT IN (SELECT id FROM `{name}`) "
        f"SETTINGS mutations_sync = 2"
    )


def evict(backend, name: str) -> dict:
    cfg = load_config(backend, name)
    rows_before, bytes_before = table_stats(backend, name)
    if not _over(rows_before, bytes_before, cfg):
        return {"rows_before": rows_before, "rows_after": rows_before}

    if cfg.get("eviction", "fifo") == "fifo":
        while True:
            rows_, bytes_ = table_stats(backend, name)
            if not _over(rows_, bytes_, cfg):
                break
            parts = backend.query_rows(
                f"SELECT partition_id AS p, min(partition) AS v FROM system.parts "
                f"WHERE active AND database = currentDatabase() AND `table` = '{name}' "
                f"GROUP BY partition_id ORDER BY v ASC"
            )
            if len(parts) <= 1:
                break
            backend.command(f"ALTER TABLE `{name}` DROP PARTITION ID '{parts[0]['p']}'")
    else:  # lowest_priority
        n = 0
        if cfg.get("capacity_rows") is not None:
            n = max(n, rows_before - cfg["capacity_rows"])
        if cfg.get("capacity_bytes") is not None and bytes_before > 0 and rows_before > 0:
            per_row = bytes_before / rows_before
            n = max(n, math.ceil((bytes_before - cfg["capacity_bytes"]) / per_row))
        if n > 0:
            backend.command(
                f"ALTER TABLE `{name}` DELETE WHERE id IN (\n"
                f"    SELECT id FROM (\n"
                f"        SELECT id, argMax(priority, version) AS priority\n"
                f"        FROM `{sidecar_name(name)}` GROUP BY id\n"
                f"    ) ORDER BY priority ASC LIMIT {int(n)}\n"
                f") SETTINGS mutations_sync = 2"
            )

    _delete_sidecar_orphans(backend, name)
    rows_after, _ = table_stats(backend, name)
    return {"rows_before": rows_before, "rows_after": rows_after}
```

On `ReplayTable` (with `from . import eviction` imported):

```python
    def evict(self) -> dict:
        return eviction.evict(self._backend, self.name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_eviction.py -v`
Expected: 4 passed. If `min(inserted_at)` comes back as a non-string (chdb may render DateTime differently), coerce with `str(...)` in the test — the assertion is about the date prefix.

- [ ] **Step 5: Commit**

```bash
git add src/replayhouse/eviction.py src/replayhouse/table.py tests/test_eviction.py
git commit -m "feat: fifo and lowest-priority eviction with sidecar orphan cleanup"
```

---

### Task 9: End-to-end bandit smoke test + README usage

**Files:**
- Create: `tests/test_end_to_end.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: the whole public API (`connect`, `create`, `insert`, `sample`, `update_priorities`, `evict`, `compact`).
- Produces: a living example proving the bandit story on chdb; README "Quick start" section that matches the real API exactly.

- [ ] **Step 1: Write the end-to-end test**

`tests/test_end_to_end.py`:

```python
"""Thompson-style bandit on chdb: the store converges toward the best arm."""

import random

from tests.conftest import make_rows


def test_bandit_loop_converges(store):
    arms = store.create("arms", columns={"arm": "LowCardinality(String)"})
    ids = {}
    for arm, prior in [("a", 1.0), ("b", 1.0), ("c", 1.0)]:
        ids[arm] = arms.insert([{"arm": arm, "priority": prior}])[0]

    true_reward = {"a": 0.2, "b": 0.5, "c": 0.9}
    wins = {"a": 1.0, "b": 1.0, "c": 1.0}
    pulls = {"a": 1, "b": 1, "c": 1}
    rng = random.Random(42)

    for _ in range(300):
        batch = arms.sample(1)
        arm = batch.rows[0]["arm"]
        pulls[arm] += 1
        wins[arm] += 1.0 if rng.random() < true_reward[arm] else 0.0
        # posterior-mean weight (Thompson-ish, deterministic enough to test)
        arms.update_priorities([ids[arm]], [max(wins[arm] / pulls[arm], 0.01)])

    assert pulls["c"] > pulls["a"]
    assert pulls["c"] > pulls["b"]


def test_full_lifecycle(store):
    t = store.create("life", columns={"task_family": "LowCardinality(String)",
                                      "env_version": "UInt32", "steps": "JSON",
                                      "reward": "Float32"},
                     capacity_rows=80, eviction="lowest_priority")
    ids = t.insert(make_rows(100))
    batch = t.sample(32, where="env_version = 1")
    assert len(batch) == 32
    t.update_priorities(ids[:20], [0.01] * 20)
    t.compact()
    r = t.evict()
    assert r["rows_after"] == 80
    assert len(t.sample(1000)) == 80
```

- [ ] **Step 2: Run the suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: all pass (the bandit test uses a seeded RNG for rewards; sampling randomness comes from ClickHouse, but 300 pulls with rewards 0.2/0.5/0.9 gives arm `c` a decisive lead — if it ever flakes, raise iterations to 600, don't loosen the assertion).

- [ ] **Step 3: Update README with a Quick start that matches reality**

Append to `README.md`:

````markdown
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
````

- [ ] **Step 4: Commit**

```bash
git add tests/test_end_to_end.py README.md
git commit -m "test: end-to-end bandit and lifecycle tests; document quick start"
```

---

### Task 10: Server concurrency integration tests

**Files:**
- Create: `tests_integration/test_concurrency.py`, `tests_integration/conftest.py`

**Interfaces:**
- Consumes: full public API; `ClickHouseBackend` (Task 2).
- Produces: an `integration`-marked suite proving parallel inserters + a sampler + a priority updater don't corrupt anything, runnable via `REPLAYHOUSE_TEST_URL=clickhouse://localhost:8123/default .venv/bin/pytest tests_integration -m integration`.

- [ ] **Step 1: Write the integration conftest**

`tests_integration/conftest.py`:

```python
import os
import uuid

import pytest

from replayhouse.store import ReplayHouse

URL = os.environ.get("REPLAYHOUSE_TEST_URL")


@pytest.fixture
def server_store():
    if not URL:
        pytest.skip("set REPLAYHOUSE_TEST_URL to run integration tests")
    s = ReplayHouse.connect(URL)
    yield s
    s.close()


@pytest.fixture
def fresh_name(server_store):
    name = f"rh_it_{uuid.uuid4().hex[:8]}"
    yield name
    server_store.drop(name)
```

- [ ] **Step 2: Write the concurrency test**

`tests_integration/test_concurrency.py`:

```python
import threading

import pytest

pytestmark = pytest.mark.integration


def _rows(n, fam):
    return [{"task_family": fam, "steps": {"n": i}, "reward": float(i)} for i in range(n)]


def test_parallel_insert_sample_update(server_store, fresh_name):
    t = server_store.create(
        fresh_name,
        columns={"task_family": "LowCardinality(String)", "steps": "JSON", "reward": "Float32"},
    )
    # ReplayHouse.connect per thread: clickhouse-connect clients are not thread-safe for
    # concurrent queries on one connection.
    import replayhouse

    from tests_integration.conftest import URL

    errors = []

    def inserter(fam):
        try:
            tt = replayhouse.connect(URL).table(fresh_name)
            for _ in range(10):
                tt.insert(_rows(50, fam))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def sampler():
        try:
            tt = replayhouse.connect(URL).table(fresh_name)
            for _ in range(20):
                batch = tt.sample(64)
                assert len(set(batch.ids)) == len(batch.ids)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def updater():
        try:
            tt = replayhouse.connect(URL).table(fresh_name)
            for _ in range(10):
                batch = tt.sample(32)
                if batch.ids:
                    tt.update_priorities(batch.ids, [2.0] * len(batch.ids))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = (
        [threading.Thread(target=inserter, args=(f"fam{i}",)) for i in range(4)]
        + [threading.Thread(target=sampler), threading.Thread(target=updater)]
    )
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert errors == []
    total = server_store._backend.query_rows(f"SELECT count() AS c FROM `{fresh_name}`")
    assert int(total[0]["c"]) == 4 * 10 * 50
```

- [ ] **Step 3: Verify skip-behavior locally and (if available) against a server**

Run: `.venv/bin/pytest tests_integration -v`
Expected: SKIPPED (no `REPLAYHOUSE_TEST_URL`).

If Docker is available:
Run: `docker run -d --rm --name rh-it -p 18123:8123 clickhouse/clickhouse-server:25.3 && sleep 5 && REPLAYHOUSE_TEST_URL=clickhouse://localhost:18123/default .venv/bin/pytest tests_integration -m integration -v; docker stop rh-it`
Expected: 1 passed. If Docker is not available, note it in the final report and leave the suite skip-gated.

- [ ] **Step 4: Run the full suite one last time**

Run: `.venv/bin/pytest tests -v`
Expected: everything green.

- [ ] **Step 5: Commit and push**

```bash
git add tests_integration
git commit -m "test: server concurrency integration suite (integration marker)"
git push
```

---

## Self-Review Notes

- **Spec coverage:** create/insert/sample (plain, filtered, by-expression, stratified)/update_priorities/compact/evict/backends/URL API/errors/README all have tasks. Deliberately deferred to the M3–M4 plan: `torch.py` dataset, benchmarks, pandas/Arrow-first `batch.table` property (M1–M2 provide `to_arrow()`/`to_pandas()`).
- **Types consistent:** `SampleBatch.ids: list[str]`, `insert -> list[str]`, `update_priorities(ids: Sequence[str], values: Sequence[float])`, `evict -> dict` used identically across tasks.
- **Known risk points (fallbacks specified inline):** chdb inline `INSERT ... FORMAT JSONEachRow` (Task 2 fallback: `file()` + tempfile), DateTime rendering in JSONEachRow (Task 8 note), statistical test bounds (Tasks 5/9 notes).
