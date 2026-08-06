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

    def _wrap_call(self, fn):
        """Wrap a client call to convert exceptions to BackendError."""
        try:
            return fn()
        except Exception as e:
            raise BackendError(str(e)) from e

    def query_arrow(self, sql: str) -> pa.Table:
        return self._wrap_call(lambda: self._client.query_arrow(sql))

    def query_rows(self, sql: str) -> list[dict[str, Any]]:
        def _query():
            raw = self._client.raw_query(sql, fmt="JSONEachRow")
            return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        return self._wrap_call(_query)

    def insert_rows(self, table: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self._wrap_call(lambda: self._client.raw_insert(
            f"`{table}`", insert_block=_ndjson(rows), fmt="JSONEachRow"
        ))

    def command(self, sql: str) -> None:
        self._wrap_call(lambda: self._client.command(sql))

    def close(self) -> None:
        self._client.close()
