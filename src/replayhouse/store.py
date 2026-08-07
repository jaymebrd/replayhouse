from __future__ import annotations

from . import ddl
from .backend import Backend, backend_from_url
from .errors import TableExistsError
from .table import ReplayTable


class ReplayHouse:
    def __init__(self, backend: Backend):
        self._backend = backend

    @classmethod
    def connect(cls, url: str) -> "ReplayHouse":
        return cls(backend_from_url(url))

    def create(self, name, columns, *, ttl_days=None, capacity_bytes=None,
               capacity_rows=None, eviction="fifo", exists_ok: bool = False) -> ReplayTable:
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
        config = ddl.build_config(
            capacity_bytes=capacity_bytes, capacity_rows=capacity_rows, eviction=eviction
        )
        self._backend.command(ddl.main_table_ddl(name, columns, ttl_days, config))
        try:
            self._backend.command(ddl.sidecar_ddl(name))
        except Exception:
            # Roll back the main table if sidecar creation fails
            self._backend.command(f"DROP TABLE IF EXISTS `{name}`")
            raise
        return self.table(name)

    def table(self, name: str) -> ReplayTable:
        return ReplayTable(self._backend, ddl.validate_name(name))

    def drop(self, name: str) -> None:
        ddl.validate_name(name)
        self._backend.command(f"DROP TABLE IF EXISTS `{ddl.sidecar_name(name)}`")
        self._backend.command(f"DROP TABLE IF EXISTS `{name}`")

    def query(self, sql: str) -> list[dict]:
        """Run arbitrary SQL against the store's backend and return rows.

        Trusted input (same contract as sample's by=/where=): this is a
        database client, not a sandbox. 64-bit integers arrive as strings
        (JSONEachRow); coerce with int() as needed.
        """
        return self._backend.query_rows(sql)

    def close(self) -> None:
        self._backend.close()


def connect(url: str) -> ReplayHouse:
    return ReplayHouse.connect(url)
