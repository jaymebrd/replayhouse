from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from . import priorities
from .backend import Backend
from ._ids import uuid7
from .ddl import sidecar_name, validate_name
from .errors import SchemaError
from .sampling import fetch_sql, phase1_sql, stratified_sql


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


class ReplayTable:
    def __init__(self, backend: Backend, name: str):
        self._backend = backend
        self.name = name
        self._sidecar = sidecar_name(name)

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

    def sample(self, k: int, *, by: str = "priority", where: str | None = None,
               stratify_by: str | None = None) -> SampleBatch:
        if stratify_by is not None:
            validate_name(stratify_by)
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
        ids = [r["id"] for r in chosen]
        if not ids:
            return SampleBatch()
        rows = self._backend.query_rows(fetch_sql(self.name, ids))
        return SampleBatch(ids=[r["id"] for r in rows], rows=rows)

    def update_priorities(self, ids, values) -> None:
        priorities.update_priorities(self._backend, self._sidecar, ids, values)

    def compact(self) -> None:
        priorities.compact(self._backend, self._sidecar)
