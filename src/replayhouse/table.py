from __future__ import annotations

import time
from typing import Any

from .backend import Backend
from ._ids import uuid7
from .ddl import sidecar_name
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
