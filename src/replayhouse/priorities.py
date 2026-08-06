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
