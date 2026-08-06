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
    # Sidecar orphans (rows whose main-table id has already been deleted, e.g. by
    # TTL expiry) must be cleaned on every call, not only when eviction actually
    # runs below. Otherwise they accumulate forever and waste LIMIT slots in the
    # phase-1 sampling query, causing `sample(k)` to silently return fewer than
    # k rows.
    _delete_sidecar_orphans(backend, name)
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
