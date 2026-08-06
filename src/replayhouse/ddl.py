from __future__ import annotations

import json
import re

from .errors import SchemaError

RESERVED_COLUMNS = ("id", "inserted_at", "priority")
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
