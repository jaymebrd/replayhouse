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
