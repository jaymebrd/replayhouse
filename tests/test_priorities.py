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
