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
