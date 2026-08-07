import pytest

from replayhouse.errors import BackendError
from tests.conftest import make_rows


def test_query_returns_rows(store, table):
    table.insert(make_rows(5))
    rows = store.query("SELECT count() AS c FROM exp")
    assert int(rows[0]["c"]) == 5


def test_query_arbitrary_sql(store, table):
    table.insert(make_rows(10, task_family="a"))
    table.insert(make_rows(20, task_family="b"))
    rows = store.query(
        "SELECT task_family, count() AS c FROM exp GROUP BY task_family ORDER BY c"
    )
    assert [(r["task_family"], int(r["c"])) for r in rows] == [("a", 10), ("b", 20)]


def test_query_bad_sql_raises_backend_error(store):
    with pytest.raises(BackendError):
        store.query("SELECT nonsense FROM nowhere")
