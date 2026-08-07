import pytest

from replayhouse import TableExistsError


def test_create_twice_raises_typed_error(store):
    store.create("dup", columns={"x": "UInt32"})
    with pytest.raises(TableExistsError) as exc:
        store.create("dup", columns={"x": "UInt32"})
    assert exc.value.table_name == "dup"


def test_create_exists_ok_returns_usable_table(store):
    t1 = store.create("keep", columns={"x": "UInt32"})
    t1.insert([{"x": 1}])
    t2 = store.create("keep", columns={"x": "UInt32"}, exists_ok=True)
    t2.insert([{"x": 2}])
    assert int(store.query("SELECT count() AS c FROM keep")[0]["c"]) == 2


def test_exists_ok_false_is_default(store):
    store.create("strict", columns={"x": "UInt32"})
    with pytest.raises(TableExistsError):
        store.create("strict", columns={"x": "UInt32"}, exists_ok=False)
