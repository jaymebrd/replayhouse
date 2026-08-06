import pytest

from replayhouse.backend import ChdbBackend, parse_url
from replayhouse.errors import BackendError


def test_parse_url_chdb():
    kind, cfg = parse_url("chdb:///some/dir/db")
    assert kind == "chdb"
    assert cfg == {"path": "/some/dir/db"}


def test_parse_url_clickhouse_defaults():
    kind, cfg = parse_url("clickhouse://host.example.com/mydb")
    assert kind == "clickhouse"
    assert cfg == {
        "host": "host.example.com", "port": 8123, "username": "default",
        "password": "", "database": "mydb", "secure": False,
    }


def test_parse_url_clickhouse_secure_with_creds():
    kind, cfg = parse_url("clickhouses://alice:s3cret@h:9443/db1")
    assert cfg["port"] == 9443 and cfg["secure"] is True
    assert cfg["username"] == "alice" and cfg["password"] == "s3cret"


def test_parse_url_rejects_unknown_scheme():
    with pytest.raises(BackendError):
        parse_url("postgres://x/y")


@pytest.fixture
def backend(tmp_path):
    b = ChdbBackend(str(tmp_path / "db"))
    yield b
    b.close()


def test_chdb_roundtrip_rows_and_json(backend):
    backend.command("CREATE TABLE t (x UInt32, s String, j JSON) ENGINE = MergeTree ORDER BY x")
    backend.insert_rows("t", [{"x": 1, "s": "a", "j": {"k": [1, 2]}}, {"x": 2, "s": "b", "j": {"k": []}}])
    rows = backend.query_rows("SELECT x, s, j FROM t ORDER BY x")
    assert [r["s"] for r in rows] == ["a", "b"]
    assert rows[0]["j"] == {"k": [1, 2]}


def test_chdb_query_arrow(backend):
    tbl = backend.query_arrow("SELECT number FROM numbers(3)")
    assert tbl.num_rows == 3


def test_chdb_insert_empty_is_noop(backend):
    backend.command("CREATE TABLE e (x UInt32) ENGINE = MergeTree ORDER BY x")
    backend.insert_rows("e", [])
    assert backend.query_rows("SELECT count() AS c FROM e")[0]["c"] in (0, "0")


def test_chdb_bad_sql_raises(backend):
    with pytest.raises(BackendError):
        backend.query_rows("SELECT nonsense FROM nowhere")
