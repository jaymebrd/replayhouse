from tests.conftest import make_rows


def _count(store, tbl):
    return int(store._backend.query_rows(f"SELECT count() AS c FROM `{tbl}`")[0]["c"])


def _mk(store, **kw):
    return store.create(
        "ev", columns={"task_family": "LowCardinality(String)", "env_version": "UInt32",
                       "steps": "JSON", "reward": "Float32"}, **kw)


def test_evict_noop_without_capacity(store):
    t = _mk(store)
    t.insert(make_rows(10))
    r = t.evict()
    assert r == {"rows_before": 10, "rows_after": 10}


def test_fifo_drops_oldest_partitions(store):
    t = _mk(store, capacity_rows=130, eviction="fifo")
    t.insert(make_rows(60, inserted_at="2026-08-04 10:00:00"))
    t.insert(make_rows(60, inserted_at="2026-08-05 10:00:00"))
    t.insert(make_rows(60, inserted_at="2026-08-06 10:00:00"))
    r = t.evict()
    assert r == {"rows_before": 180, "rows_after": 120}
    oldest = store._backend.query_rows("SELECT min(inserted_at) AS m FROM ev")[0]["m"]
    assert str(oldest).startswith("2026-08-05")
    assert _count(store, "ev__priorities") == 120  # orphans cleaned


def test_fifo_never_drops_last_partition(store):
    t = _mk(store, capacity_rows=5, eviction="fifo")
    t.insert(make_rows(60, inserted_at="2026-08-06 10:00:00"))
    r = t.evict()
    assert r["rows_after"] == 60  # single partition retained even though over capacity


def test_lowest_priority_deletes_bottom_rows(store):
    t = _mk(store, capacity_rows=100, eviction="lowest_priority")
    low_ids = t.insert(make_rows(50, task_family="low", priority=0.1))
    t.insert(make_rows(100, task_family="high", priority=5.0))
    r = t.evict()
    assert r == {"rows_before": 150, "rows_after": 100}
    fams = store._backend.query_rows("SELECT DISTINCT task_family AS f FROM ev")
    assert [x["f"] for x in fams] == ["high"]
    assert _count(store, "ev__priorities") == 100
    assert low_ids  # (sanity: fixture returned ids)


def test_evict_cleans_orphans_even_when_under_capacity(store):
    # No capacity config at all: evict() should still never leave sidecar
    # orphans behind, e.g. after a TTL mutation deletes main rows out from
    # under the sidecar.
    t = _mk(store)
    ids = t.insert(make_rows(10))
    store._backend.command(
        f"ALTER TABLE `ev` DELETE WHERE id = '{ids[0]}' SETTINGS mutations_sync = 2"
    )
    assert _count(store, "ev") == 9
    assert _count(store, "ev__priorities") == 10  # orphan still present

    r = t.evict()
    assert r == {"rows_before": 9, "rows_after": 9}
    assert _count(store, "ev__priorities") == _count(store, "ev") == 9

    batch = t.sample(9)
    assert len(batch) == 9  # no short batch caused by dead ids in phase-1 LIMIT


def test_lowest_priority_evicts_to_capacity_despite_preexisting_orphans(store):
    t = _mk(store, capacity_rows=100, eviction="lowest_priority")
    ids = t.insert(make_rows(20, task_family="soon_gone", priority=0.05))
    # Manufacture sidecar orphans before the table is ever over capacity.
    store._backend.command(
        f"ALTER TABLE `ev` DELETE WHERE id = '{ids[0]}' SETTINGS mutations_sync = 2"
    )
    assert _count(store, "ev__priorities") > _count(store, "ev")

    t.insert(make_rows(19, task_family="soon_gone", priority=0.05))
    t.insert(make_rows(100, task_family="high", priority=5.0))
    r = t.evict()
    assert r["rows_after"] <= 100
