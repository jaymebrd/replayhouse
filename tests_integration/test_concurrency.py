import threading

import pytest

pytestmark = pytest.mark.integration


def _rows(n, fam):
    return [{"task_family": fam, "steps": {"n": i}, "reward": float(i)} for i in range(n)]


def test_parallel_insert_sample_update(server_store, fresh_name):
    t = server_store.create(
        fresh_name,
        columns={"task_family": "LowCardinality(String)", "steps": "JSON", "reward": "Float32"},
    )
    # ReplayHouse.connect per thread: clickhouse-connect clients are not thread-safe for
    # concurrent queries on one connection.
    import replayhouse

    from tests_integration.conftest import URL

    errors = []

    def inserter(fam):
        try:
            tt = replayhouse.connect(URL).table(fresh_name)
            for _ in range(10):
                tt.insert(_rows(50, fam))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def sampler():
        try:
            tt = replayhouse.connect(URL).table(fresh_name)
            for _ in range(20):
                batch = tt.sample(64)
                assert len(set(batch.ids)) == len(batch.ids)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def updater():
        try:
            tt = replayhouse.connect(URL).table(fresh_name)
            for _ in range(10):
                batch = tt.sample(32)
                if batch.ids:
                    tt.update_priorities(batch.ids, [2.0] * len(batch.ids))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = (
        [threading.Thread(target=inserter, args=(f"fam{i}",)) for i in range(4)]
        + [threading.Thread(target=sampler), threading.Thread(target=updater)]
    )
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert errors == []
    total = server_store._backend.query_rows(f"SELECT count() AS c FROM `{fresh_name}`")
    assert int(total[0]["c"]) == 4 * 10 * 50
