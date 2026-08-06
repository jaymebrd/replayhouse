import pytest

from replayhouse.store import ReplayHouse


@pytest.fixture
def store(tmp_path):
    s = ReplayHouse.connect(f"chdb://{tmp_path}/db")
    yield s
    s.close()


@pytest.fixture
def table(store):
    return store.create(
        "exp",
        columns={
            "task_family": "LowCardinality(String)",
            "env_version": "UInt32",
            "steps": "JSON",
            "reward": "Float32",
        },
    )


def make_rows(n, *, task_family="web", env_version=1, priority=None, inserted_at=None, reward=None):
    out = []
    for i in range(n):
        r = {
            "task_family": task_family,
            "env_version": env_version,
            "steps": {"i": i},
            "reward": float(i) if reward is None else float(reward),
        }
        if priority is not None:
            r["priority"] = float(priority)
        if inserted_at is not None:
            r["inserted_at"] = inserted_at
        out.append(r)
    return out
