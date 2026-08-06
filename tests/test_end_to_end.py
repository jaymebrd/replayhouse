"""Thompson-style bandit on chdb: the store converges toward the best arm."""

import random

from tests.conftest import make_rows


def test_bandit_loop_converges(store):
    arms = store.create("arms", columns={"arm": "LowCardinality(String)"})
    ids = {}
    for arm, prior in [("a", 1.0), ("b", 1.0), ("c", 1.0)]:
        ids[arm] = arms.insert([{"arm": arm, "priority": prior}])[0]

    true_reward = {"a": 0.2, "b": 0.5, "c": 0.9}
    wins = {"a": 1.0, "b": 1.0, "c": 1.0}
    pulls = {"a": 1, "b": 1, "c": 1}
    rng = random.Random(42)

    for _ in range(300):
        batch = arms.sample(1)
        arm = batch.rows[0]["arm"]
        pulls[arm] += 1
        wins[arm] += 1.0 if rng.random() < true_reward[arm] else 0.0
        # posterior-mean weight (Thompson-ish, deterministic enough to test)
        arms.update_priorities([ids[arm]], [max(wins[arm] / pulls[arm], 0.01)])

    assert pulls["c"] > pulls["a"]
    assert pulls["c"] > pulls["b"]


def test_full_lifecycle(store):
    t = store.create("life", columns={"task_family": "LowCardinality(String)",
                                      "env_version": "UInt32", "steps": "JSON",
                                      "reward": "Float32"},
                     capacity_rows=80, eviction="lowest_priority")
    ids = t.insert(make_rows(100))
    batch = t.sample(32, where="env_version = 1")
    assert len(batch) == 32
    t.update_priorities(ids[:20], [0.01] * 20)
    t.compact()
    r = t.evict()
    assert r["rows_after"] == 80
    assert len(t.sample(1000)) == 80
