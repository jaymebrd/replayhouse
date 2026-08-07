import pytest

from replayhouse.errors import SchemaError
from replayhouse.sampling import phase1_sql, sample_key
from tests.conftest import make_rows


def test_sample_key_seeded_form():
    key = sample_key("priority", seed=42)
    assert "cityHash64(id, 42)" in key and "randCanonical" not in key


def test_sample_key_unseeded_form_unchanged():
    key = sample_key("priority")
    assert "randCanonical()" in key and "cityHash64" not in key


def test_phase1_join_mode_qualifies_id():
    sql = phase1_sql("exp", 10, by="priority * reward", seed=7)
    assert "cityHash64(m.id, 7)" in sql


def test_same_seed_same_batch(table):
    table.insert(make_rows(100))
    a = table.sample(20, seed=42)
    b = table.sample(20, seed=42)
    assert sorted(a.ids) == sorted(b.ids)


def test_different_seeds_differ(table):
    table.insert(make_rows(100))
    a = table.sample(20, seed=1)
    b = table.sample(20, seed=2)
    assert sorted(a.ids) != sorted(b.ids)


def test_seeded_with_filter_and_stratify(table):
    table.insert(make_rows(30, task_family="a", env_version=1))
    table.insert(make_rows(30, task_family="b", env_version=1))
    table.insert(make_rows(30, task_family="c", env_version=2))
    a = table.sample(10, where="env_version = 1", stratify_by="task_family", seed=5)
    b = table.sample(10, where="env_version = 1", stratify_by="task_family", seed=5)
    assert sorted(a.ids) == sorted(b.ids)
    assert {r["task_family"] for r in a.rows} == {"a", "b"}


def test_seeded_weights_still_honored(table):
    table.insert(make_rows(200, task_family="heavy", priority=9.0))
    table.insert(make_rows(200, task_family="light", priority=1.0))
    batch = table.sample(200, seed=3)
    heavy = sum(1 for r in batch.rows if r["task_family"] == "heavy")
    # Same analytic expectation (~164) as the unseeded statistical test;
    # deterministic given the seed, so this can never flake once green.
    assert 130 < heavy < 195


def test_seeded_excludes_nonpositive_weights(table):
    table.insert(make_rows(5, task_family="dead", priority=0.0))
    table.insert(make_rows(5, task_family="live"))
    batch = table.sample(10, seed=11)
    assert len(batch) == 5
    assert {r["task_family"] for r in batch.rows} == {"live"}


def test_non_int_seed_raises(table):
    table.insert(make_rows(5))
    with pytest.raises(SchemaError):
        table.sample(5, seed="42; DROP TABLE exp")
