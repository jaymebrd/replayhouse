import pytest

from replayhouse.errors import SchemaError
from replayhouse.sampling import phase1_sql, validate_ids
from tests.conftest import make_rows


def test_phase1_sql_default_uses_sidecar_only():
    sql = phase1_sql("exp", 10)
    assert "exp__priorities" in sql and "JOIN" not in sql and "IN (SELECT" not in sql


def test_phase1_sql_where_adds_semijoin():
    sql = phase1_sql("exp", 10, where="env_version >= 2")
    assert "IN (SELECT id FROM `exp` WHERE (env_version >= 2))" in sql


def test_phase1_sql_main_column_skips_sidecar():
    sql = phase1_sql("exp", 10, by="reward")
    assert "exp__priorities" not in sql


def test_phase1_sql_mixed_expression_joins():
    sql = phase1_sql("exp", 10, by="priority * reward")
    assert "INNER JOIN" in sql and "exp__priorities" in sql


def test_validate_ids():
    good = ["0198a111-2222-7333-8444-555566667777"]
    assert validate_ids(good) == good
    with pytest.raises(SchemaError):
        validate_ids(["x'); DROP TABLE exp; --"])


def test_sample_without_replacement_and_k(table):
    table.insert(make_rows(50))
    batch = table.sample(20)
    assert len(batch) == 20
    assert len(set(batch.ids)) == 20
    assert set(batch.rows[0]) >= {"id", "task_family", "env_version", "steps", "reward"}


def test_sample_more_than_population_returns_all(table):
    table.insert(make_rows(5))
    assert len(table.sample(100)) == 5


def test_sample_excludes_nonpositive_priority(table):
    table.insert(make_rows(5, task_family="dead", priority=0.0))
    table.insert(make_rows(5, task_family="live"))
    batch = table.sample(10)
    assert len(batch) == 5
    assert {r["task_family"] for r in batch.rows} == {"live"}


def test_sample_respects_where(table):
    table.insert(make_rows(10, env_version=1))
    table.insert(make_rows(10, env_version=2))
    batch = table.sample(20, where="env_version = 2")
    assert len(batch) == 10
    assert all(int(r["env_version"]) == 2 for r in batch.rows)


def test_sample_by_main_column(table):
    table.insert(make_rows(1, reward=0.0))      # weight 0 -> excluded
    table.insert(make_rows(10, reward=3.0))
    batch = table.sample(20, by="reward")
    assert len(batch) == 10


def test_sample_weights_are_honored_statistically(table):
    table.insert(make_rows(200, task_family="heavy", priority=9.0))
    table.insert(make_rows(200, task_family="light", priority=1.0))
    heavy = 0
    for _ in range(5):
        batch = table.sample(200)
        heavy += sum(1 for r in batch.rows if r["task_family"] == "heavy")
    heavy_avg = heavy / 5
    # Analytic expectation ~164 of 200; generous non-flaky bounds.
    assert 130 < heavy_avg < 195


def test_empty_table_returns_empty_batch(table):
    batch = table.sample(10)
    assert len(batch) == 0 and batch.ids == [] and batch.to_arrow().num_rows == 0
