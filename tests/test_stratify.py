from collections import Counter

from tests.conftest import make_rows


def _families(batch):
    return Counter(r["task_family"] for r in batch.rows)


def test_stratified_even_coverage_across_skewed_groups(table):
    table.insert(make_rows(100, task_family="big"))
    table.insert(make_rows(50, task_family="mid"))
    table.insert(make_rows(10, task_family="tiny"))
    batch = table.sample(30, stratify_by="task_family")
    fams = _families(batch)
    assert fams == {"big": 10, "mid": 10, "tiny": 10}


def test_stratified_respects_where(table):
    table.insert(make_rows(20, task_family="a", env_version=1))
    table.insert(make_rows(20, task_family="b", env_version=1))
    table.insert(make_rows(20, task_family="c", env_version=2))
    batch = table.sample(10, stratify_by="task_family", where="env_version = 1")
    fams = _families(batch)
    assert set(fams) == {"a", "b"} and sum(fams.values()) == 10


def test_stratified_by_main_column(table):
    table.insert(make_rows(20, task_family="a", reward=1.0))
    table.insert(make_rows(20, task_family="b", reward=2.0))
    batch = table.sample(10, stratify_by="task_family", by="reward")
    assert set(_families(batch)) == {"a", "b"}


def test_stratified_empty_filter_returns_empty(table):
    table.insert(make_rows(5))
    batch = table.sample(10, stratify_by="task_family", where="env_version = 999")
    assert len(batch) == 0
