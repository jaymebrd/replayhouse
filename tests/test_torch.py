import itertools
import subprocess
import sys

import pytest

from tests.conftest import make_rows

torch = pytest.importorskip("torch")


def test_importing_replayhouse_does_not_import_torch():
    # Lazy-import contract: the core package must not pull torch in.
    code = "import replayhouse, sys; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_dataset_yields_batches_and_stops_at_num_batches(table):
    from replayhouse.torch import ReplayIterableDataset

    table.insert(make_rows(500))
    ds = ReplayIterableDataset(table, 100, num_batches=3)
    batches = list(ds)
    assert len(batches) == 3
    assert all(len(b) == 100 for b in batches)


def test_dataset_infinite_mode_with_islice(table):
    from replayhouse.torch import ReplayIterableDataset

    table.insert(make_rows(50))
    ds = ReplayIterableDataset(table, 10)
    batches = list(itertools.islice(iter(ds), 5))
    assert len(batches) == 5


def test_dataset_stops_on_empty_store(table):
    from replayhouse.torch import ReplayIterableDataset

    assert list(ReplayIterableDataset(table, 10)) == []


def test_dataset_respects_where_and_by(table):
    from replayhouse.torch import ReplayIterableDataset

    table.insert(make_rows(20, env_version=1))
    table.insert(make_rows(20, env_version=2))
    ds = ReplayIterableDataset(table, 50, where="env_version = 2", num_batches=1)
    (batch,) = list(ds)
    assert len(batch) == 20
    assert all(int(r["env_version"]) == 2 for r in batch.rows)


def test_dataset_works_under_dataloader(table):
    from torch.utils.data import DataLoader

    from replayhouse.torch import ReplayIterableDataset

    table.insert(make_rows(200))
    ds = ReplayIterableDataset(table, 64, num_batches=2)
    seen = 0
    for batch in DataLoader(ds, batch_size=None, num_workers=0):
        rewards = torch.tensor([float(r["reward"]) for r in batch.rows])
        assert rewards.shape == (len(batch),)
        seen += 1
    assert seen == 2


def test_dataset_is_iterable_dataset_subclass():
    from torch.utils.data import IterableDataset

    from replayhouse.torch import ReplayIterableDataset

    assert issubclass(ReplayIterableDataset, IterableDataset)
