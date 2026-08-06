"""PyTorch integration. Requires the extra: pip install 'replayhouse[torch]'."""

from __future__ import annotations

from typing import Iterator

from .table import ReplayTable, SampleBatch

try:
    from torch.utils.data import IterableDataset
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "replayhouse.torch requires PyTorch; install with: pip install 'replayhouse[torch]'"
    ) from e


class ReplayIterableDataset(IterableDataset):
    """Iterates weighted sample batches drawn from a ReplayTable.

    Each item is a whole SampleBatch (the store does the batching), so use
    DataLoader(ds, batch_size=None) with num_workers=0 — backend connections
    are not fork-safe, and extra workers would only draw overlapping random
    batches. Iteration stops when the store yields an empty batch, or after
    num_batches if given.
    """

    def __init__(self, table: ReplayTable, batch_size: int, *,
                 by: str = "priority", where: str | None = None,
                 stratify_by: str | None = None,
                 num_batches: int | None = None):
        self._table = table
        self._batch_size = int(batch_size)
        self._by = by
        self._where = where
        self._stratify_by = stratify_by
        self._num_batches = num_batches

    def __iter__(self) -> Iterator[SampleBatch]:
        produced = 0
        while self._num_batches is None or produced < self._num_batches:
            batch = self._table.sample(
                self._batch_size, by=self._by, where=self._where,
                stratify_by=self._stratify_by,
            )
            if len(batch) == 0:
                return
            yield batch
            produced += 1
