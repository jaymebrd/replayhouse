"""Prioritized experience replay demo: train a reward model on chdb.

Inserts synthetic experiences, trains a small torch model on weighted
batches, and feeds per-sample |error| back as new priorities — the classic
PER loop, with ClickHouse as the buffer.

Run: python examples/train_reward_model.py   (needs replayhouse[torch])
"""

import random
import tempfile

import torch

import replayhouse
from replayhouse.torch import ReplayIterableDataset


def main():
    rng = random.Random(0)
    with tempfile.TemporaryDirectory() as tmp:
        store = replayhouse.connect(f"chdb://{tmp}/db")
        t = store.create("exp", columns={"x1": "Float32", "x2": "Float32",
                                         "reward": "Float32"})
        rows = []
        for _ in range(2000):
            x1, x2 = rng.uniform(-1, 1), rng.uniform(-1, 1)
            rows.append({"x1": x1, "x2": x2,
                         "reward": 2.0 * x1 - 1.0 * x2 + rng.gauss(0, 0.05)})
        t.insert(rows)

        model = torch.nn.Linear(2, 1)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = torch.nn.MSELoss()

        first = last = None
        ds = ReplayIterableDataset(t, 256, num_batches=40)
        for i, batch in enumerate(ds):
            x = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in batch.rows])
            y = torch.tensor([[float(r["reward"])] for r in batch.rows])
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            # PER: resample hard examples more often.
            errors = (pred - y).abs().squeeze(1).detach()
            t.update_priorities(batch.ids, [max(float(e), 0.01) for e in errors])

            first = first if first is not None else loss.item()
            last = loss.item()
            if (i + 1) % 10 == 0:
                print(f"batch {i + 1}: loss={last:.4f}")

        print(f"final loss: {last:.4f} (first was {first:.4f})")
        store.close()


if __name__ == "__main__":
    main()
