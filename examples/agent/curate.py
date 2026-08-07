"""Build a curated fine-tuning set from recorded trajectories.

The pitch in one query: a weighted, without-replacement draw fused with SQL
filters and stratification, exported straight to Parquet.

Run: python examples/agent/curate.py --db ./replay_demo_db
"""

from __future__ import annotations

import argparse
from collections import Counter

import pyarrow.parquet as pq

import replayhouse


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="./replay_demo_db")
    p.add_argument("--out", default="curated.parquet")
    p.add_argument("--k", type=int, default=200)
    args = p.parse_args(argv)

    store = replayhouse.connect(f"chdb://{args.db}")
    t = store.table("trajectories")

    batch = t.sample(
        args.k,
        by="reward + 0.1",                # weight toward successes, keep failures
        where="env_version >= 1 AND total_tokens < 2000",
        stratify_by="task_family",        # even coverage across families
    )
    fams = Counter(r["task_family"] for r in batch.rows)
    mean = sum(float(r["reward"]) for r in batch.rows) / max(len(batch), 1)
    print(f"sampled {len(batch)} trajectories, mean reward {mean:.2f}")
    for fam, n in sorted(fams.items()):
        print(f"  {fam}: {n}")

    pq.write_table(batch.to_arrow(), args.out)
    print(f"exported {len(batch)} rows to {args.out}")
    store.close()


if __name__ == "__main__":
    main()
