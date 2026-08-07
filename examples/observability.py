"""One store for training AND observability — the same rows, queried.

Seeds a week of synthetic trajectories (unless --db points at a real store,
e.g. one made by examples/agent/run_agent.py), then runs every query in
observability.sql and prints the results. The .sql file is Grafana-ready:
point a ClickHouse datasource at your store and paste the queries.

Run: python examples/observability.py
"""

from __future__ import annotations

import argparse
import random
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import replayhouse

SQL_FILE = Path(__file__).with_name("observability.sql")
FAMILIES = ("browse", "code", "search", "summarize")


def seed(store) -> None:
    rng = random.Random(7)
    t = store.create("trajectories", columns={
        "task_family": "LowCardinality(String)", "env_version": "UInt32",
        "model": "LowCardinality(String)", "steps": "JSON",
        "answer": "String", "reward": "Float32", "total_tokens": "UInt32",
    })
    rows = []
    for day in range(7):
        for fam in FAMILIES:
            drift = 0.04 * day if fam != "browse" else -0.02 * day  # one regressing family
            for _ in range(18):
                r = 1.0 if rng.random() < 0.5 + drift else 0.0
                rows.append({
                    "task_family": fam, "env_version": 1, "model": "demo-v0",
                    "steps": {"trace": []}, "answer": "x", "reward": r,
                    "total_tokens": rng.randint(200, 3000),
                    "inserted_at": (datetime.now() - timedelta(days=6 - day)).strftime("%Y-%m-%d 12:00:00"),
                    "priority": max(r, 0.1),
                })
    t.insert(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=None, help="existing store; default seeds a demo")
    args = p.parse_args(argv)

    queries = [q.strip() for q in SQL_FILE.read_text().split(";") if q.strip()]
    with tempfile.TemporaryDirectory() as tmp:
        store = replayhouse.connect(f"chdb://{args.db or tmp + '/db'}")
        if args.db is None:
            seed(store)
        for q in queries:
            header = q.splitlines()[0].lstrip("- ")
            print(f"\n== {header}")
            # Demo shortcut via the private backend - the .sql file is the deliverable;
            # point Grafana (or any client) at your store for real dashboards.
            for row in store._backend.query_rows(q):
                print("  " + "  ".join(f"{k}={v}" for k, v in row.items()))
        print(f"\nran {len(queries)} queries")
        store.close()


if __name__ == "__main__":
    main()
