"""Sidecar dedup variants: is argMaxState (or FINAL) worth a schema change?

The scaling probe showed phase-1 sampling cost is dominated by the
argMax-GROUP-BY dedup over the priority sidecar. This measures four ways to
read "current priority per id" at 10M/50M ids, each in a compacted state
and a 3x-bloated state (two extra priority versions per id):

  A repl+argMax   current design: ReplacingMergeTree, argMax GROUP BY at read
  B agg+argMaxMerge  AggregatingMergeTree sidecar, states merged in background
  C repl+FINAL    ReplacingMergeTree read with FINAL (merge-sorted dedup)
  D repl raw      no dedup at all on a compacted table (theoretical floor)

Run: .venv/bin/python benchmarks/bench_sidecar_variants.py
"""

from __future__ import annotations

import shutil
import statistics
import time
from pathlib import Path

import replayhouse

SIZES = [10_000_000, 50_000_000]
K = 8192
DB_ROOT = Path(__file__).resolve().parent.parent / "tmp" / "sidecar_bench"

KEY = "-log(1 - randCanonical())"

QUERIES = {
    "A repl+argMax": f"""
        SELECT id FROM (
            SELECT id, argMax(priority, version) AS p
            FROM side_repl GROUP BY id
        ) WHERE p > 0 ORDER BY {KEY} / p LIMIT {K}""",
    "B agg+argMaxMerge": f"""
        SELECT id FROM (
            SELECT id, argMaxMerge(priority) AS p
            FROM side_agg GROUP BY id
        ) WHERE p > 0 ORDER BY {KEY} / p LIMIT {K}""",
    "C repl+FINAL": f"""
        SELECT id FROM side_repl FINAL
        WHERE priority > 0 ORDER BY {KEY} / priority LIMIT {K}""",
    "D repl raw (floor)": f"""
        SELECT id FROM side_repl
        WHERE priority > 0 ORDER BY {KEY} / priority LIMIT {K}""",
}


def timed(store, sql, repeat=5):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        store.query(sql)
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000


def bench(n):
    path = DB_ROOT / f"db_{n}"
    if path.exists():
        shutil.rmtree(path)
    store = replayhouse.connect(f"chdb://{path}")
    q = store.query  # raw SQL via the public API

    store._backend.command(
        "CREATE TABLE side_repl (id UUID, priority Float32, version UInt64) "
        "ENGINE = ReplacingMergeTree(version) ORDER BY id")
    store._backend.command(
        "CREATE TABLE side_agg (id UUID, "
        "priority AggregateFunction(argMax, Float32, UInt64)) "
        "ENGINE = AggregatingMergeTree ORDER BY id")

    store._backend.command(
        f"INSERT INTO side_repl SELECT generateUUIDv7(), "
        f"toFloat32(0.01 + pow(randCanonical(), 3) * 10), 1 FROM numbers({n})")
    store._backend.command(
        "INSERT INTO side_agg SELECT id, argMaxState(priority, version) "
        "FROM side_repl WHERE version = 1 GROUP BY id")

    results = {}

    # --- compacted state ---
    store._backend.command("OPTIMIZE TABLE side_repl FINAL")
    store._backend.command("OPTIMIZE TABLE side_agg FINAL")
    for name, sql in QUERIES.items():
        results[(name, "compact")] = timed(store, sql)

    # --- 3x bloat: two more versions per id (D is skipped: raw scan of a
    # bloated table returns duplicate ids and is not a valid variant there) ---
    for v in (2, 3):
        store._backend.command(
            f"INSERT INTO side_repl SELECT id, priority * 1.1, {v} "
            f"FROM side_repl WHERE version = 1")
        store._backend.command(
            f"INSERT INTO side_agg SELECT id, argMaxState(toFloat32(priority * 1.1), "
            f"toUInt64({v})) FROM side_repl WHERE version = 1 GROUP BY id")
    for name, sql in QUERIES.items():
        if name.startswith("D"):
            continue
        results[(name, "bloat3x")] = timed(store, sql, repeat=3)

    store.close()
    shutil.rmtree(path, ignore_errors=True)
    return results


def main():
    DB_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"{'variant':<22}{'state':<10}" + "".join(f"{n:>14,}" for n in SIZES),
          flush=True)
    all_results = {n: bench(n) for n in SIZES}
    for name in QUERIES:
        for state in ("compact", "bloat3x"):
            row = []
            for n in SIZES:
                v = all_results[n].get((name, state))
                row.append(f"{v:>13.0f}m" if v is not None else f"{'—':>14}")
            print(f"{name:<22}{state:<10}" + "".join(row), flush=True)


if __name__ == "__main__":
    main()
