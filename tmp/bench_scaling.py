"""Scaling probe: does phase-1 sampling need a sub-linear structure?

Populates stores of increasing size via bulk INSERT SELECT (bypassing the
Python insert path — we're measuring the sampling/update loop, not ingest),
then times the operations a trainer actually performs.
"""

import shutil
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replayhouse import connect
from replayhouse.sampling import phase1_sql

SIZES = [1_000_000, 10_000_000, 50_000_000]
K = 8192
DB_ROOT = Path(__file__).resolve().parent / "bench_dbs"


def timed(fn, repeat=5):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def bench(n):
    path = DB_ROOT / f"db_{n}"
    if path.exists():
        shutil.rmtree(path)
    store = connect(f"chdb://{path}")
    t = store.create(
        "bench",
        columns={
            "task_family": "LowCardinality(String)",
            "env_version": "UInt32",
            "steps": "JSON",
            "reward": "Float32",
        },
    )
    be = store._backend

    t0 = time.perf_counter()
    be.command(f"""
        INSERT INTO bench (id, inserted_at, task_family, env_version, steps, reward)
        SELECT generateUUIDv7(),
               now() - toIntervalSecond(rand() % 864000),
               concat('fam_', toString(number % 10)),
               1 + toUInt32(number % 5),
               CAST('{{"i":1}}' AS JSON),
               toFloat32(randCanonical())
        FROM numbers({n})
    """)
    be.command("""
        INSERT INTO bench__priorities
        SELECT id, toFloat32(0.01 + pow(randCanonical(), 3) * 10), 1 FROM bench
    """)
    ingest_s = time.perf_counter() - t0

    r = {"n": n, "ingest_s": ingest_s}

    r["phase1_ms"] = timed(lambda: be.query_rows(phase1_sql("bench", K))) * 1000
    r["sample_ms"] = timed(lambda: t.sample(K)) * 1000
    r["sample_where_ms"] = timed(lambda: t.sample(K, where="env_version >= 3"), 3) * 1000
    r["sample_by_main_ms"] = timed(lambda: t.sample(K, by="reward"), 3) * 1000

    batch = t.sample(K)
    r["update_8k_ms"] = timed(lambda: t.update_priorities(batch.ids, [1.0] * len(batch.ids)), 3) * 1000

    # Sidecar bloat: 3x versions per id, then compare and compact.
    be.command("INSERT INTO bench__priorities SELECT id, priority, 2 FROM bench__priorities WHERE version = 1")
    be.command("INSERT INTO bench__priorities SELECT id, priority, 3 FROM bench__priorities WHERE version = 1")
    r["sample_bloat3x_ms"] = timed(lambda: t.sample(K), 3) * 1000
    t0 = time.perf_counter()
    t.compact()
    r["compact_s"] = time.perf_counter() - t0
    r["sample_postcompact_ms"] = timed(lambda: t.sample(K), 3) * 1000

    store.close()
    shutil.rmtree(path, ignore_errors=True)
    return r


def main():
    DB_ROOT.mkdir(exist_ok=True)
    cols = ["n", "ingest_s", "phase1_ms", "sample_ms", "sample_where_ms",
            "sample_by_main_ms", "update_8k_ms", "sample_bloat3x_ms",
            "compact_s", "sample_postcompact_ms"]
    print("\t".join(cols), flush=True)
    for n in SIZES:
        r = bench(n)
        print("\t".join(f"{r[c]:.1f}" if isinstance(r[c], float) else str(r[c]) for c in cols), flush=True)


if __name__ == "__main__":
    main()
