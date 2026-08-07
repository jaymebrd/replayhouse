import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def _run(args, cwd):
    return subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True,
                          text=True, timeout=600, check=True).stdout


def test_record_then_curate_roundtrip(tmp_path):
    db = tmp_path / "db"
    out = tmp_path / "curated.parquet"
    rec = _run(["-m", "examples.agent.run_agent", "--db", str(db),
                "--episodes", "60", "--seed", "1"], cwd=ROOT)
    assert "recorded 60 episodes" in rec

    cur = _run(["-m", "examples.agent.curate", "--db", str(db),
                "--out", str(out), "--k", "30"], cwd=ROOT)
    assert "exported 30 rows" in cur
    assert all(f"  {fam}: 10" in cur for fam in ("arithmetic", "lookup", "multi_hop"))

    table = pq.read_table(out)
    assert table.num_rows == 30
    assert {"task_family", "steps", "reward", "answer"} <= set(table.column_names)


def test_simulator_reward_signal_is_mixed(tmp_path):
    rec = _run(["-m", "examples.agent.run_agent", "--db", str(tmp_path / "db2"),
                "--episodes", "80", "--seed", "2"], cwd=ROOT)
    mean = float(rec.split("mean reward ")[1].rstrip(")\n"))
    assert 0.4 < mean < 0.95  # skill levels produce a mixed signal, not all-success
