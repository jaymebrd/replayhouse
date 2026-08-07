import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_observability_runs_all_queries():
    out = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "observability.py")],
        capture_output=True, text=True, timeout=600, check=True,
    ).stdout
    assert "ran 6 queries" in out
    for header in ("Reward by day", "Worst 10 trajectories", "Token spend",
                   "Success rate", "priority distribution", "Store size"):
        assert header in out
    # Seed dates are relative to today, so the "last day" stat should never
    # go stale to NULL/None regardless of when this test runs.
    assert "last_day=None" not in out
