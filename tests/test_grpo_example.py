import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_grpo_loop_learns():
    pytest.importorskip("torch")
    out = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "grpo_loop.py"), "--rounds", "30"],
        capture_output=True, text=True, timeout=600, check=True,
    ).stdout
    first = float(out.split("first mean reward: ")[1].split("\n")[0])
    final = float(out.split("final mean reward: ")[1].split("\n")[0])
    assert final > first + 0.15  # policy actually improved via replayed advantages
