import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _run(script):
    return subprocess.run(
        [sys.executable, str(EXAMPLES / script)],
        capture_output=True, text=True, timeout=300, check=True,
    ).stdout


def test_bandit_example_runs_and_converges():
    out = _run("bandit.py")
    assert "best arm: smart_tool" in out


def test_train_reward_model_example_runs():
    pytest.importorskip("torch")
    out = _run("train_reward_model.py")
    assert "final loss:" in out
