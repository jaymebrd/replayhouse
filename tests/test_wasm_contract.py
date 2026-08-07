import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.skipif(not (ROOT / "web" / "node_modules" / "chdb-wasm").exists(),
                    reason="run `npm --prefix web install` first")
def test_js_contract_against_wasm_engine():
    r = subprocess.run(["npm", "--prefix", str(ROOT / "web"), "test"],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
