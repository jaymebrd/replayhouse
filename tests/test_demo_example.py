import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_demo_headless_learns():
    pytest.importorskip("torch")
    out = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "demo.py"),
         "--headless", "--steps", "60"],
        capture_output=True, text=True, timeout=600, check=True,
    ).stdout
    assert "demo complete: loss " in out
    first, last = out.split("demo complete: loss ")[1].split(" -> ")
    assert float(last) < float(first)


def test_demo_attention_migrates_to_hard_examples():
    pytest.importorskip("torch")
    out = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "demo.py"),
         "--headless", "--steps", "80"],
        capture_output=True, text=True, timeout=600, check=True,
    ).stdout
    lines = [l for l in out.splitlines() if l.startswith("step")]
    late = lines[-5:]
    # In prioritized mode the sampled batch's mean priority should exceed the
    # population mean once errors differentiate (that IS the mechanism).
    ratios = []
    for l in late:
        sampled = float(l.split("sampled_mean_p ")[1].split()[0])
        pop = float(l.split("pop_mean_p ")[1].split()[0])
        ratios.append(sampled / max(pop, 1e-9))
    assert sum(ratios) / len(ratios) > 1.1


def test_render_is_pure_and_complete():
    pytest.importorskip("torch")
    from examples.demo import DemoState, render

    s = DemoState(step=7, mode="prioritized", losses=[1.0, 0.5, 0.25],
                  hist=[5, 3, 2, 0, 0, 0, 0, 0, 0, 1],
                  hist_edges=(0.01, 2.0), sampled_mean_p=0.9,
                  pop_mean_p=0.5, top_decile_share=0.4)
    frame = render(s)
    for token in ("step 7", "prioritized", "0.2500"[:5], "bin 0", "1.80x",
                  "40%", "[space]"):
        assert token in frame, token
    assert frame == render(s)  # pure: same state, same frame


def test_render_uniform_mode_labels():
    pytest.importorskip("torch")
    from examples.demo import DemoState, render

    frame = render(DemoState(mode="uniform", losses=[1.0]))
    assert "uniform" in frame and "re-prioritize" in frame
