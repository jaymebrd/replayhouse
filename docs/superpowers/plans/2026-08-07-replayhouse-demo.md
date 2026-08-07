# ReplayHouse Interactive Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `examples/demo.py` — a terminal animation of prioritized experience replay where every displayed number comes from a real chdb store: live priority histogram, loss sparkline, sampling-attention stats, and a keyboard toggle between prioritized and uniform sampling.

**Architecture:** One file, two layers. The **engine** runs the honest PER loop (torch linear model on synthetic 2D data; `sample(by="priority")` or `by="1"` for uniform; `update_priorities(|error|)`; store stats via `store.query`). The **renderer** is a pure function `render(state) -> str` (ANSI, stdlib-only) driven by a loop with raw-mode keyboard input (`termios`/`select`); non-TTY stdout automatically falls back to `--headless` line output, which is also what CI smoke-tests. No new dependencies (torch via the existing extra; TUI is pure stdlib).

**Tech Stack:** Python ≥ 3.10, replayhouse (main, 87 tests), torch (existing extra), stdlib `termios`/`tty`/`select` (guarded for non-POSIX).

## Global Constraints

- Branch `replayhouse-demo` off `main`; commit style per task; `.venv/bin/pytest tests -q` green before each commit (87 baseline).
- No new dependencies; torch imported at module top with a friendly `sys.exit` message if missing (same pattern as live-mode anthropic guidance: instruct `pip install 'replayhouse[torch]'`).
- Every displayed statistic must come from the real store or the real training step — no synthesized display values. The priority histogram is computed from `store.query` over the sidecar (`argMax(priority, version) GROUP BY id`), not from a Python-side cache.
- Uniform mode uses `by="1"` (constant weight, strictly positive → valid; exercises the engine's mode-B path). Prioritized mode uses the default `by="priority"`.
- `render(state) -> str` is a pure function (no I/O) so it is unit-testable without a TTY.
- Non-TTY stdout (CI, pipes) or `--headless` runs the same engine, printing one line per step and a final `demo complete: loss <first:.4f> -> <last:.4f>` summary; smoke tests use this path with `pytest.importorskip("torch")`, subprocess timeout 600.
- Keyboard handling is POSIX-only (`termios`); on ImportError fall back to headless with a notice (keeps the file importable on Windows).

## File Structure

```
examples/demo.py            # engine + pure renderer + TTY loop + headless mode
tests/test_demo_example.py  # headless smoke test + pure-renderer unit tests
README.md                   # one bullet under Examples
```

---

### Task 1: Engine + headless mode

**Files:**
- Create: `examples/demo.py`
- Test: `tests/test_demo_example.py`

**Interfaces:**
- Consumes: `replayhouse.connect/create/insert/sample/update_priorities`, `store.query`.
- Produces (Task 2 relies on these exact names):
  - `@dataclass DemoState`: fields `step: int`, `mode: str` (`"prioritized"`|`"uniform"`), `losses: list[float]`, `hist: list[int]` (10 bins), `hist_edges: tuple[float, float]`, `sampled_mean_p: float`, `pop_mean_p: float`, `top_decile_share: float`, `paused: bool`.
  - `class DemoEngine`: `__init__(db_path: str, n_rows: int = 2000, batch: int = 256, seed: int = 0)`, `.step() -> DemoState` (one train step + fresh store stats), `.toggle_mode() -> None`, `.close() -> None`.
  - `run_headless(engine, steps: int) -> tuple[float, float]` returning `(first_loss, last_loss)`.
  - CLI: `python examples/demo.py [--steps N] [--headless] [--batch N]`.

- [ ] **Step 1: Write `examples/demo.py` (engine + headless; TTY loop arrives in Task 2)**

```python
"""Watch prioritized experience replay happen — against a real store.

A tiny torch model learns y = 2*x1 - x2 from a ReplayHouse store. Every
frame is real: batches are weighted draws from chdb, per-sample errors go
back as priorities, and the histogram is a live query over the priority
sidecar. Press [u] to switch to uniform sampling and watch the difference
(interactive mode arrives with the TTY loop; --headless prints lines).

Run: python examples/demo.py            (needs replayhouse[torch])
     python examples/demo.py --headless --steps 100
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass, field

try:
    import torch
except ImportError:
    sys.exit("the demo needs torch: pip install 'replayhouse[torch]'")

import replayhouse

TRUE_W = (2.0, -1.0)
BINS = 10


@dataclass
class DemoState:
    step: int = 0
    mode: str = "prioritized"
    losses: list = field(default_factory=list)
    hist: list = field(default_factory=lambda: [0] * BINS)
    hist_edges: tuple = (0.0, 1.0)
    sampled_mean_p: float = 0.0
    pop_mean_p: float = 0.0
    top_decile_share: float = 0.0
    paused: bool = False


class DemoEngine:
    def __init__(self, db_path: str, n_rows: int = 2000, batch: int = 256,
                 seed: int = 0):
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        self._store = replayhouse.connect(f"chdb://{db_path}")
        self._t = self._store.create(
            "demo", columns={"x1": "Float32", "x2": "Float32", "y": "Float32"})
        xs = torch.rand(n_rows, 2, generator=gen) * 2 - 1
        noise = torch.randn(n_rows, 1, generator=gen) * 0.05
        ys = xs[:, :1] * TRUE_W[0] + xs[:, 1:] * TRUE_W[1] + noise
        self._t.insert([
            {"x1": float(a), "x2": float(b), "y": float(c)}
            for (a, b), c in zip(xs.tolist(), ys.squeeze(1).tolist())
        ])
        self._model = torch.nn.Linear(2, 1)
        self._opt = torch.optim.SGD(self._model.parameters(), lr=0.05)
        self._batch = batch
        self.state = DemoState()

    def toggle_mode(self) -> None:
        self.state.mode = ("uniform" if self.state.mode == "prioritized"
                           else "prioritized")

    def _store_stats(self, sampled_ids: set) -> None:
        rows = self._store.query(
            "SELECT id, argMax(priority, version) AS p "
            "FROM demo__priorities GROUP BY id")
        ps = {r["id"]: float(r["p"]) for r in rows}
        values = sorted(ps.values())
        lo, hi = values[0], max(values[-1], values[0] + 1e-9)
        hist = [0] * BINS
        for v in values:
            hist[min(BINS - 1, int((v - lo) / (hi - lo) * BINS))] += 1
        s = self.state
        s.hist, s.hist_edges = hist, (lo, hi)
        s.pop_mean_p = sum(values) / len(values)
        sampled = [ps[i] for i in sampled_ids if i in ps]
        s.sampled_mean_p = sum(sampled) / max(len(sampled), 1)
        decile_cut = values[int(len(values) * 0.9)]
        s.top_decile_share = (sum(1 for v in sampled if v >= decile_cut)
                              / max(len(sampled), 1))

    def step(self) -> DemoState:
        by = "priority" if self.state.mode == "prioritized" else "1"
        b = self._t.sample(self._batch, by=by)
        x = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in b.rows])
        y = torch.tensor([[float(r["y"])] for r in b.rows])
        pred = self._model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        self._opt.zero_grad(); loss.backward(); self._opt.step()
        errors = (pred - y).abs().squeeze(1).detach()
        self._t.update_priorities(b.ids, [max(float(e), 0.01) for e in errors])
        self.state.step += 1
        self.state.losses.append(float(loss))
        self._store_stats(set(b.ids))
        return self.state

    def close(self) -> None:
        self._store.close()


def run_headless(engine: DemoEngine, steps: int) -> tuple:
    first = None
    for _ in range(steps):
        s = engine.step()
        first = first if first is not None else s.losses[-1]
        print(f"step {s.step:>4}  loss {s.losses[-1]:.4f}  "
              f"sampled_mean_p {s.sampled_mean_p:.3f}  "
              f"pop_mean_p {s.pop_mean_p:.3f}  "
              f"top_decile_share {s.top_decile_share:.2f}", flush=True)
    print(f"demo complete: loss {first:.4f} -> {s.losses[-1]:.4f}")
    return first, s.losses[-1]


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--headless", action="store_true")
    args = p.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        engine = DemoEngine(f"{tmp}/db", batch=args.batch)
        try:
            if args.headless or not sys.stdout.isatty():
                run_headless(engine, args.steps)
            else:
                run_headless(engine, args.steps)  # TTY loop replaces this in Task 2
        finally:
            engine.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the failing tests**

`tests/test_demo_example.py`:

```python
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
         "--headless", "--steps", "40"],
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
```

- [ ] **Step 3: Run tests to verify they fail, then pass**

Run: `.venv/bin/pytest tests/test_demo_example.py -v`
Expected: FAIL first (no file), then after Step 1 lands: 2 passed. The migration ratio threshold 1.1 is generous — with weighted sampling the sampled-mean is mathematically the priority-squared mean over the priority mean and sits well above the population mean once errors spread; if it flakes, raise `--steps`, never lower the ratio.

- [ ] **Step 4: Full suite and commit**

Run: `.venv/bin/pytest tests -q` (89 expected)

```bash
git add examples/demo.py tests/test_demo_example.py
git commit -m "feat: PER demo engine with headless mode - every stat from the real store"
```

---

### Task 2: ANSI renderer + TTY loop + README bullet

**Files:**
- Modify: `examples/demo.py`, `README.md`
- Test: `tests/test_demo_example.py` (add renderer unit tests)

**Interfaces:**
- Consumes: `DemoState`, `DemoEngine`, `run_headless` from Task 1 (exact names above).
- Produces: `render(state: DemoState, width: int = 72) -> str` (pure; ANSI escapes allowed inside, no I/O); `run_tty(engine, steps)` (raw-mode keyboard: space=pause, u=toggle mode, q=quit; ~8 fps via `select` timeout 0.12s); `main` routes TTY→`run_tty`, non-TTY/`--headless`→`run_headless`, `termios` ImportError→headless with a printed notice.

- [ ] **Step 1: Add the renderer and TTY loop to `examples/demo.py`**

```python
SPARK = "▁▂▃▄▅▆▇█"
CLEAR = "\x1b[2J\x1b[H"
DIM, BOLD, RESET = "\x1b[2m", "\x1b[1m", "\x1b[0m"
BAR_COLOR = "\x1b[38;5;208m"   # single accent; degrades fine on 16-color


def _spark(values, width):
    tail = values[-width:]
    if not tail:
        return ""
    lo, hi = min(tail), max(max(tail), min(tail) + 1e-9)
    return "".join(SPARK[int((v - lo) / (hi - lo) * (len(SPARK) - 1))]
                   for v in tail)


def render(state: DemoState, width: int = 72) -> str:
    s = state
    out = [CLEAR + BOLD + "ReplayHouse: prioritized replay, live from chdb"
           + RESET]
    mode_note = ("sampling ∝ priority" if s.mode == "prioritized"
                 else "sampling uniform (press u to re-prioritize)")
    out.append(f"step {s.step:<6} mode {BOLD}{s.mode}{RESET} ({mode_note})"
               f"{'   [paused]' if s.paused else ''}")
    out.append("")
    loss = s.losses[-1] if s.losses else float("nan")
    out.append(f"loss {loss:8.4f}  {DIM}{_spark(s.losses, width - 16)}{RESET}")
    out.append("")
    out.append(f"priority histogram {DIM}(live query over demo__priorities; "
               f"range {s.hist_edges[0]:.2f}–{s.hist_edges[1]:.2f}){RESET}")
    peak = max(max(s.hist), 1)
    for i, count in enumerate(s.hist):
        bar = BAR_COLOR + "█" * int(count / peak * (width - 22)) + RESET
        out.append(f"  bin {i:<2} {count:>5} {bar}")
    out.append("")
    ratio = s.sampled_mean_p / max(s.pop_mean_p, 1e-9)
    out.append(f"sampled-batch mean priority {s.sampled_mean_p:.3f} vs "
               f"population {s.pop_mean_p:.3f}  ({BOLD}{ratio:.2f}x{RESET})")
    out.append(f"share of batch from top-decile priority: "
               f"{BOLD}{s.top_decile_share:.0%}{RESET}")
    out.append("")
    out.append(f"{DIM}[space] pause   [u] uniform/prioritized   [q] quit{RESET}")
    return "\n".join(out)


def run_tty(engine: DemoEngine, steps: int) -> None:
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while engine.state.step < steps:
            if not engine.state.paused:
                engine.step()
            sys.stdout.write(render(engine.state))
            sys.stdout.flush()
            r, _, _ = select.select([sys.stdin], [], [], 0.12)
            if r:
                key = sys.stdin.read(1)
                if key == "q":
                    break
                if key == " ":
                    engine.state.paused = not engine.state.paused
                if key == "u":
                    engine.toggle_mode()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print()
```

and route in `main` (replacing the Task 1 placeholder):

```python
        try:
            if args.headless or not sys.stdout.isatty():
                run_headless(engine, args.steps)
            else:
                try:
                    run_tty(engine, args.steps)
                except ImportError:
                    print("no termios here - falling back to headless")
                    run_headless(engine, args.steps)
        finally:
            engine.close()
```

- [ ] **Step 2: Add renderer unit tests**

Append to `tests/test_demo_example.py`:

```python
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
```

- [ ] **Step 3: Run the tests**

Run: `.venv/bin/pytest tests/test_demo_example.py -v`
Expected: 4 passed. (The loss token check uses `"0.25"` — keep the assertion as written.)

- [ ] **Step 4: Add the README bullet**

In the `## Examples {#examples}` section, add as the FIRST bullet:

```markdown
- [`examples/demo.py`](examples/demo.py) — watch prioritized replay happen:
  a live terminal animation where the histogram is a real query over the
  priority sidecar and `[u]` flips between prioritized and uniform sampling.
```

- [ ] **Step 5: Full suite and commit**

Run: `.venv/bin/pytest tests -q` (91 expected)

```bash
git add examples/demo.py tests/test_demo_example.py README.md
git commit -m "feat: ANSI TTY renderer and keyboard controls for the PER demo"
```

---

## Self-Review Notes

- **Honesty constraint held:** histogram/means/decile all derive from `store.query` over the sidecar per frame; batch and priorities go through the public API; nothing rendered is synthesized. Uniform mode via `by="1"` is a real engine path (mode B, constant positive weight).
- **Type consistency:** `DemoState` fields referenced by `render` match Task 1's dataclass exactly; `run_headless` signature shared.
- **Testability without TTY:** `render` is pure (asserted by double-render equality); TTY loop is not CI-tested (no pty), accepted and stated — the engine and renderer carry the coverage.
- **Placeholder scan:** clean; the Task 1 `main` explicitly marks the line Task 2 replaces.
