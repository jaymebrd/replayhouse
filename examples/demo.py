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


def render(state, width: int = 72) -> str:
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
        # Note: stats read post-update priorities, so sampled_mean_p reflects each row's fresh |error|.
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
                try:
                    run_tty(engine, args.steps)
                except ImportError:
                    print("no termios here - falling back to headless")
                    run_headless(engine, args.steps)
        finally:
            engine.close()


if __name__ == "__main__":
    main()
