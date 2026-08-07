"""Record agent episodes into a ReplayHouse store.

Default: a deterministic simulated agent (no network, no keys).
--live:  a real Claude tool-use agent (requires `pip install anthropic`
         and ANTHROPIC_API_KEY). Same schema either way.

Run: python examples/agent/run_agent.py --episodes 50
"""

from __future__ import annotations

import argparse
import random
import sys

import replayhouse

from . import toyworld

COLUMNS = {
    "task_family": "LowCardinality(String)",
    "env_version": "UInt32",
    "model": "LowCardinality(String)",
    "steps": "JSON",
    "answer": "String",
    "reward": "Float32",
    "total_tokens": "UInt32",
}

TOOL_DEFS = [
    {
        "name": "kb_lookup",
        "description": "Look up an entity in the knowledge base. Call this for any "
                       "question about an entity; input is the entity key.",
        "input_schema": {"type": "object", "properties": {"key": {"type": "string"}},
                          "required": ["key"]},
    },
    {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression. Call this for any "
                       "arithmetic; input is the expression string.",
        "input_schema": {"type": "object",
                          "properties": {"expression": {"type": "string"}},
                          "required": ["expression"]},
    },
]


def run_live_episode(client, task) -> dict:
    # Manual agentic loop rather than the SDK tool runner: we intercept every
    # step to record the trajectory, and examples avoid beta dependencies.
    messages = [{"role": "user", "content": f"{task.question} Use the tools, then "
                 f"reply with ONLY the final answer text."}]
    trace, total_tokens = [], 0
    for _ in range(6):
        response = client.messages.create(
            model="claude-opus-5", max_tokens=16000,
            tools=TOOL_DEFS, messages=messages,
        )
        total_tokens += response.usage.input_tokens + response.usage.output_tokens
        if response.stop_reason == "refusal":
            return _episode(task, trace, "", total_tokens, model=response.model)
        if response.stop_reason != "tool_use":
            text = next((b.text for b in response.content if b.type == "text"), "")
            return _episode(task, trace, text.strip(), total_tokens,
                            model=response.model)
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                arg = next(iter(block.input.values()), "")
                output = toyworld.TOOLS[block.name](str(arg))
                trace.append({"tool": block.name, "input": str(arg), "output": output})
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": output})
        messages.append({"role": "user", "content": results})
    return _episode(task, trace, "", total_tokens, model="claude-opus-5")


def _episode(task, trace, answer, total_tokens, model) -> dict:
    return {
        "task_family": task.family, "env_version": 1, "model": model,
        "steps": {"question": task.question, "trace": trace},
        "answer": answer, "reward": 1.0 if answer == task.answer else 0.0,
        "total_tokens": total_tokens,
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="./replay_demo_db")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--live", action="store_true")
    args = p.parse_args(argv)

    client = None
    if args.live:
        try:
            import anthropic
        except ImportError:
            sys.exit("live mode needs the SDK: pip install anthropic")
        client = anthropic.Anthropic()

    rng = random.Random(args.seed)
    store = replayhouse.connect(f"chdb://{args.db}")
    t = store.create("trajectories", columns=COLUMNS, exists_ok=True)

    rows = []
    for _ in range(args.episodes):
        task = toyworld.generate_task(rng)
        row = run_live_episode(client, task) if args.live \
            else toyworld.simulate_episode(task, rng)
        row["priority"] = max(row["reward"], 0.1)  # failures stay sampleable
        rows.append(row)
    t.insert(rows)
    mean = sum(r["reward"] for r in rows) / len(rows)
    print(f"recorded {len(rows)} episodes (mean reward {mean:.2f})")
    store.close()


if __name__ == "__main__":
    main()
