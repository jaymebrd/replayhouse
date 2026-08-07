"""A tiny deterministic tool-use world for recording agent trajectories.

Three task families over a small knowledge base and a calculator:
- lookup:    "What is <entity>?"                      (1 tool call)
- arithmetic:"What is a + b * c?"                     (1 tool call)
- multi_hop: "What is the <relation> of <entity>?"    (2 tool calls)

`simulate_episode` plays a scripted policy with per-family skill levels, so
recorded trajectories contain both successes and realistic failures without
any LLM involved.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

KB = {
    "mercury": "the closest planet to the sun",
    "vostok": "the coldest research station",
    "sequoia": "the tallest tree species",
    "mariana": "the deepest ocean trench",
    "sahara": "the largest hot desert",
    "capital_of_atlantis": "poseidonia",
    "capital_of_lemuria": "shambala",
    "mayor_of_poseidonia": "coral the wise",
    "mayor_of_shambala": "lotus the elder",
}

SKILL = {"lookup": 0.75, "arithmetic": 0.9, "multi_hop": 0.5}
FAMILIES = tuple(SKILL)


def kb_lookup(key: str) -> str:
    return KB.get(key.strip().lower(), "NOT FOUND")


def calculator(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/ ().")
        if not set(expression) <= allowed:
            return "ERROR: invalid characters"
        return str(eval(expression, {"__builtins__": {}}, {}))  # toy world only
    except Exception as e:
        return f"ERROR: {e}"


TOOLS = {"kb_lookup": kb_lookup, "calculator": calculator}


@dataclass
class Task:
    family: str
    question: str
    answer: str
    plan: list  # [(tool_name, tool_input), ...]


def generate_task(rng: random.Random) -> Task:
    family = rng.choice(FAMILIES)
    if family == "lookup":
        key = rng.choice(["mercury", "vostok", "sequoia", "mariana", "sahara"])
        return Task(family, f"What is {key}?", KB[key], [("kb_lookup", key)])
    if family == "arithmetic":
        a, b, c = rng.randint(2, 99), rng.randint(2, 99), rng.randint(2, 9)
        expr = f"{a} + {b} * {c}"
        return Task(family, f"What is {expr}?", str(a + b * c), [("calculator", expr)])
    realm = rng.choice(["atlantis", "lemuria"])
    capital = KB[f"capital_of_{realm}"]
    return Task(
        family,
        f"Who is the mayor of the capital of {realm}?",
        KB[f"mayor_of_{capital}"],
        [("kb_lookup", f"capital_of_{realm}"), ("kb_lookup", f"mayor_of_{capital}")],
    )


def simulate_episode(task: Task, rng: random.Random) -> dict:
    steps, ok = [], rng.random() < SKILL[task.family]
    for i, (tool, tool_input) in enumerate(task.plan):
        if not ok and i == len(task.plan) - 1:
            tool_input = tool_input + "_oops"  # realistic failure: garbled final hop
        output = TOOLS[tool](tool_input)
        steps.append({"tool": tool, "input": tool_input, "output": output})
    answer = steps[-1]["output"] if steps else ""
    reward = 1.0 if answer == task.answer else 0.0
    return {
        "task_family": task.family,
        "env_version": 1,
        "model": "simulated-v0",
        "steps": {"question": task.question, "trace": steps},
        "answer": answer,
        "reward": reward,
        "total_tokens": 40 * len(steps) + rng.randint(10, 60),
    }
