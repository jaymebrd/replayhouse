"""3-arm Thompson-style bandit on embedded chdb. No torch required.

Run: python examples/bandit.py
"""

import random
import tempfile

import replayhouse


def main():
    with tempfile.TemporaryDirectory() as tmp:
        store = replayhouse.connect(f"chdb://{tmp}/db")
        arms = store.create("arms", columns={"arm": "LowCardinality(String)"})

        true_reward = {"slow_tool": 0.2, "fast_tool": 0.5, "smart_tool": 0.9}
        ids = {a: arms.insert([{"arm": a}])[0] for a in true_reward}
        wins = {a: 1.0 for a in true_reward}
        pulls = {a: 1 for a in true_reward}
        rng = random.Random(0)

        for step in range(500):
            arm = arms.sample(1).rows[0]["arm"]
            pulls[arm] += 1
            wins[arm] += 1.0 if rng.random() < true_reward[arm] else 0.0
            arms.update_priorities([ids[arm]], [max(wins[arm] / pulls[arm], 0.01)])
            if (step + 1) % 100 == 0:
                print(f"step {step + 1}: pulls={pulls}")

        best = max(pulls, key=pulls.get)
        print(f"best arm: {best} ({pulls[best]} pulls)")
        store.close()


if __name__ == "__main__":
    main()
