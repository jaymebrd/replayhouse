"""A GRPO-shaped loop with ReplayHouse as the rollout buffer.

Shape matches LLM RL practice: G rollouts per prompt, group-relative
advantages, advantage-weighted resampling from the store, policy update,
priorities refreshed from |advantage|. The "policy" is a tiny softmax net
and the "environment" a contextual bandit, so it runs offline in seconds —
swap those two pieces for a model and rollout workers and the plumbing is
unchanged.

Rollouts are seeded, but the store's weighted draws use ClickHouse-side randomness,
so exact reward numbers vary slightly between runs. The learning-curve margin in
the smoke test accounts for this.

Run: python examples/grpo_loop.py   (needs replayhouse[torch])
"""

from __future__ import annotations

import argparse
import tempfile

import torch

import replayhouse

N_PROMPTS, N_ACTIONS, GROUP, DIM = 8, 4, 8, 6


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    contexts = torch.randn(N_PROMPTS, DIM)
    true_w = torch.randn(DIM, N_ACTIONS)
    best = (contexts @ true_w).argmax(dim=1)          # hidden optimal action

    policy = torch.nn.Linear(DIM, N_ACTIONS)
    opt = torch.optim.Adam(policy.parameters(), lr=0.05)

    with tempfile.TemporaryDirectory() as tmp:
        store = replayhouse.connect(f"chdb://{tmp}/db")
        t = store.create("rollouts", columns={
            "prompt_id": "UInt32", "action": "UInt8",
            "reward": "Float32", "advantage": "Float32",
        })

        first = None
        for round_ in range(args.rounds):
            rows, round_rewards = [], []
            for pid in range(N_PROMPTS):
                logits = policy(contexts[pid])
                dist = torch.distributions.Categorical(logits=logits)
                acts = dist.sample((GROUP,))
                rewards = [1.0 if a == best[pid] else 0.1 for a in acts]
                mean_r = sum(rewards) / GROUP
                round_rewards += rewards
                for a, r in zip(acts.tolist(), rewards):
                    rows.append({"prompt_id": pid, "action": a, "reward": r,
                                 "advantage": r - mean_r,
                                 "priority": max(abs(r - mean_r), 0.01)})
            t.insert(rows)

            batch = t.sample(256, by="exp(2 * advantage)")   # favor above-group-mean
            ctx = contexts[[int(r["prompt_id"]) for r in batch.rows]]
            act = torch.tensor([int(r["action"]) for r in batch.rows])
            adv = torch.tensor([float(r["advantage"]) for r in batch.rows])
            logp = torch.distributions.Categorical(
                logits=policy(ctx)).log_prob(act)
            loss = -(logp * adv).mean()
            opt.zero_grad(); loss.backward(); opt.step()

            # Priorities feed eviction and by="priority" consumers. This loop samples
            # by advantage directly, so the refresh here shows the pattern, not the math.
            t.update_priorities(batch.ids,
                                [max(abs(float(r["advantage"])), 0.01)
                                 for r in batch.rows])

            mean_reward = sum(round_rewards) / len(round_rewards)
            if first is None:
                first = mean_reward
                print(f"first mean reward: {first:.3f}")
            if (round_ + 1) % 10 == 0:
                print(f"round {round_ + 1}: mean reward {mean_reward:.3f}")
        print(f"final mean reward: {mean_reward:.3f}")
        store.close()


if __name__ == "__main__":
    main()
