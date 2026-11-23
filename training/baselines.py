"""Reference policies that bound the learning problem: random and a hand-coded expert.

Run with `uv run python training/baselines.py`. Results are written to
`logs/baselines.json` and cited in the report so the comparison is reproducible.
"""

import json

import numpy as np

from environment.custom_env import (CHARGE, MOVE_E, MOVE_N, MOVE_S, MOVE_W, SCAN,
                                    STRESS_THRESHOLD, TREAT, AgriScoutEnv)
from training.common import LOGS_DIR

EVAL_SEED0, N_EPISODES = 10_000, 20


def expert_action(env):
    # Hand-coded strategy: treat what is known, recharge when low, else scan or approach.
    r, c = env.pos
    if 0 <= env.known[r, c] < STRESS_THRESHOLD:
        return TREAT
    if env.battery < 25:
        if (r, c) == env.base:
            return CHARGE
        return MOVE_N if r > 0 else MOVE_W
    if env.known[r, c] < 0:
        return SCAN
    known_stressed = np.argwhere((env.known >= 0) & (env.known < STRESS_THRESHOLD))
    unknown = np.argwhere(env.known < 0)
    targets = known_stressed if len(known_stressed) else unknown
    if not len(targets):
        return CHARGE if (r, c) == env.base else (MOVE_N if r > 0 else MOVE_W)
    tr, tc = targets[np.abs(targets - [r, c]).sum(axis=1).argmin()]
    if tr < r:
        return MOVE_N
    if tr > r:
        return MOVE_S
    if tc > c:
        return MOVE_E
    return MOVE_W


def run_policy(name, n_stressed=5, seed0=EVAL_SEED0):
    # Score one reference policy over the held-out evaluation seeds.
    returns, cleared, treated, wasted = [], 0, 0, 0
    for i in range(N_EPISODES):
        env = AgriScoutEnv(n_stressed=n_stressed)
        env.reset(seed=seed0 + i)
        total, done = 0.0, False
        while not done:
            action = env.action_space.sample() if name == "random" else expert_action(env)
            _, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
        cleared += info["stressed_left"] == 0
        treated += info["treated_ok"]
        wasted += info["treated_bad"]
    return {
        "mean": round(float(np.mean(returns)), 2), "std": round(float(np.std(returns)), 2),
        "farms_cleared": f"{cleared}/{N_EPISODES}",
        "treated_ok": treated, "treated_wasted": wasted,
    }


def main():
    # Evaluate both reference policies and store them for the report.
    results = {}
    for name in ("random", "expert_heuristic"):
        results[name] = run_policy(name)
        results[name]["generalization_mean"] = run_policy(name, n_stressed=8, seed0=20_000)["mean"]
        r = results[name]
        print(f"{name:18s} mean {r['mean']:8.1f} ± {r['std']:5.1f} | farms cleared {r['farms_cleared']} "
              f"| treated {r['treated_ok']} ok / {r['treated_wasted']} wasted | unseen regime {r['generalization_mean']}")
    LOGS_DIR.mkdir(exist_ok=True)
    (LOGS_DIR / "baselines.json").write_text(json.dumps(results, indent=2))
    print(f"Saved to {LOGS_DIR / 'baselines.json'}")


if __name__ == "__main__":
    main()
