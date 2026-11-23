"""AgriScout demo: run the best trained agent live in the 3D Rwanda viewer.

Usage: `uv run main.py` (auto-picks the best model across all logged runs),
or `uv run main.py --algo ppo --episodes 5`.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import QValueActor

from environment.custom_env import ACTION_NAMES, AgriScoutEnv
from environment.rendering import LiveRenderer
from training.common import LOGS_DIR, MODELS_DIR, N_ACTIONS, make_env, mlp
from training.pg_training import make_actor_critic


def find_best_run(algo_filter=None):
    # Pick the logged run with the highest evaluation mean reward.
    best = None
    for log_file in LOGS_DIR.glob("*/*.json"):
        record = json.loads(log_file.read_text())
        if algo_filter and record["algo"] != algo_filter:
            continue
        if best is None or record["eval_mean"] > best["eval_mean"]:
            best = record
    return best


def load_policy(record):
    # Rebuild the actor for the winning run and load its trained weights.
    algo, hidden = record["algo"], tuple(record["config"]["hidden"])
    env = make_env()
    if algo == "dqn":
        actor = QValueActor(mlp(N_ACTIONS, hidden), in_keys=["observation"], spec=env.action_spec)
        weights = MODELS_DIR / "dqn" / f"dqn_{record['run']}.pt"
    else:
        actor, _ = make_actor_critic(env, hidden)
        weights = MODELS_DIR / "pg" / f"{algo}_{record['run']}.pt"
    actor.load_state_dict(torch.load(weights, weights_only=True))
    actor.eval()
    env.close()
    return actor


def choose_action(actor, obs):
    # Greedy action from the trained policy for one raw observation.
    td = TensorDict({"observation": torch.as_tensor(obs)}, batch_size=[])
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        actor(td)
    return int(td["action"].argmax())


def main():
    # Launch the viewer, then run the agent and narrate every step in the terminal.
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "reinforce", "ppo", "a2c"],
                        help="restrict to one algorithm (default: best overall)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    record = find_best_run(args.algo)
    if record:
        print(f"Best model: {record['algo'].upper()} run '{record['run']}' "
              f"(eval mean reward {record['eval_mean']} ± {record['eval_std']})")
        actor = load_policy(record)
    else:
        print("No trained models found in logs/ — running a RANDOM agent.")
        print("Train first, e.g.: uv run python training/dqn_training.py")
        actor = None

    env = AgriScoutEnv()
    renderer = LiveRenderer(open_browser=not args.no_browser, fps=args.fps)
    env._renderer = renderer
    print("Viewer: http://localhost:8000 waiting for the browser to connect...")
    renderer.wait_for_client()

    for ep in range(args.episodes):
        obs, info = env.reset(seed=50_000 + ep)
        env.render()
        total, done = 0.0, False
        print(f"\n=== Episode {ep + 1} | {info['stressed_left']} stressed fields hidden ===")
        while not done:
            action = choose_action(actor, obs) if actor else env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            total += reward
            done = terminated or truncated
            print(f"  step {info['steps']:>3} | {ACTION_NAMES[action]:<10} | reward {reward:+6.1f} "
                  f"| battery {info['battery']:5.1f} | stressed left {info['stressed_left']}")
        outcome = "all stress cleared!" if info["stressed_left"] == 0 else (
            "battery died" if terminated and info["battery"] <= 0 else "time limit reached")
        print(f"=== Episode {ep + 1} done: return {total:.1f} | {outcome} "
              f"| treated OK {info['treated_ok']} | wasted sprays {info['treated_bad']} ===")

    print("\nDemo finished. Viewer stays live — press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
