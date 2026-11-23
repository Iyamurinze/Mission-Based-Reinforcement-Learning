"""Shared helpers for all four TorchRL trainers: env factory, nets, logging, eval."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.utils import ExplorationType, set_exploration_type

from environment.custom_env import AgriScoutEnv

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
OBS_DIM, N_ACTIONS = 19, 7


def make_env(seed=None, **env_kwargs):
    # Build the AgriScout env wrapped for TorchRL.
    env = GymWrapper(AgriScoutEnv(**env_kwargs))
    if seed is not None:
        env.set_seed(seed)
    return env


def mlp(out_features, hidden=(128, 128)):
    # Plain ReLU MLP from the observation vector to `out_features`.
    layers, size = [], OBS_DIM
    for h in hidden:
        layers += [nn.Linear(size, h), nn.ReLU()]
        size = h
    layers.append(nn.Linear(size, out_features))
    return nn.Sequential(*layers)


class EpisodeTracker:
    """Accumulates per-episode returns out of flat collector batches."""

    def __init__(self):
        self.current = 0.0
        self.returns = []

    def update(self, td):
        # Walk the batch, closing an episode at every done flag.
        rewards = td["next", "reward"].reshape(-1)
        dones = td["next", "done"].reshape(-1)
        for r, d in zip(rewards.tolist(), dones.tolist()):
            self.current += r
            if d:
                self.returns.append(self.current)
                self.current = 0.0
        return self.returns


def evaluate(policy, n_episodes=20, seed0=10_000, explore=ExplorationType.MODE, **env_kwargs):
    # Greedy evaluation over fresh seeds; returns per-episode totals.
    env = make_env(**env_kwargs)
    totals = []
    with torch.no_grad(), set_exploration_type(explore):
        for i in range(n_episodes):
            env.set_seed(seed0 + i)
            td = env.rollout(1000, policy)
            totals.append(td["next", "reward"].sum().item())
    env.close()
    return totals


def save_run(algo, run_name, config, episode_returns, curves, eval_returns, gen_returns, policy=None):
    # Persist one experiment: config, learning curves, eval scores, weights.
    log_dir = LOGS_DIR / algo
    log_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "algo": algo, "run": run_name, "config": config,
        "episode_returns": [round(r, 2) for r in episode_returns],
        "curves": {k: [round(v, 5) for v in vals] for k, vals in curves.items()},
        "eval_mean": round(float(np.mean(eval_returns)), 2),
        "eval_std": round(float(np.std(eval_returns)), 2),
        "gen_mean": round(float(np.mean(gen_returns)), 2),
        "gen_std": round(float(np.std(gen_returns)), 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (log_dir / f"{run_name}.json").write_text(json.dumps(record))
    if policy is not None:
        model_dir = MODELS_DIR / ("dqn" if algo == "dqn" else "pg")
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), model_dir / f"{algo}_{run_name}.pt")
    print(f"[{algo}/{run_name}] eval {record['eval_mean']} ± {record['eval_std']} "
          f"| generalization {record['gen_mean']} ± {record['gen_std']}")
    return record
