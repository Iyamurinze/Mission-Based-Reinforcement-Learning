"""DQN training on AgriScout with TorchRL (value-based baseline + 10-run sweep)."""

import argparse

import torch
from tensordict.nn import TensorDictSequential
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate

from training.common import N_ACTIONS, EpisodeTracker, evaluate, make_env, mlp, save_run

DEFAULTS = {
    "lr": 1e-3, "gamma": 0.99, "buffer_size": 50_000, "batch_size": 128,
    "eps_init": 1.0, "eps_end": 0.05, "eps_anneal": 40_000,
    "target": "hard", "target_interval": 500, "tau": 0.995,
    "hidden": (128, 128), "total_frames": 250_000, "frames_per_batch": 256,
    "updates_per_batch": 8, "warmup": 1_000, "seed": 0,
}

SWEEP = {
    "r01_baseline": {},
    "r02_low_lr": {"lr": 1e-4},
    "r03_high_lr": {"lr": 5e-3},
    "r04_low_gamma": {"gamma": 0.90},
    "r05_tiny_buffer": {"buffer_size": 2_000},
    "r06_big_batch": {"batch_size": 512},
    "r07_greedy_fast": {"eps_anneal": 5_000, "eps_end": 0.01},
    "r08_explore_long": {"eps_anneal": 55_000, "eps_end": 0.15},
    "r09_soft_target": {"target": "soft", "tau": 0.99},
    "r10_slow_target": {"target_interval": 5_000},
    "r11_small_net": {"hidden": (64,)},
    "r12_deep_net": {"hidden": (256, 256, 128), "lr": 5e-4},
}


def train_dqn(run_name, overrides):
    # Train one DQN configuration and log its results.
    cfg = {**DEFAULTS, **overrides}
    torch.manual_seed(cfg["seed"])
    env = make_env(seed=cfg["seed"], explore_starts=True)

    actor = QValueActor(mlp(N_ACTIONS, cfg["hidden"]), in_keys=["observation"], spec=env.action_spec)
    egreedy = EGreedyModule(env.action_spec, eps_init=cfg["eps_init"], eps_end=cfg["eps_end"],
                            annealing_num_steps=cfg["eps_anneal"])
    policy = TensorDictSequential(actor, egreedy)

    collector = SyncDataCollector(env, policy, frames_per_batch=cfg["frames_per_batch"],
                                  total_frames=cfg["total_frames"], init_random_frames=cfg["warmup"])
    buffer = ReplayBuffer(storage=LazyTensorStorage(cfg["buffer_size"]), batch_size=cfg["batch_size"])

    loss_module = DQNLoss(actor, action_space=env.action_spec, delay_value=True)
    loss_module.make_value_estimator(gamma=cfg["gamma"])
    updater = (HardUpdate(loss_module, value_network_update_interval=cfg["target_interval"])
               if cfg["target"] == "hard" else SoftUpdate(loss_module, eps=cfg["tau"]))
    optim = torch.optim.Adam(loss_module.parameters(), lr=cfg["lr"])

    tracker, losses = EpisodeTracker(), []
    for i, td in enumerate(collector):
        buffer.extend(td.reshape(-1))
        tracker.update(td)
        if len(buffer) >= cfg["warmup"]:
            for _ in range(cfg["updates_per_batch"]):
                loss = loss_module(buffer.sample())["loss"]
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(loss_module.parameters(), 10.0)
                optim.step()
                updater.step()
                losses.append(loss.item())
        egreedy.step(td.numel())
        if i % 20 == 0 and tracker.returns:
            print(f"  [{run_name}] frames {(i + 1) * cfg['frames_per_batch']:>6} "
                  f"| episodes {len(tracker.returns):>4} | last-10 return "
                  f"{sum(tracker.returns[-10:]) / min(10, len(tracker.returns)):7.1f}")
    collector.shutdown()

    eval_returns = evaluate(actor)
    gen_returns = evaluate(actor, seed0=20_000, n_stressed=8)
    cfg["hidden"] = list(cfg["hidden"])
    return save_run("dqn", run_name, cfg, tracker.returns, {"loss": losses[::10]},
                    eval_returns, gen_returns, policy=actor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="run a single named config from the sweep")
    parser.add_argument("--frames", type=int, help="override total_frames (smoke tests)")
    args = parser.parse_args()
    runs = {args.run: SWEEP[args.run]} if args.run else SWEEP
    for name, overrides in runs.items():
        if args.frames:
            overrides = {**overrides, "total_frames": args.frames}
        print(f"=== DQN {name} ===")
        train_dqn(name, overrides)
