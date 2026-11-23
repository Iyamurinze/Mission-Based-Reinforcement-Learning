"""Policy-gradient training on AgriScout with TorchRL: REINFORCE, PPO and A2C."""

import argparse

import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.modules import OneHotCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss
from torchrl.objectives.value import GAE

from training.common import N_ACTIONS, EpisodeTracker, evaluate, make_env, mlp, save_run


def make_actor_critic(env, hidden):
    # Categorical policy head and a state-value critic over the observation.
    actor = ProbabilisticActor(
        TensorDictModule(mlp(N_ACTIONS, hidden), in_keys=["observation"], out_keys=["logits"]),
        spec=env.action_spec, in_keys=["logits"],
        distribution_class=OneHotCategorical, return_log_prob=True,
    )
    critic = ValueOperator(mlp(1, hidden), in_keys=["observation"])
    return actor, critic


def returns_to_go(rewards, dones, gamma):
    # Discounted return at every step, resetting at episode boundaries.
    G, out = 0.0, torch.zeros_like(rewards)
    for i in range(len(rewards) - 1, -1, -1):
        G = rewards[i] + gamma * G * (1.0 - dones[i])
        out[i] = G
    return out


REINFORCE_DEFAULTS = {
    "lr": 1e-3, "gamma": 0.99, "frames_per_batch": 2_048, "entropy_coef": 0.0,
    "baseline": "standardize", "hidden": (128, 128), "total_frames": 250_000, "seed": 0,
}

REINFORCE_SWEEP = {
    "r01_baseline": {},
    "r02_low_lr": {"lr": 1e-4},
    "r03_high_lr": {"lr": 5e-3},
    "r04_low_gamma": {"gamma": 0.90},
    "r05_no_baseline": {"baseline": "none"},
    "r06_entropy": {"entropy_coef": 0.01},
    "r07_high_entropy": {"entropy_coef": 0.05},
    "r08_small_batch": {"frames_per_batch": 512},
    "r09_big_batch": {"frames_per_batch": 8_192},
    "r10_small_net": {"hidden": (64,)},
    "r11_deep_net": {"hidden": (256, 256, 128), "lr": 5e-4},
}


def train_reinforce(run_name, overrides):
    # Monte-Carlo policy gradient built from TorchRL primitives.
    cfg = {**REINFORCE_DEFAULTS, **overrides}
    torch.manual_seed(cfg["seed"])
    env = make_env(seed=cfg["seed"], explore_starts=True)
    actor, _ = make_actor_critic(env, cfg["hidden"])
    collector = SyncDataCollector(env, actor, frames_per_batch=cfg["frames_per_batch"],
                                  total_frames=cfg["total_frames"])
    optim = torch.optim.Adam(actor.parameters(), lr=cfg["lr"])

    tracker, losses, entropies = EpisodeTracker(), [], []
    for i, td in enumerate(collector):
        tracker.update(td)
        flat = td.reshape(-1)
        rewards = flat["next", "reward"].squeeze(-1)
        dones = flat["next", "done"].squeeze(-1).float()
        done_idx = dones.nonzero()
        if not len(done_idx):
            continue
        mask = torch.arange(len(rewards)) <= done_idx[-1]

        G = returns_to_go(rewards, dones, cfg["gamma"])
        if cfg["baseline"] == "standardize":
            G = (G - G[mask].mean()) / (G[mask].std() + 1e-6)
        dist = actor.get_dist(flat)
        logp = dist.log_prob(flat["action"])
        entropy = dist.entropy().mean()
        loss = -(logp * G)[mask].mean() - cfg["entropy_coef"] * entropy

        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
        optim.step()
        losses.append(loss.item())
        entropies.append(entropy.item())
        if i % 5 == 0 and tracker.returns:
            print(f"  [{run_name}] frames {(i + 1) * cfg['frames_per_batch']:>7} "
                  f"| episodes {len(tracker.returns):>4} | last-10 return "
                  f"{sum(tracker.returns[-10:]) / min(10, len(tracker.returns)):7.1f}")
    collector.shutdown()

    cfg["hidden"] = list(cfg["hidden"])
    return save_run("reinforce", run_name, cfg, tracker.returns,
                    {"loss": losses, "entropy": entropies},
                    evaluate(actor), evaluate(actor, seed0=20_000, n_stressed=8), policy=actor)


PPO_DEFAULTS = {
    "lr": 3e-4, "gamma": 0.99, "lmbda": 0.95, "clip_epsilon": 0.2, "entropy_coef": 0.01,
    "epochs": 4, "minibatch": 256, "frames_per_batch": 1_024, "hidden": (128, 128),
    "total_frames": 250_000, "seed": 0,
}

PPO_SWEEP = {
    "r01_baseline": {},
    "r02_low_lr": {"lr": 3e-5},
    "r03_high_lr": {"lr": 3e-3},
    "r04_tight_clip": {"clip_epsilon": 0.1},
    "r05_loose_clip": {"clip_epsilon": 0.4},
    "r06_no_entropy": {"entropy_coef": 0.0},
    "r07_high_entropy": {"entropy_coef": 0.05},
    "r08_low_lambda": {"lmbda": 0.8},
    "r09_many_epochs": {"epochs": 10},
    "r10_big_rollout": {"frames_per_batch": 4_096},
    "r11_low_gamma": {"gamma": 0.90},
}


def train_ppo(run_name, overrides):
    # Clipped-surrogate PPO with GAE advantages.
    cfg = {**PPO_DEFAULTS, **overrides}
    torch.manual_seed(cfg["seed"])
    env = make_env(seed=cfg["seed"], explore_starts=True)
    actor, critic = make_actor_critic(env, cfg["hidden"])
    collector = SyncDataCollector(env, actor, frames_per_batch=cfg["frames_per_batch"],
                                  total_frames=cfg["total_frames"])
    advantage = GAE(gamma=cfg["gamma"], lmbda=cfg["lmbda"], value_network=critic, average_gae=True)
    loss_module = ClipPPOLoss(actor, critic, clip_epsilon=cfg["clip_epsilon"],
                              entropy_coef=cfg["entropy_coef"])
    optim = torch.optim.Adam(loss_module.parameters(), lr=cfg["lr"])

    tracker, losses, entropies = EpisodeTracker(), [], []
    for i, td in enumerate(collector):
        tracker.update(td)
        for _ in range(cfg["epochs"]):
            with torch.no_grad():
                advantage(td)
            flat = td.reshape(-1)
            for idx in torch.randperm(len(flat)).split(cfg["minibatch"]):
                loss_vals = loss_module(flat[idx])
                loss = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(loss_module.parameters(), 10.0)
                optim.step()
            losses.append(loss.item())
            entropies.append(loss_vals["entropy"].item())
        if i % 10 == 0 and tracker.returns:
            print(f"  [{run_name}] frames {(i + 1) * cfg['frames_per_batch']:>7} "
                  f"| episodes {len(tracker.returns):>4} | last-10 return "
                  f"{sum(tracker.returns[-10:]) / min(10, len(tracker.returns)):7.1f}")
    collector.shutdown()

    cfg["hidden"] = list(cfg["hidden"])
    return save_run("ppo", run_name, cfg, tracker.returns,
                    {"loss": losses, "entropy": entropies},
                    evaluate(actor), evaluate(actor, seed0=20_000, n_stressed=8), policy=actor)


A2C_DEFAULTS = {
    "lr": 7e-4, "gamma": 0.99, "lmbda": 0.95, "entropy_coef": 0.01, "critic_coef": 0.5,
    "frames_per_batch": 256, "hidden": (128, 128), "total_frames": 250_000, "seed": 0,
}

A2C_SWEEP = {
    "r01_baseline": {},
    "r02_low_lr": {"lr": 7e-5},
    "r03_high_lr": {"lr": 3e-3},
    "r04_low_gamma": {"gamma": 0.90},
    "r05_no_entropy": {"entropy_coef": 0.0},
    "r06_high_entropy": {"entropy_coef": 0.05},
    "r07_td_advantage": {"lmbda": 0.0},
    "r08_mc_advantage": {"lmbda": 1.0},
    "r09_small_rollout": {"frames_per_batch": 64},
    "r10_big_rollout": {"frames_per_batch": 1_024},
    "r11_strong_critic": {"critic_coef": 1.0},
}


def train_a2c(run_name, overrides):
    # Synchronous advantage actor-critic: one on-policy update per rollout.
    cfg = {**A2C_DEFAULTS, **overrides}
    torch.manual_seed(cfg["seed"])
    env = make_env(seed=cfg["seed"], explore_starts=True)
    actor, critic = make_actor_critic(env, cfg["hidden"])
    collector = SyncDataCollector(env, actor, frames_per_batch=cfg["frames_per_batch"],
                                  total_frames=cfg["total_frames"])
    advantage = GAE(gamma=cfg["gamma"], lmbda=cfg["lmbda"], value_network=critic,
                    average_gae=True, vectorized=cfg["lmbda"] > 0)
    loss_module = A2CLoss(actor, critic, entropy_coef=cfg["entropy_coef"],
                          critic_coef=cfg["critic_coef"])
    optim = torch.optim.Adam(loss_module.parameters(), lr=cfg["lr"])

    tracker, losses, entropies = EpisodeTracker(), [], []
    for i, td in enumerate(collector):
        tracker.update(td)
        with torch.no_grad():
            advantage(td)
        flat = td.reshape(-1)
        loss_vals = loss_module(flat)
        loss = loss_vals["loss_objective"] + loss_vals["loss_critic"]
        if "loss_entropy" in loss_vals.keys():
            loss = loss + loss_vals["loss_entropy"]
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(loss_module.parameters(), 10.0)
        optim.step()
        losses.append(loss.item())
        with torch.no_grad():
            entropies.append(actor.get_dist(flat).entropy().mean().item())
        if i % 40 == 0 and tracker.returns:
            print(f"  [{run_name}] frames {(i + 1) * cfg['frames_per_batch']:>7} "
                  f"| episodes {len(tracker.returns):>4} | last-10 return "
                  f"{sum(tracker.returns[-10:]) / min(10, len(tracker.returns)):7.1f}")
    collector.shutdown()

    cfg["hidden"] = list(cfg["hidden"])
    return save_run("a2c", run_name, cfg, tracker.returns,
                    {"loss": losses[::4], "entropy": entropies[::4]},
                    evaluate(actor), evaluate(actor, seed0=20_000, n_stressed=8), policy=actor)


TRAINERS = {
    "reinforce": (train_reinforce, REINFORCE_SWEEP),
    "ppo": (train_ppo, PPO_SWEEP),
    "a2c": (train_a2c, A2C_SWEEP),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=TRAINERS, required=True)
    parser.add_argument("--run", help="run a single named config from the sweep")
    parser.add_argument("--frames", type=int, help="override total_frames (smoke tests)")
    args = parser.parse_args()
    train_fn, sweep = TRAINERS[args.algo]
    runs = {args.run: sweep[args.run]} if args.run else sweep
    for name, overrides in runs.items():
        if args.frames:
            overrides = {**overrides, "total_frames": args.frames}
        print(f"=== {args.algo.upper()} {name} ===")
        train_fn(name, overrides)
