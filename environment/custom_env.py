"""AgriScout: a crop-scouting drone over a grid of Rwandan farm fields.

The drone must find stressed fields (low NDVI), treat them before stress
spreads to neighbours, and manage its battery by recharging at the base.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

MOVE_N, MOVE_S, MOVE_E, MOVE_W, SCAN, TREAT, CHARGE = range(7)
ACTION_NAMES = ["move_north", "move_south", "move_east", "move_west", "scan", "treat", "charge"]

HEALTHY_NDVI = (0.65, 0.90)
STRESSED_NDVI = (0.15, 0.45)
STRESS_THRESHOLD = 0.5


class AgriScoutEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 8}

    def __init__(self, grid_size=8, n_stressed=5, max_steps=250, battery_capacity=100.0,
                 spread_prob=0.006, explore_starts=False, render_mode=None):
        # Store config and declare the Gymnasium action/observation spaces.
        super().__init__()
        self.explore_starts = explore_starts
        self.grid_size = grid_size
        self.n_stressed = n_stressed
        self.max_steps = max_steps
        self.battery_capacity = battery_capacity
        self.spread_prob = spread_prob
        self.render_mode = render_mode
        self.base = (0, 0)
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(19,), dtype=np.float32)
        self._renderer = None

    def reset(self, seed=None, options=None):
        # Place the drone at base with full battery and hide a fresh stress pattern.
        super().reset(seed=seed)
        if options and "n_stressed" in options:
            self.n_stressed = options["n_stressed"]
        n = self.grid_size
        self.ndvi = self.np_random.uniform(*HEALTHY_NDVI, size=(n, n))
        stressed = self.np_random.choice(n * n, size=self.n_stressed, replace=False)
        self.ndvi.flat[stressed] = self.np_random.uniform(*STRESSED_NDVI, size=self.n_stressed)
        self.known = np.full((n, n), -1.0)
        satellite = self.np_random.random((n, n)) < 0.3
        self.known[satellite] = self.ndvi[satellite]
        self.pos = list(self.base)
        self.battery = self.battery_capacity
        if self.explore_starts:
            self.pos = [int(self.np_random.integers(n)), int(self.np_random.integers(n))]
            self.battery = float(self.np_random.uniform(40.0, self.battery_capacity))
        self.wind = self.np_random.uniform(0.0, 0.5)
        self.steps = 0
        self.treated_ok = 0
        self.treated_bad = 0
        return self._observation(), self._info()

    def step(self, action):
        # Apply one drone command, spread stress, and compute the reward.
        reward = -0.1
        self.steps += 1
        phi_before = self._potential()
        r, c = self.pos

        if action in (MOVE_N, MOVE_S, MOVE_E, MOVE_W):
            dr, dc = [(-1, 0), (1, 0), (0, 1), (0, -1)][action]
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                self.pos = [nr, nc]
            else:
                reward -= 0.5
            self.battery -= 0.8 + 0.5 * self.wind
        elif action == SCAN:
            self.battery -= 2.0
            newly = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        newly += self.known[nr, nc] < 0
                        self.known[nr, nc] = self.ndvi[nr, nc]
            reward += 0.25 * newly if newly else -0.2
        elif action == TREAT:
            self.battery -= 3.0
            if self.ndvi[r, c] < STRESS_THRESHOLD:
                reward += 20.0
                self.treated_ok += 1
            else:
                reward -= 5.0
                self.treated_bad += 1
            self.ndvi[r, c] = self.np_random.uniform(*HEALTHY_NDVI)
            self.known[r, c] = self.ndvi[r, c]
        elif action == CHARGE:
            if tuple(self.pos) == self.base and self.battery < self.battery_capacity:
                self.battery = min(self.battery_capacity, self.battery + 25.0)
            else:
                reward -= 0.5

        self._spread_stress()
        self.wind = float(np.clip(self.wind + self.np_random.uniform(-0.05, 0.05), 0.0, 1.0))
        reward += 0.99 * self._potential() - phi_before

        terminated, truncated = False, False
        if self.battery <= 0 and tuple(self.pos) != self.base:
            reward -= 60.0
            terminated = True
        elif not (self.ndvi < STRESS_THRESHOLD).any():
            reward += 25.0
            terminated = True
        elif self.steps >= self.max_steps:
            reward -= 2.0 * int((self.ndvi < STRESS_THRESHOLD).sum())
            truncated = True

        if self.render_mode == "human":
            self.render()
        return self._observation(), reward, terminated, truncated, self._info()

    def _potential(self):
        # Shaping potential (Ng et al.): closeness to the nearest KNOWN stressed field.
        targets = np.argwhere((self.known >= 0) & (self.known < STRESS_THRESHOLD))
        if not len(targets):
            return -0.4 * (2 * self.grid_size - 2)
        return -0.4 * float(np.abs(targets - self.pos).sum(axis=1).min())

    def _spread_stress(self):
        # Each stressed field may infect one random healthy neighbour.
        stressed = np.argwhere(self.ndvi < STRESS_THRESHOLD)
        for r, c in stressed:
            if self.np_random.random() < self.spread_prob:
                dr, dc = [(-1, 0), (1, 0), (0, 1), (0, -1)][self.np_random.integers(4)]
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size and self.ndvi[nr, nc] >= STRESS_THRESHOLD:
                    self.ndvi[nr, nc] = self.np_random.uniform(*STRESSED_NDVI)

    def _observation(self):
        # Build the 19-value sensor vector the drone actually perceives.
        n, (r, c) = self.grid_size, self.pos
        patch = np.full(9, -1.0)
        for i, (dr, dc) in enumerate([(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                patch[i] = self.known[nr, nc]
        known_stressed = np.argwhere((self.known >= 0) & (self.known < STRESS_THRESHOLD))
        if len(known_stressed):
            nearest = known_stressed[np.abs(known_stressed - [r, c]).sum(axis=1).argmin()]
            target = [(nearest[0] - r) / (n - 1), (nearest[1] - c) / (n - 1), 1.0]
        else:
            target = [0.0, 0.0, 0.0]
        obs = [r / (n - 1), c / (n - 1), self.battery / self.battery_capacity,
               *patch,
               (self.base[0] - r) / (n - 1), (self.base[1] - c) / (n - 1),
               *target, self.wind, 1.0 - self.steps / self.max_steps]
        return np.asarray(obs, dtype=np.float32)

    def _info(self):
        # Expose true state for logging and rendering (never fed to the agent).
        return {
            "pos": tuple(self.pos), "battery": self.battery, "wind": self.wind,
            "steps": self.steps, "stressed_left": int((self.ndvi < STRESS_THRESHOLD).sum()),
            "treated_ok": self.treated_ok, "treated_bad": self.treated_bad,
        }

    def render_state(self):
        # Full snapshot consumed by the three.js renderer.
        return {
            "grid_size": self.grid_size, "pos": list(self.pos),
            "battery": round(self.battery, 1), "wind": round(self.wind, 2),
            "steps": self.steps, "ndvi": np.round(self.ndvi, 3).tolist(),
            "known": np.round(self.known, 3).tolist(), "base": list(self.base),
            "stressed_left": int((self.ndvi < STRESS_THRESHOLD).sum()),
        }

    def render(self):
        # Push the current state to the attached live renderer, if any.
        if self._renderer is not None:
            self._renderer.update(self.render_state())
