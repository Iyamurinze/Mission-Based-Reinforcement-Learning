"""Sanity checks for the AgriScout environment."""

import numpy as np
from environment.custom_env import AgriScoutEnv, SCAN, TREAT, CHARGE, STRESS_THRESHOLD


def test_reset_shapes_and_ranges():
    # Reset must return a valid observation inside the declared space.
    env = AgriScoutEnv()
    obs, info = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    assert info["stressed_left"] == 5
    assert info["pos"] == (0, 0)


def test_random_episode_terminates():
    # A random policy must always end an episode within the step limit.
    env = AgriScoutEnv()
    env.reset(seed=1)
    for _ in range(env.max_steps + 1):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            break
    assert terminated or truncated


def test_treat_cures_stressed_field():
    # Treating a stressed field restores it and pays a positive reward.
    env = AgriScoutEnv()
    env.reset(seed=3)
    env.ndvi[0, 0] = 0.2
    _, reward, _, _, info = env.step(TREAT)
    assert reward > 0
    assert env.ndvi[0, 0] >= STRESS_THRESHOLD
    assert info["treated_ok"] == 1


def test_treat_healthy_is_penalized():
    # Spraying a healthy field wastes resources and is punished.
    env = AgriScoutEnv()
    env.reset(seed=3)
    env.ndvi[0, 0] = 0.8
    env.ndvi[1, 1] = 0.2
    _, reward, _, _, _ = env.step(TREAT)
    assert reward < 0


def test_scan_reveals_truth():
    # A scan copies the true NDVI into the drone's knowledge map.
    env = AgriScoutEnv()
    env.reset(seed=5)
    assert env.known[0, 0] == -1.0
    env.step(SCAN)
    assert env.known[0, 0] == env.ndvi[0, 0]


def test_battery_death_terminates():
    # Running out of battery away from base ends the episode with a crash penalty.
    env = AgriScoutEnv()
    env.reset(seed=7)
    env.pos = [4, 4]
    env.battery = 0.5
    _, reward, terminated, _, _ = env.step(0)
    assert terminated
    assert reward < -15


def test_charge_only_at_base():
    # Charging works at base and is a wasted action elsewhere.
    env = AgriScoutEnv()
    env.reset(seed=9)
    env.battery = 50.0
    env.step(CHARGE)
    assert env.battery > 50.0
    env.pos = [3, 3]
    before = env.battery
    env.step(CHARGE)
    assert env.battery == before


def test_seeded_resets_reproduce():
    # The same seed must produce the same hidden stress pattern.
    a, b = AgriScoutEnv(), AgriScoutEnv()
    a.reset(seed=11)
    b.reset(seed=11)
    assert np.allclose(a.ndvi, b.ndvi)
