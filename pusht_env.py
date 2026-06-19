from __future__ import annotations

from typing import Any

import numpy as np


def make_env(seed: int | None = None, render_mode: str | None = None):
    import gymnasium as gym
    import gym_pusht  # noqa: F401

    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env


def set_env_state_exact(env: Any, state: np.ndarray, reset_space: bool = False) -> np.ndarray:
    """Set a Push-T 5D state without the legacy block-position round-trip shift.

    ``gym_pusht``'s private ``_set_state`` assigns block position and then block
    angle. For the T block, changing angle after position rotates around the
    center of mass and changes the observed block position by up to ~90 px for
    states from ``pusht_cchi_v7_replay.zarr``. The exact round-trip order is:
    set angle first, then set position.

    ``reset_space=True`` rebuilds bodies/joints before setting the state; this
    is appropriate for counterfactual scoring envs and demo replay. We also
    zero velocities because the state observation does not include them.
    """
    unwrapped = env.unwrapped
    if reset_space:
        unwrapped._setup()
    state = np.asarray(state, dtype=np.float64)
    unwrapped.agent.position = list(state[:2])
    unwrapped.agent.velocity = (0, 0)
    unwrapped.block.velocity = (0, 0)
    unwrapped.block.angular_velocity = 0
    unwrapped.block.angle = float(state[4])
    unwrapped.block.position = list(state[2:4])
    unwrapped.space.step(unwrapped.dt)
    return np.asarray(unwrapped.get_obs(), dtype=np.float64)
