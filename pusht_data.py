from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


WORKSPACE = 512.0
GOAL_XY = np.array([390.0, 256.0], dtype=np.float32)
OBSTACLE_XY = np.array([256.0, 256.0], dtype=np.float32)


def normalize_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    positions = state[..., :4] / WORKSPACE * 2.0 - 1.0
    angle = state[..., 4:5]
    return np.concatenate([positions, np.cos(angle), np.sin(angle)], axis=-1)


def normalize_action(action: np.ndarray) -> np.ndarray:
    return action.astype(np.float32) / WORKSPACE * 2.0 - 1.0


def denormalize_action(action: np.ndarray | torch.Tensor):
    return (action + 1.0) / 2.0 * WORKSPACE


class PushTChunkDataset(Dataset[dict[str, torch.Tensor]]):
    """Isolated copy of the zarr Push-T chunk loader.

    It intentionally does not import from `old/`. `zarr` is imported lazily so
    synthetic open-loop visuals still work on machines without the real dataset.
    """

    def __init__(
        self,
        zarr_path: str | Path,
        horizon: int = 16,
        obs_steps: int = 2,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 0,
        max_samples: int | None = None,
    ):
        try:
            import zarr
        except ImportError as exc:
            raise RuntimeError("Install zarr or use data.source=synthetic for Push-T open-loop.") from exc

        zarr_path = Path(zarr_path)
        if not zarr_path.exists():
            raise FileNotFoundError(f"Push-T zarr not found: {zarr_path}")
        root = zarr.open(str(zarr_path), "r")
        states = np.asarray(root["data/state"])
        actions = np.asarray(root["data/action"])
        episode_ends = np.asarray(root["meta/episode_ends"])
        starts = np.concatenate([[0], episode_ends[:-1]])

        rng = np.random.default_rng(seed)
        order = rng.permutation(len(episode_ends))
        num_val = max(1, int(len(episode_ends) * val_fraction))
        val_episodes = set(order[:num_val].tolist())
        keep = [i for i in range(len(episode_ends)) if (i in val_episodes) == (split == "val")]

        obs_list, chunk_list, raw_state_list, route_list = [], [], [], []
        for ep in keep:
            lo, hi = int(starts[ep]), int(episode_ends[ep])
            ep_states = normalize_state(states[lo:hi])
            ep_actions = normalize_action(actions[lo:hi])
            length = hi - lo
            for t in range(length):
                obs_idx = np.clip(np.arange(t - obs_steps + 1, t + 1), 0, length - 1)
                chunk_idx = np.clip(np.arange(t, t + horizon), 0, length - 1)
                chunk_px = actions[lo + chunk_idx]
                # A lightweight route label for diagnostics: above vs below the obstacle.
                route = 1 if float(np.mean(chunk_px[:, 1] - OBSTACLE_XY[1])) >= 0.0 else 0
                obs_list.append(ep_states[obs_idx].reshape(-1))
                chunk_list.append(ep_actions[chunk_idx].reshape(-1))
                raw_state_list.append(states[lo + t])
                route_list.append(route)
                if max_samples is not None and len(obs_list) >= max_samples:
                    break
            if max_samples is not None and len(obs_list) >= max_samples:
                break

        self.obs = torch.from_numpy(np.stack(obs_list)).float()
        self.chunks = torch.from_numpy(np.stack(chunk_list)).float()
        self.raw_states = torch.from_numpy(np.stack(raw_state_list)).float()
        self.route = torch.tensor(route_list, dtype=torch.long)
        self.horizon = int(horizon)
        self.obs_steps = int(obs_steps)

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "obs": self.obs[idx],
            "chunk": self.chunks[idx],
            "raw_state": self.raw_states[idx],
            "route": self.route[idx],
        }


class SyntheticPushTForkDataset(Dataset[dict[str, torch.Tensor]]):
    """Push-T-shaped open-loop fork: same context, two valid routes around a block."""

    def __init__(self, n: int = 12000, horizon: int = 16, obs_steps: int = 2, seed: int = 0):
        rng = np.random.default_rng(seed)
        obs_list, chunk_list, raw_state_list, route_list = [], [], [], []
        t = np.linspace(0.0, 1.0, horizon, dtype=np.float32)
        for _ in range(int(n)):
            route = int(rng.integers(0, 2))
            side = 1.0 if route == 1 else -1.0
            start = np.array(
                [122.0 + rng.normal(0.0, 8.0), 256.0 + rng.normal(0.0, 8.0)],
                dtype=np.float32,
            )
            goal = GOAL_XY + rng.normal(0.0, 6.0, size=2).astype(np.float32)
            x = (1.0 - t) * start[0] + t * goal[0]
            arch = side * (88.0 * np.sin(np.pi * t) + rng.normal(0.0, 2.0, size=horizon))
            y = (1.0 - t) * start[1] + t * goal[1] + arch.astype(np.float32)
            chunk = np.stack([x, y], axis=-1)
            chunk += rng.normal(0.0, 3.0, size=chunk.shape).astype(np.float32)
            raw_state = np.array([start[0], start[1], 256.0, 256.0, math.pi / 4.0], dtype=np.float32)
            obs = normalize_state(np.repeat(raw_state[None], obs_steps, axis=0)).reshape(-1)
            obs_list.append(obs)
            chunk_list.append(normalize_action(chunk).reshape(-1))
            raw_state_list.append(raw_state)
            route_list.append(route)
        self.obs = torch.from_numpy(np.stack(obs_list)).float()
        self.chunks = torch.from_numpy(np.stack(chunk_list)).float()
        self.raw_states = torch.from_numpy(np.stack(raw_state_list)).float()
        self.route = torch.tensor(route_list, dtype=torch.long)
        self.horizon = int(horizon)
        self.obs_steps = int(obs_steps)

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "obs": self.obs[idx],
            "chunk": self.chunks[idx],
            "raw_state": self.raw_states[idx],
            "route": self.route[idx],
        }


def make_openloop_datasets(cfg: dict[str, Any], seed: int):
    data = cfg["data"]
    source = str(data.get("source", "auto"))
    horizon = int(data.get("horizon", 16))
    obs_steps = int(data.get("obs_steps", 2))
    if source == "real" or (source == "auto" and Path(str(data.get("zarr_path", ""))).exists()):
        common = dict(
            zarr_path=data["zarr_path"],
            horizon=horizon,
            obs_steps=obs_steps,
            val_fraction=float(data.get("val_fraction", 0.1)),
            max_samples=data.get("max_samples"),
        )
        return (
            PushTChunkDataset(split="train", seed=seed, **common),
            PushTChunkDataset(split="val", seed=seed, **common),
            "real",
        )
    return (
        SyntheticPushTForkDataset(int(data.get("n_train", 12000)), horizon, obs_steps, seed),
        SyntheticPushTForkDataset(int(data.get("n_val", 2000)), horizon, obs_steps, seed + 1000),
        "synthetic",
    )

