from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class BranchConfig:
    n_branches: int = 3
    separation: float = 0.7
    thickness: float = 0.045
    wiggle: float = 0.2
    x_range: float = 1.0


def branch_centers(x: np.ndarray, cfg: BranchConfig) -> np.ndarray:
    """Return branch centers with shape (n_branches, len(x), 2)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    centers = []
    for k in range(cfg.n_branches):
        offset = cfg.separation * (k - (cfg.n_branches - 1) / 2.0)
        y0 = x
        y1 = offset + cfg.wiggle * np.sin(3.0 * x + k)
        centers.append(np.stack([y0, y1], axis=-1))
    return np.stack(centers, axis=0)


def true_neg_log_density(x: np.ndarray, y: np.ndarray, cfg: BranchConfig) -> np.ndarray:
    """Analytic -log p(y|x) for the known branch mixture."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 2)
    centers = branch_centers(x.astype(np.float32), cfg)
    var = float(cfg.thickness) ** 2
    diff = y[None] - centers
    sq = (diff ** 2).sum(-1)
    log_comp = -sq / (2.0 * var) - np.log(2.0 * np.pi * var)
    log_mix = np.logaddexp.reduce(log_comp, axis=0) - np.log(centers.shape[0])
    return (-log_mix).astype(np.float64)


def sample_dataset(n: int, cfg: BranchConfig, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-cfg.x_range, cfg.x_range, size=n).astype(np.float32)
    branch = rng.integers(0, cfg.n_branches, size=n)
    centers = branch_centers(x, cfg)
    y = centers[branch, np.arange(n)]
    y = y + cfg.thickness * rng.standard_normal((n, 2)).astype(np.float32)
    return {"x": x[:, None].astype(np.float32), "y": y.astype(np.float32), "branch": branch}

