"""Conditional DDPM action-chunk generator for Push-T (DP-on-harness baseline).

Self-contained copy of the generator from old/src/cv_ebm/diffusion_policy.py
(reranker / value head dropped -- we only need the deployable generator). Shapes
match the VQ-SIGReg pipeline exactly so the closed-loop harness is identical:

  obs   : (B, obs_steps * 6)   = (B, 12)   conditioning (normalize_state output)
  chunk : (B, horizon * 2)     = (B, 32)   normalized absolute action chunk
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_beta_schedule(num_steps: int, kind: str = "cosine") -> torch.Tensor:
    if kind == "linear":
        return torch.linspace(1e-4, 0.02, num_steps)
    s = 0.008
    t = torch.linspace(0, num_steps, num_steps + 1) / num_steps
    alphas_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return betas.clamp(1e-5, 0.999)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / max(half - 1, 1))
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class _ResBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.cond = nn.Linear(cond_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        r = F.silu(self.fc1(h) + self.cond(cond))
        r = self.fc2(r)
        return self.norm(h + r)


class ConditionalEpsMLP(nn.Module):
    def __init__(self, chunk_dim: int, obs_dim: int, hidden: int = 512, t_embed: int = 128, depth: int = 4):
        super().__init__()
        self.chunk_dim = chunk_dim
        self.t_embed = t_embed
        self.in_proj = nn.Linear(chunk_dim, hidden)
        self.cond_proj = nn.Sequential(
            nn.Linear(t_embed + obs_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.blocks = nn.ModuleList([_ResBlock(hidden, hidden) for _ in range(depth)])
        self.out = nn.Linear(hidden, chunk_dim)

    def forward(self, a_t: torch.Tensor, t: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        cond = self.cond_proj(torch.cat([timestep_embedding(t, self.t_embed), obs], dim=-1))
        h = self.in_proj(a_t)
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out(h)


class DiffusionPolicy(nn.Module):
    """Conditional DDPM over action chunks with ancestral + DDIM sampling."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        d = cfg["data"]
        m = cfg.get("model", {})
        self.horizon = int(d["horizon"])
        self.obs_steps = int(d["obs_steps"])
        self.chunk_dim = self.horizon * 2
        self.obs_dim = self.obs_steps * 6  # STATE_FEATURES (pos4 + cos + sin)
        self.num_steps = int(m.get("diffusion_steps", 100))
        self.net = ConditionalEpsMLP(
            self.chunk_dim, self.obs_dim,
            hidden=int(m.get("hidden_dim", 512)),
            t_embed=int(m.get("t_embed", 128)),
            depth=int(m.get("depth", 4)),
        )
        betas = make_beta_schedule(self.num_steps, m.get("beta_schedule", "cosine"))
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("alphas_bar_prev", torch.cat([torch.ones(1), alphas_bar[:-1]]))

    def loss(self, chunk: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        b = chunk.shape[0]
        t = torch.randint(0, self.num_steps, (b,), device=chunk.device)
        noise = torch.randn_like(chunk)
        ab = self.alphas_bar[t][:, None]
        a_t = ab.sqrt() * chunk + (1 - ab).sqrt() * noise
        pred = self.net(a_t, t, obs)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, num_samples: int = 1, steps: int | None = None, method: str = "ddpm") -> torch.Tensor:
        device = obs.device
        if obs.shape[0] == 1 and num_samples > 1:
            obs = obs.expand(num_samples, -1)
        n = obs.shape[0]
        x = torch.randn(n, self.chunk_dim, device=device)
        if method == "ddim":
            return self._ddim(x, obs, steps or self.num_steps)
        for i in reversed(range(self.num_steps)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            eps = self.net(x, t, obs)
            ab = self.alphas_bar[i]
            ab_prev = self.alphas_bar_prev[i]
            beta = self.betas[i]
            alpha = self.alphas[i]
            x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
            x0 = x0.clamp(-1.0, 1.0)
            mean = (ab_prev.sqrt() * beta / (1 - ab)) * x0 + (alpha.sqrt() * (1 - ab_prev) / (1 - ab)) * x
            if i > 0:
                var = beta * (1 - ab_prev) / (1 - ab)
                x = mean + var.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x.clamp(-1.0, 1.0)

    @torch.no_grad()
    def _ddim(self, x: torch.Tensor, obs: torch.Tensor, steps: int) -> torch.Tensor:
        idx = torch.linspace(self.num_steps - 1, 0, steps, device=x.device).long()
        for j, i in enumerate(idx.tolist()):
            t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
            eps = self.net(x, t, obs)
            ab = self.alphas_bar[i]
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1.0, 1.0)
            i_prev = int(idx[j + 1]) if j + 1 < len(idx) else 0
            ab_prev = self.alphas_bar[i_prev]
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return x.clamp(-1.0, 1.0)
