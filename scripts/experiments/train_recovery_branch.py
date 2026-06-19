#!/usr/bin/env python
"""Dedicated recovery branch on a FROZEN base (candidate generation, lever 4 +
journal rec #1: learned recovery action + learned trigger).

Baking recovery into the shared codebook forgets the demos (0.875 -> 0.838). So
instead we keep the base model 100% frozen (the 0.875 prior is preserved exactly)
and add a small, dedicated branch:
  * a gate head:  obs -> P(recovery helps)   trained on real-env firings
                  (283 helpful = 1, 663 unhelpful = 0)  [fixes the blunt-push kills]
  * a chunk head: obs -> recovery action chunk (tanh)  trained on the 283 helpful
                  recoveries                          [learned recovery action]

Inference (self-contained gated rollout, no render_pusht_gif flags):
  - prior selection by default (base untouched);
  - only at a stuck state (same cheap trigger as the geometric recovery) consult
    the gate; if it predicts the recovery helps, execute the branch's learned
    recovery chunk; otherwise stay with the prior.

The hand-crafted geometric pushes were used only to *collect* labels offline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vq_sigreg.pusht_data import PushTChunkDataset, denormalize_action, normalize_state
from vq_sigreg.pusht_env import make_env
from vq_sigreg.pusht_policy import VQSigRegPolicy, load_policy

GOAL_PX = np.asarray([256.0, 256.0], dtype=np.float32)


class RecoveryBranch(nn.Module):
    """Frozen-base recovery branch with a multimodal (mixture) chunk head.

    A mixture-density head avoids the mode-averaging of an MSE head when similar
    stall states have divergent valid recoveries (e.g. push around the block on
    either side). At inference we take the highest-weight component's mean.
    """

    def __init__(self, obs_dim: int, chunk_dim: int, hidden: int = 256, n_components: int = 4, sigma: float = 0.05):
        super().__init__()
        self.chunk_dim = chunk_dim
        self.n_components = int(n_components)
        self.sigma = float(sigma)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.mean_head = nn.Linear(hidden, self.n_components * chunk_dim)
        self.mix_head = nn.Linear(hidden, self.n_components)
        self.gate_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        means = torch.tanh(self.mean_head(h)).reshape(-1, self.n_components, self.chunk_dim)
        mix_logits = self.mix_head(h)
        gate_logit = self.gate_head(h).squeeze(-1)
        return means, mix_logits, gate_logit

    def nll(self, obs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        means, mix_logits, _ = self.forward(obs)
        log_w = torch.log_softmax(mix_logits, dim=-1)
        sq = (means - target[:, None, :]).square().sum(dim=-1)
        log_comp = -0.5 * sq / (self.sigma ** 2)
        return -(torch.logsumexp(log_w + log_comp, dim=-1)).mean()

    @torch.no_grad()
    def best_chunk(self, obs: torch.Tensor) -> torch.Tensor:
        means, mix_logits, _ = self.forward(obs)
        sel = mix_logits.argmax(dim=-1)
        return means[torch.arange(means.shape[0], device=means.device), sel]

    @torch.no_grad()
    def gate_prob(self, obs: torch.Tensor) -> torch.Tensor:
        _, _, gate_logit = self.forward(obs)
        return torch.sigmoid(gate_logit)


def geometric_chunk_px(raw_state, horizon, standoff=70.0, contact=35.0, push=200.0):
    agent = np.asarray(raw_state[:2], dtype=np.float32)
    block = np.asarray(raw_state[2:4], dtype=np.float32)
    d = GOAL_PX - block
    n = float(np.linalg.norm(d))
    d = d / n if n > 1e-4 else np.asarray([np.cos(raw_state[4]), np.sin(raw_state[4])], dtype=np.float32)
    way = np.stack([agent, block - d * standoff, block - d * contact, block + d * push], axis=0)
    seg = np.asarray([0.0, 0.35, 0.55, 1.0], dtype=np.float32)
    fr = np.linspace(0.0, 1.0, horizon, dtype=np.float32)
    out = np.empty((horizon, 2), dtype=np.float32)
    for i, f in enumerate(fr):
        s = min(max(int(np.searchsorted(seg, f, side="right") - 1), 0), len(seg) - 2)
        loc = (f - seg[s]) / max(float(seg[s + 1] - seg[s]), 1e-6)
        out[i] = (1.0 - loc) * way[s] + loc * way[s + 1]
    return np.clip(out, 0.0, 512.0)


def stuck(obs_raw, history, obs_steps, init_block, init_goal, bd=6.0, rd=1.5, gp=-2.0):
    block = np.asarray(obs_raw[2:4], dtype=np.float32)
    total = float(np.linalg.norm(block - init_block))
    win = np.stack(history[-obs_steps:], 0).reshape(obs_steps, -1)
    rb = (win[:, 2:4] + 1.0) * 0.5 * 512.0
    recent = float(np.linalg.norm(rb[-1] - rb[0]))
    prog = init_goal - float(np.linalg.norm(GOAL_PX - block))
    return (total <= bd and recent <= rd) or (prog <= gp)


@torch.no_grad()
def gated_rollout(policy, branch, cfg, device, seed, gate_thresh, max_uses, min_plans, max_plans, chunk_mode="mdn", max_steps=300, replan=16):
    env = make_env()
    obs_raw, _ = env.reset(seed=seed)
    policy.reset()
    policy.selector = "prior"
    obs_steps = int(cfg["data"]["obs_steps"])
    horizon = int(policy.model.horizon)
    history = [normalize_state(np.asarray(obs_raw))] * obs_steps
    init_block = np.asarray(obs_raw[2:4], dtype=np.float32).copy()
    init_goal = float(np.linalg.norm(GOAL_PX - init_block))
    max_cov = 0.0
    success = False
    steps = 0
    plan_calls = 0
    uses = 0
    while steps < max_steps:
        obs_vec = np.concatenate(history[-obs_steps:]).astype(np.float32)
        obs_t = torch.from_numpy(obs_vec)[None].to(device)
        plan_calls += 1
        use_branch = False
        if min_plans <= plan_calls <= max_plans and uses < max_uses and stuck(obs_raw, history, obs_steps, init_block, init_goal):
            if branch.gate_prob(obs_t).item() >= gate_thresh:
                use_branch = True
                if chunk_mode == "geometric":
                    actions = geometric_chunk_px(np.asarray(obs_raw), horizon)
                else:
                    actions = denormalize_action(branch.best_chunk(obs_t)[0].reshape(-1, 2).cpu().numpy())
                uses += 1
        if not use_branch:
            actions = denormalize_action(policy.plan(obs_t, np.asarray(obs_raw)).chunk.reshape(-1, 2).cpu().numpy())
        done = False
        for a in actions[:replan]:
            obs_raw, reward, term, trunc, info = env.step(np.clip(a, 0.0, 512.0).astype(np.float32))
            steps += 1
            max_cov = max(max_cov, float(reward))
            success = success or bool(info.get("is_success", False))
            history.append(normalize_state(np.asarray(obs_raw)))
            if term or trunc or steps >= max_steps:
                done = bool(term or trunc)
                break
        if done:
            break
    env.close()
    return {"seed": seed, "max_coverage": float(max_cov), "success": float(success), "branch_uses": uses}


def evaluate(policy, branch, cfg, device, thresh, max_uses, min_plans, max_plans, chunk_mode="mdn", episodes=50, start=10000):
    rolls = [gated_rollout(policy, branch, cfg, device, start + i, thresh, max_uses, min_plans, max_plans, chunk_mode=chunk_mode) for i in range(episodes)]
    cov = np.asarray([r["max_coverage"] for r in rolls])
    return {
        "chunk_mode": chunk_mode,
        "thresh": thresh,
        "mean_max_coverage": float(cov.mean()),
        "success_rate": float(np.mean([r["success"] for r in rolls])),
        "frac_above_0.9": float((cov > 0.9).mean()),
        "frac_stalled": float((cov < 0.1).mean()),
        "episodes_with_branch": float(np.mean([r["branch_uses"] > 0 for r in rolls])),
    }, rolls


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="outputs/vq_sigreg_pusht_k16_continuous_residual_s003_g48_a035/vq_continuous_residual_latest.pt")
    p.add_argument("--recovery-data", default="outputs/_recovery/recovery_data.pt")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--out-dir", default="outputs/vq_sigreg_pusht_k16_recovery_branch")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--max-uses", type=int, default=1)
    p.add_argument("--min-plans", type=int, default=2)
    p.add_argument("--max-plans", type=int, default=8)
    p.add_argument("--n-components", type=int, default=4)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=29)
    args = p.parse_args()

    torch.manual_seed(int(args.seed)); np.random.seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    policy, cfg = load_policy(args.checkpoint, device)
    if not isinstance(policy, VQSigRegPolicy):
        raise SystemExit("need VQSigRegPolicy checkpoint")
    for prm in policy.model.parameters():
        prm.requires_grad_(False)

    rec = torch.load(args.recovery_data, map_location="cpu", weights_only=False)
    margin = float(rec.get("progress_margin_px", 12.0))
    all_obs = rec["all_obs"].float().to(device)
    all_chunk = rec["all_chunk"].float().to(device)
    all_prog = rec["all_progress_px"].float().to(device)
    gate_label = (all_prog >= margin).float()
    pos_mask = gate_label > 0.5
    pos_obs = all_obs[pos_mask]
    pos_chunk = all_chunk[pos_mask]
    print(f"firings={all_obs.shape[0]} positives={int(pos_mask.sum())} negatives={int((~pos_mask).sum())}")

    obs_dim = all_obs.shape[1]
    chunk_dim = all_chunk.shape[1]
    branch = RecoveryBranch(obs_dim, chunk_dim, n_components=int(args.n_components), sigma=float(args.sigma)).to(device)
    opt = torch.optim.AdamW(branch.parameters(), lr=float(args.lr), weight_decay=1e-5)
    n_all = all_obs.shape[0]
    n_pos = pos_obs.shape[0]
    pos_weight = torch.tensor([(n_all - int(pos_mask.sum())) / max(1, int(pos_mask.sum()))], device=device)

    branch.train()
    for step in range(1, int(args.steps) + 1):
        gi = torch.randint(0, n_all, (int(args.batch),), device=device)
        ci = torch.randint(0, n_pos, (int(args.batch),), device=device)
        _, _, gate_logit = branch(all_obs[gi])
        gate_loss = F.binary_cross_entropy_with_logits(gate_logit, gate_label[gi], pos_weight=pos_weight)
        chunk_loss = branch.nll(pos_obs[ci], pos_chunk[ci])
        loss = gate_loss + chunk_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 500 == 0 or step == int(args.steps):
            with torch.no_grad():
                gl = branch.gate_prob(all_obs)
                acc = ((gl >= 0.5).float() == gate_label).float().mean()
            print(json.dumps({"step": step, "gate_loss": float(gate_loss), "chunk_nll": float(chunk_loss), "gate_acc": float(acc)}))
    branch.eval()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"branch": branch.state_dict(), "obs_dim": obs_dim, "chunk_dim": chunk_dim}, out_dir / "recovery_branch.pt")

    results = []
    for chunk_mode in ("mdn", "geometric"):
        for thresh in (0.7, 0.85, 0.95):
            m, _ = evaluate(policy, branch, cfg, device, thresh, int(args.max_uses), int(args.min_plans), int(args.max_plans), chunk_mode=chunk_mode)
            results.append(m)
            print(json.dumps(m))
    (out_dir / "branch_eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
