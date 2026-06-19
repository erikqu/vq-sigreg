#!/usr/bin/env python
"""Evaluate a Diffusion Policy on the EXACT VQ-SIGReg cold-stepped harness.

Same make_env, cold-stepping, coverage metric, seeds, max_steps and replan_every
as rollout_oracle_ceiling.py's prior baseline -- the only thing that changes is
the generator (DDPM sample instead of VQ prior argmax). DP sampling is
stochastic, so we report mean +/- std over several RNG seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vq_sigreg.diffusion_policy import DiffusionPolicy
from vq_sigreg.pusht_data import denormalize_action, normalize_state
from vq_sigreg.pusht_env import make_env
from rollout_oracle_ceiling import cold_step, coverage_of


@torch.no_grad()
def run_episode(model, cfg, device, seed, max_steps, replan_every, sample_steps, method, harness="cold"):
    obs_steps = int(cfg["data"]["obs_steps"])
    horizon = int(cfg["data"]["horizon"])
    env = make_env(seed=seed, render_mode=None)
    obs_raw, _ = env.reset(seed=seed)
    history = [normalize_state(np.asarray(obs_raw))] * obs_steps
    step_fn = cold_step if harness == "cold" else (lambda e, a: e.step(np.clip(np.asarray(a), 0.0, 512.0).astype(np.float32)))
    max_cov = 0.0
    success = False
    t = 0
    terminated = truncated = False
    while t < max_steps and not (terminated or truncated):
        obs_vec = np.concatenate(history[-obs_steps:]).astype(np.float32)
        obs = torch.from_numpy(obs_vec)[None].to(device)
        chunk = model.sample(obs, num_samples=1, steps=sample_steps, method=method)[0]
        chunk = chunk.clamp(-1.0, 1.0).reshape(horizon, 2).detach().cpu().numpy()
        chunk_px = denormalize_action(chunk)
        for action in chunk_px[: int(replan_every)]:
            obs_raw, reward, terminated, truncated, info = step_fn(env, action)
            t += 1
            max_cov = max(max_cov, coverage_of(info, reward, env))
            success = success or bool(info.get("is_success", False))
            history.append(normalize_state(np.asarray(obs_raw)))
            if terminated or truncated or t >= max_steps:
                break
    env.close()
    return max_cov, success


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--start-seed", type=int, default=10000)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--replan-every", type=int, default=16)
    ap.add_argument("--rng-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--weights", choices=["ema", "model"], default="ema")
    ap.add_argument("--method", choices=["ddpm", "ddim"], default="ddpm")
    ap.add_argument("--sample-steps", type=int, default=0, help="0 = full num_steps")
    ap.add_argument("--harness", choices=["cold", "warm"], default="cold", help="warm = official gym_pusht stepping (matches render_pusht_gif)")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = DiffusionPolicy(cfg).to(device)
    model.load_state_dict(ckpt["ema_state" if args.weights == "ema" else "model_state"])
    model.eval()
    steps = None if int(args.sample_steps) <= 0 else int(args.sample_steps)

    per_rng = []
    for rs in args.rng_seeds:
        torch.manual_seed(rs)
        np.random.seed(rs)
        covs, succ = [], []
        for ep in range(int(args.episodes)):
            c, s = run_episode(model, cfg, device, int(args.start_seed) + ep, int(args.max_steps),
                               int(args.replan_every), steps, args.method, harness=args.harness)
            covs.append(c)
            succ.append(s)
        covs = np.asarray(covs)
        rec = {
            "rng_seed": rs,
            "mean_coverage": float(covs.mean()),
            "gt_0.9_rate": float((covs > 0.9).mean()),
            "strict_gt_0.95_rate": float((covs > 0.95).mean()),
            "env_success_rate": float(np.mean(succ)),
            "per_episode_cov": [float(x) for x in covs.tolist()],
        }
        per_rng.append(rec)
        print(f"rng={rs}  mean_cov={rec['mean_coverage']:.4f}  >0.9={rec['gt_0.9_rate']:.3f}  strict={rec['strict_gt_0.95_rate']:.3f}", flush=True)

    mc = np.array([r["mean_coverage"] for r in per_rng])
    g9 = np.array([r["gt_0.9_rate"] for r in per_rng])
    st = np.array([r["strict_gt_0.95_rate"] for r in per_rng])
    agg = {
        "checkpoint": args.checkpoint,
        "weights": args.weights,
        "method": args.method,
        "replan_every": args.replan_every,
        "episodes": args.episodes,
        "n_rng": len(per_rng),
        "mean_coverage_mean": float(mc.mean()),
        "mean_coverage_std": float(mc.std()),
        "gt_0.9_mean": float(g9.mean()),
        "strict_mean": float(st.mean()),
        "per_rng": per_rng,
    }
    print(f"\nDP {args.weights} replan={args.replan_every}: mean_cov={mc.mean():.4f} +/- {mc.std():.4f}  >0.9={g9.mean():.3f}  strict={st.mean():.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(agg, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
