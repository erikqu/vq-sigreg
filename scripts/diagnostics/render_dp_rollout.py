#!/usr/bin/env python
"""Render closed-loop Push-T rollouts for the in-repo Diffusion Policy baseline.

Same warm-stepping harness as render_pusht_gif.py (official gym_pusht stepping).
Self-contained diagnostic - not a shared evaluator flag surface.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from vq_sigreg.diffusion_policy import DiffusionPolicy
from vq_sigreg.pusht_data import denormalize_action, normalize_state
from vq_sigreg.pusht_env import make_env


def coverage_of(info: dict, reward: float, env) -> float:
    return float(info.get("coverage", reward * getattr(env.unwrapped, "success_threshold", 0.95)))


@torch.no_grad()
def rollout_episode(
    model: DiffusionPolicy,
    cfg: dict,
    device: torch.device,
    seed: int,
    max_steps: int,
    replan_every: int,
    sample_steps: int | None,
    method: str,
    capture_frames: bool,
    rng_seed: int = 0,
) -> dict:
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    obs_steps = int(cfg["data"]["obs_steps"])
    horizon = int(cfg["data"]["horizon"])
    env = make_env(seed=seed, render_mode="rgb_array" if capture_frames else None)
    obs_raw, _ = env.reset(seed=seed)
    history = [normalize_state(np.asarray(obs_raw))] * obs_steps
    frames: list[np.ndarray] = []
    if capture_frames:
        frames.append(np.asarray(env.render()))

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
            obs_raw, reward, terminated, truncated, info = env.step(
                np.clip(np.asarray(action), 0.0, 512.0).astype(np.float32)
            )
            t += 1
            max_cov = max(max_cov, coverage_of(info, reward, env))
            success = success or bool(info.get("is_success", False))
            history.append(normalize_state(np.asarray(obs_raw)))
            if capture_frames:
                frames.append(np.asarray(env.render()))
            if terminated or truncated or t >= max_steps:
                break
    env.close()
    return {
        "seed": seed,
        "max_coverage": max_cov,
        "success": success,
        "frames": frames,
    }


def write_gif(rollouts: list[dict], out: Path, keep: int, fps: int, every: int) -> None:
    kept = sorted(rollouts, key=lambda r: r["max_coverage"], reverse=True)[:keep]
    if not kept or "frames" not in kept[0] or not kept[0]["frames"]:
        return
    length = max(len(r["frames"]) for r in kept)
    tiles = []
    for r in kept:
        frames = r["frames"]
        frames = frames + [frames[-1]] * (length - len(frames))
        tiles.append(np.stack(frames[::every]))
    grid = np.concatenate(tiles, axis=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, list(grid), fps=fps, loop=0)


def write_single_gif(rollout: dict, out: Path, fps: int, every: int) -> None:
    frames = rollout.get("frames") or []
    if not frames:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames[::every], fps=fps, loop=0)


def save_stills(rollout: dict, out_dir: Path, prefix: str) -> list[str]:
    frames = rollout.get("frames") or []
    if not frames:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = [0, len(frames) // 2, len(frames) - 1]
    names = ["start", "mid", "end"]
    saved = []
    for idx, name in zip(picks, names):
        path = out_dir / f"{prefix}_seed{rollout['seed']}_{name}.png"
        imageio.imwrite(path, frames[idx])
        saved.append(str(path))
    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--start-seed", type=int, default=10_000)
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="Explicit seeds (overrides episodes/start-seed).")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--replan-every", type=int, default=16)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--weights", choices=["ema", "model"], default="ema")
    ap.add_argument("--method", choices=["ddpm", "ddim"], default="ddpm")
    ap.add_argument("--sample-steps", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--single-out-dir", default=None, help="Write one GIF per seed into this directory.")
    ap.add_argument("--stills-dir", default=None, help="Write start/mid/end PNGs per seed.")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--keep", type=int, default=3)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = DiffusionPolicy(cfg).to(device)
    model.load_state_dict(ckpt["ema_state" if args.weights == "ema" else "model_state"])
    model.eval()
    steps = None if int(args.sample_steps) <= 0 else int(args.sample_steps)

    seeds = [int(s) for s in args.seeds] if args.seeds else [int(args.start_seed) + i for i in range(int(args.episodes))]
    want_frames = (not args.no_gif) or bool(args.single_out_dir) or bool(args.stills_dir)
    rollouts = []
    for seed in seeds:
        result = rollout_episode(
            model, cfg, device, seed, int(args.max_steps), int(args.replan_every),
            steps, args.method, capture_frames=want_frames, rng_seed=int(args.rng_seed),
        )
        rollouts.append(result)
        print(f"seed {seed}: max_coverage={result['max_coverage']:.3f}  success={result['success']}")

    covs = np.asarray([r["max_coverage"] for r in rollouts])
    metrics = {
        "checkpoint": str(args.checkpoint),
        "model_type": "diffusion_policy",
        "replan_every": int(args.replan_every),
        "method": args.method,
        "mean_max_coverage": float(covs.mean()),
        "success_rate": float(np.mean([r["success"] for r in rollouts])),
        "frac_solved_above_0.9": float((covs > 0.9).mean()),
        "per_episode": [
            {"seed": float(r["seed"]), "max_coverage": float(r["max_coverage"]), "success": float(r["success"])}
            for r in rollouts
        ],
    }

    if args.single_out_dir:
        for r in rollouts:
            out = Path(args.single_out_dir) / f"seed{r['seed']}.gif"
            write_single_gif(r, out, int(args.fps), int(args.every))
            print(f"gif -> {out}")

    if args.stills_dir:
        still_paths = []
        for r in rollouts:
            still_paths.extend(save_stills(r, Path(args.stills_dir), "diffusion"))
        metrics["stills"] = still_paths

    if not args.no_gif and args.out:
        write_gif(rollouts, Path(args.out), int(args.keep), int(args.fps), int(args.every))
        metrics["gif"] = str(args.out)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
