#!/usr/bin/env python
"""Collect real-env-verified recovery (stall_obs -> recovery_chunk) labels.

Candidate-generation (lever 4) data step. The codebook + prior are trained only
on expert demo states, so closed-loop stall states (block stuck / pushed the
wrong way, no codebook chunk makes progress) are off-distribution and unsolvable.

We roll out the geometric recovery policy in the REAL Push-T env on TRAIN seeds.
Every time the recovery wrapper fires (its behind->contact->push maneuver), we
record (stall_obs, recovery_chunk, raw_state) and then measure, in the same warm
rollout, whether that maneuver causally moved the block toward the goal over the
next chunk. Only causally-helpful recoveries are kept as labels.

Why real-env, not sim lookahead: the calibrated set_env_state_exact reset is
cold/memoryless and disagrees with the warm rollout (it reports the prior
self-recovers from stalls when, warm, it does not), so cold-sim cannot verify
these labels. The geometric pushes are used ONLY here, offline, as expert labels;
inference will use only the model + prior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vq_sigreg.pusht_data import denormalize_action, normalize_state
from vq_sigreg.pusht_env import make_env
from vq_sigreg.pusht_policy import VQSigRegPolicy, load_policy

GOAL_PX = np.asarray([256.0, 256.0], dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="outputs/vq_sigreg_pusht_k16_continuous_residual_s003_g48_a035/vq_continuous_residual_latest.pt")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--start-seed", type=int, default=10100)
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--replan-every", type=int, default=16)
    p.add_argument("--recovery-max-uses", type=int, default=6)
    p.add_argument("--recovery-min-plans", type=int, default=2)
    p.add_argument("--recovery-max-plans", type=int, default=18)
    p.add_argument("--recovery-push-px", type=float, default=200.0)
    p.add_argument("--progress-margin-px", type=float, default=12.0, help="Keep label if block moved this much toward goal over the chunk.")
    p.add_argument("--out", default="outputs/_recovery/recovery_data.pt")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    policy, cfg = load_policy(args.checkpoint, device)
    if not isinstance(policy, VQSigRegPolicy):
        raise SystemExit("need a VQSigRegPolicy checkpoint")
    policy.selector = "recovery_prior"
    policy.recovery_max_uses = int(args.recovery_max_uses)
    policy.recovery_min_plans = int(args.recovery_min_plans)
    policy.recovery_max_plans = int(args.recovery_max_plans)
    policy.recovery_push_px = float(args.recovery_push_px)
    policy.recovery_min_goal_dist_px = 0.0
    obs_steps = int(cfg["data"]["obs_steps"])
    replan_every = int(args.replan_every)

    obs_recs: list[np.ndarray] = []
    chunk_recs: list[np.ndarray] = []
    progress: list[float] = []
    all_obs: list[np.ndarray] = []
    all_chunk: list[np.ndarray] = []
    all_prog: list[float] = []
    n_fired = 0

    for ep in range(int(args.episodes)):
        seed = int(args.start_seed) + ep
        obs_raw, _ = env_reset(policy, seed)
        env = policy._env  # set by env_reset
        history = [normalize_state(np.asarray(obs_raw))] * obs_steps
        steps = 0
        while steps < args.max_steps:
            obs_vec = np.concatenate(history[-obs_steps:]).astype(np.float32)
            obs_t = torch.from_numpy(obs_vec)[None].to(device)
            plan = policy.plan(obs_t, np.asarray(obs_raw))
            actions = denormalize_action(plan.chunk.reshape(-1, 2).detach().cpu().numpy())
            is_recovery = int(plan.selected) == -1
            block_before = np.asarray(obs_raw[2:4], dtype=np.float32).copy()
            goal_before = float(np.linalg.norm(GOAL_PX - block_before))

            done = False
            for a in actions[:replan_every]:
                obs_raw, reward, term, trunc, _ = env.step(np.clip(a, 0.0, 512.0).astype(np.float32))
                steps += 1
                history.append(normalize_state(np.asarray(obs_raw)))
                if term or trunc or steps >= args.max_steps:
                    done = bool(term or trunc)
                    break

            if is_recovery:
                n_fired += 1
                block_after = np.asarray(obs_raw[2:4], dtype=np.float32)
                goal_after = float(np.linalg.norm(GOAL_PX - block_after))
                prog = goal_before - goal_after
                progress.append(prog)
                all_obs.append(obs_vec.copy())
                all_chunk.append(plan.chunk.detach().cpu().numpy().astype(np.float32))
                all_prog.append(float(prog))
                if prog >= args.progress_margin_px:
                    obs_recs.append(obs_vec.copy())
                    chunk_recs.append(plan.chunk.detach().cpu().numpy().astype(np.float32))
            if done:
                break
        env.close()
        if (ep + 1) % 40 == 0:
            print(f"ep {ep+1}/{args.episodes}  fired={n_fired}  kept={len(obs_recs)}")

    if progress:
        pr = np.asarray(progress)
        print(json.dumps({
            "recovery_fired": n_fired,
            "kept": len(obs_recs),
            "progress_p50_px": float(np.median(pr)),
            "progress_p90_px": float(np.quantile(pr, 0.9)),
            "frac_helpful": float((pr >= args.progress_margin_px).mean()),
        }, indent=2))
    if not obs_recs:
        raise SystemExit("collected 0 recovery labels; loosen thresholds")

    out = {
        "obs": torch.from_numpy(np.stack(obs_recs)).float(),
        "chunk": torch.from_numpy(np.stack(chunk_recs)).float(),
        "progress_px": torch.tensor(progress, dtype=torch.float32),
        # All recovery firings (helpful and not), for gate training:
        "all_obs": torch.from_numpy(np.stack(all_obs)).float(),
        "all_chunk": torch.from_numpy(np.stack(all_chunk)).float(),
        "all_progress_px": torch.tensor(all_prog, dtype=torch.float32),
        "progress_margin_px": float(args.progress_margin_px),
        "source_checkpoint": str(args.checkpoint),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(json.dumps({"num_records": len(obs_recs), "out": str(args.out)}, indent=2))


def env_reset(policy, seed: int):
    env = make_env()
    obs_raw, _ = env.reset(seed=int(seed))
    policy.reset()
    policy._env = env
    return obs_raw, None


if __name__ == "__main__":
    main()
