#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

from vq_sigreg.pusht_env import make_env, set_env_state_exact


def replay_episode(states: np.ndarray, actions: np.ndarray, lo: int, hi: int) -> dict[str, float]:
    env = make_env(render_mode=None)
    env.reset(seed=0)
    obs = set_env_state_exact(env, states[lo], reset_space=True)
    init_l2 = float(np.linalg.norm(obs - states[lo]))
    max_reward = 0.0
    final_reward = 0.0
    max_coverage = 0.0
    final_coverage = 0.0
    success = False
    steps = 0
    for action in actions[lo:hi]:
        obs, reward, terminated, truncated, info = env.step(np.clip(action, 0.0, 512.0).astype(np.float32))
        steps += 1
        final_reward = float(reward)
        max_reward = max(max_reward, final_reward)
        final_coverage = float(info.get("coverage", final_reward * getattr(env.unwrapped, "success_threshold", 0.95)))
        max_coverage = max(max_coverage, final_coverage)
        success = success or bool(info.get("is_success", False))
        if terminated or truncated:
            break
    env.close()
    return {
        "init_l2": init_l2,
        "steps": float(steps),
        "max_reward": max_reward,
        "final_reward": final_reward,
        "max_coverage": max_coverage,
        "final_coverage": final_coverage,
        "success": float(success),
    }


def one_step_transition(states: np.ndarray, actions: np.ndarray, t: int) -> dict[str, float]:
    env = make_env(render_mode=None)
    env.reset(seed=0)
    obs0 = set_env_state_exact(env, states[t], reset_space=True)
    obs1, reward, terminated, truncated, info = env.step(np.clip(actions[t], 0.0, 512.0).astype(np.float32))
    env.close()
    angle_err = float(abs(((obs1[4] - states[t + 1, 4] + np.pi) % (2.0 * np.pi)) - np.pi))
    return {
        "reset_l2": float(np.linalg.norm(obs0 - states[t])),
        "next_l2": float(np.linalg.norm(obs1 - states[t + 1])),
        "agent_next_l2": float(np.linalg.norm(obs1[:2] - states[t + 1, :2])),
        "block_next_l2": float(np.linalg.norm(obs1[2:4] - states[t + 1, 2:4])),
        "angle_next_abs": angle_err,
        "reward": float(reward),
        "success": float(bool(info.get("is_success", False))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P0 Push-T harness sanity check: demo replay and one-step transitions.")
    parser.add_argument("--zarr-path", default="data/pusht/pusht_cchi_v7_replay.zarr")
    parser.add_argument("--episodes", type=int, default=206)
    parser.add_argument("--transition-episodes", type=int, default=32)
    parser.add_argument("--out", default="outputs/pusht_harness/sanity_check_replay.json")
    parser.add_argument("--min-success-rate", type=float, default=0.9)
    parser.add_argument("--min-reward-09-rate", type=float, default=0.9)
    parser.add_argument("--min-mean-max-coverage", type=float, default=0.9)
    parser.add_argument("--max-transition-agent-l2", type=float, default=5.0)
    parser.add_argument("--max-transition-block-l2", type=float, default=5.0)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    root = zarr.open(str(args.zarr_path), "r")
    states = np.asarray(root["data/state"], dtype=np.float64)
    actions = np.asarray(root["data/action"], dtype=np.float64)
    ends = np.asarray(root["meta/episode_ends"], dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]])

    num_eps = min(int(args.episodes), len(ends))
    replays = [replay_episode(states, actions, int(starts[ep]), int(ends[ep])) for ep in range(num_eps)]

    transitions: list[dict[str, float]] = []
    for ep in range(min(int(args.transition_episodes), len(ends))):
        lo, hi = int(starts[ep]), int(ends[ep])
        for offset in [0, 1, 5, 10, 20, 40, 80]:
            if lo + offset + 1 < hi:
                transitions.append(one_step_transition(states, actions, lo + offset))

    success_rate = float(np.mean([row["success"] for row in replays])) if replays else 0.0
    reward_09_rate = float(np.mean([row["max_reward"] > 0.9 for row in replays])) if replays else 0.0
    mean_max_coverage = float(np.mean([row["max_coverage"] for row in replays])) if replays else 0.0
    mean_max_reward = float(np.mean([row["max_reward"] for row in replays])) if replays else 0.0
    mean_agent_l2 = float(np.mean([row["agent_next_l2"] for row in transitions])) if transitions else 0.0
    mean_block_l2 = float(np.mean([row["block_next_l2"] for row in transitions])) if transitions else 0.0
    summary = {
        "zarr_path": str(args.zarr_path),
        "episodes": float(num_eps),
        "success_rate": success_rate,
        "max_reward_gt_0.9_rate": reward_09_rate,
        "mean_max_reward": mean_max_reward,
        "mean_max_coverage": mean_max_coverage,
        "mean_init_l2": float(np.mean([row["init_l2"] for row in replays])) if replays else 0.0,
        "transition_count": float(len(transitions)),
        "transition_mean_reset_l2": float(np.mean([row["reset_l2"] for row in transitions])) if transitions else 0.0,
        "transition_mean_next_l2": float(np.mean([row["next_l2"] for row in transitions])) if transitions else 0.0,
        "transition_mean_agent_next_l2": mean_agent_l2,
        "transition_mean_block_next_l2": mean_block_l2,
        "transition_mean_angle_next_abs": float(np.mean([row["angle_next_abs"] for row in transitions])) if transitions else 0.0,
        "gate_success": bool(
            success_rate >= float(args.min_success_rate)
            and reward_09_rate >= float(args.min_reward_09_rate)
            and mean_max_coverage >= float(args.min_mean_max_coverage)
            and mean_agent_l2 <= float(args.max_transition_agent_l2)
            and mean_block_l2 <= float(args.max_transition_block_l2)
        ),
    }
    payload = {"summary": summary, "replays": replays, "transitions": transitions}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not args.no_fail and not summary["gate_success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
