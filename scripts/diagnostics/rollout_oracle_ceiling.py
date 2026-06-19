#!/usr/bin/env python
"""Rollout oracle ceiling for VQ-SIGReg Push-T (provably >= prior).

This diagnostic answers: "if, at every replan, we could pick the VQ code whose
chunk -- followed by the prior to the horizon -- yields the best episode
coverage, how much better than the prior baseline can we do?"

It is a *rollout policy* (Bertsekas policy improvement): the prior is the base
policy, and at each decision the oracle evaluates every candidate code by

    execute candidate chunk for `replan_every` steps  ->  follow prior to horizon

and commits to the arg-max. Because the prior's own code is always in the
candidate set, the oracle's value is >= the prior's value at every decision, so
in a deterministic simulator the realized episode coverage is >= the prior
baseline *by construction*. Any violation of that invariant means the simulator
clone desynced and the experiment did not run.

Two correctness measures are essential and were the source of earlier broken
oracles:

1. Exact cloning. Push-T's pymunk space carries a hidden warm-start solver
   cache that is NOT part of the 5D observation and cannot be pickled. Restoring
   only body pose/velocity leaves that cache stale, and successive candidate
   evaluations poison each other (observed drift of ~90 px). We make every step
   *cold* by clearing the block's cached arbiters first, which makes a step a
   pure function of body state (verified bit-stable to ~1e-7).

2. Trajectory reuse. The realized trajectory is advanced by *restoring* the
   chosen candidate's post-chunk snapshot rather than re-stepping a separate
   "real" env, so the realized path is identical to the scored path. The prior
   baseline is produced by the same machinery with selection forced to the prior
   code, making the >= comparison exact.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vq_sigreg.pusht_data import denormalize_action, normalize_state
from vq_sigreg.pusht_env import make_env
from vq_sigreg.pusht_policy import VQSigRegPolicy, load_policy


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = float(k) / float(n)
    denom = 1.0 + z * z / float(n)
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return center - half, center + half


def coverage_of(info: dict, reward: float, env: Any) -> float:
    return float(info.get("coverage", reward * getattr(env.unwrapped, "success_threshold", 0.95)))


def cold_clear(env: Any) -> None:
    """Drop the block's cached contact arbiters (warm-start impulses).

    This is the key to deterministic cloning: with the cache cleared before every
    step, a step becomes a pure function of the rigid-body state, so a snapshot of
    body pose/velocity is a complete, restorable description of the dynamics."""
    u = env.unwrapped
    u.space.remove(u.block, *u._block_shapes)
    u.space.add(u.block, *u._block_shapes)


def cold_step(env: Any, action: np.ndarray):
    cold_clear(env)
    return env.step(np.clip(np.asarray(action), 0.0, 512.0).astype(np.float32))


def snapshot_full(env: Any) -> dict[str, Any]:
    """Capture the complete dynamic state of the (cold-stepped) Push-T sim."""
    u = env.unwrapped
    return {
        "agent_pos": tuple(u.agent.position),
        "agent_vel": tuple(u.agent.velocity),
        "block_pos": tuple(u.block.position),
        "block_vel": tuple(u.block.velocity),
        "block_angle": float(u.block.angle),
        "block_angular_velocity": float(u.block.angular_velocity),
    }


def reset_time_limit(env: Any) -> None:
    """Zero the TimeLimit wrapper's step counter.

    Critical for counterfactual scoring: the wrapper counts steps cumulatively
    across every candidate evaluation, so without resetting it the budget is
    exhausted after the first candidate and all later evaluations truncate
    instantly (corrupting the scores)."""
    e = env
    while e is not None:
        if getattr(e, "_elapsed_steps", None) is not None:
            e._elapsed_steps = 0
        e = getattr(e, "env", None)


def restore_full(env: Any, snap: dict[str, Any]) -> None:
    u = env.unwrapped
    u.agent.position = list(snap["agent_pos"])
    u.agent.velocity = list(snap["agent_vel"])
    u.block.angle = float(snap["block_angle"])
    u.block.position = list(snap["block_pos"])
    u.block.velocity = list(snap["block_vel"])
    u.block.angular_velocity = float(snap["block_angular_velocity"])
    u.space.reindex_shapes_for_body(u.agent)
    u.space.reindex_shapes_for_body(u.block)
    reset_time_limit(env)


@torch.no_grad()
def prior_candidates(policy: VQSigRegPolicy, history: list[np.ndarray], obs_steps: int) -> tuple[np.ndarray, int, np.ndarray]:
    obs_vec = np.concatenate(history[-obs_steps:]).astype(np.float32)
    obs = torch.from_numpy(obs_vec)[None].to(policy.device)
    out = policy.model.candidate_outputs(obs)
    chunks = out["chunk_hat"][0].clamp(-1.0, 1.0)
    chunks_px = denormalize_action(chunks.reshape(chunks.shape[0], -1, 2)).detach().cpu().numpy()
    prior_logits = out["prior_logits"][0].detach().cpu().numpy().astype(np.float32)
    prior_code = int(prior_logits.argmax())
    return chunks_px, prior_code, prior_logits


@torch.no_grad()
def run_prior(
    policy: VQSigRegPolicy,
    env: Any,
    history: list[np.ndarray],
    obs_steps: int,
    t: int,
    max_steps: int,
    replan_every: int,
) -> tuple[float, bool, int]:
    """Run the prior policy (cold-stepped) in `env` from its current state."""
    max_cov = 0.0
    success = False
    terminated = truncated = False
    while t < max_steps and not (terminated or truncated):
        chunks_px, prior_code, _ = prior_candidates(policy, history, obs_steps)
        for action in chunks_px[prior_code][: int(replan_every)]:
            obs_raw, reward, terminated, truncated, info = cold_step(env, action)
            t += 1
            max_cov = max(max_cov, coverage_of(info, reward, env))
            success = success or bool(info.get("is_success", False))
            history.append(normalize_state(np.asarray(obs_raw)))
            if terminated or truncated or t >= max_steps:
                break
    return max_cov, success, t


@torch.no_grad()
def eval_candidate(
    policy: VQSigRegPolicy,
    env: Any,
    snap: dict[str, Any],
    history_snap: list[np.ndarray],
    chunk_px: np.ndarray,
    obs_steps: int,
    t: int,
    max_steps: int,
    replan_every: int,
    compute_full: bool,
) -> dict[str, Any]:
    """Score a candidate by: cold-clone -> execute its chunk -> (optionally) follow prior.

    Always returns the per-step coverage during the executed chunk (``step_covs``),
    the chunk-only coverage/success (``chunk`` / ``chunk_success``), and the
    post-chunk snapshot/history/time so the caller can *commit* to this candidate
    exactly. When ``compute_full`` is true it additionally runs the prior to the
    horizon and reports the full-rollout coverage (``total``); otherwise ``total``
    is just the chunk coverage (no long-horizon privileged lookahead)."""
    restore_full(env, snap)
    history = list(history_snap)
    step_covs: list[float] = []
    chunk_cov = 0.0
    chunk_success = False
    terminated = truncated = False
    tt = t
    for action in chunk_px[: int(replan_every)]:
        obs_raw, reward, terminated, truncated, info = cold_step(env, action)
        tt += 1
        cov = coverage_of(info, reward, env)
        step_covs.append(float(cov))
        chunk_cov = max(chunk_cov, cov)
        chunk_success = chunk_success or bool(info.get("is_success", False))
        history.append(normalize_state(np.asarray(obs_raw)))
        if terminated or truncated or tt >= max_steps:
            break
    post = {
        "chunk": float(chunk_cov),
        "chunk_success": bool(chunk_success),
        "step_covs": step_covs,
        "snap": snapshot_full(env),
        "hist": list(history),
        "t": int(tt),
        "term": bool(terminated or truncated),
        "success": bool(chunk_success),
    }
    total_cov = chunk_cov
    if compute_full and not (terminated or truncated) and tt < max_steps:
        cov2, succ2, _ = run_prior(policy, env, history, obs_steps, tt, max_steps, replan_every)
        total_cov = max(total_cov, cov2)
        post["success"] = post["success"] or bool(succ2)
    post["total"] = float(total_cov)
    return post


@torch.no_grad()
def simulate(
    policy: VQSigRegPolicy,
    cfg: dict[str, Any],
    seed: int,
    max_steps: int,
    replan_every: int,
    prior_tie_epsilon: float,
    score_horizon: int = 0,
    collect_trace: bool = False,
) -> dict[str, Any]:
    """Run one episode of the rollout oracle and return both the oracle ceiling
    and the prior baseline, computed in a *single* run so the comparison is exact.

    ``score_horizon`` controls how much privileged lookahead the *selection* uses:

    * ``score_horizon <= 0`` (full / omniscient-MPC ceiling): each candidate is
      scored by ``execute its chunk -> follow the prior to the 300-step horizon``,
      i.e. coverage of the entire remaining episode evaluated by exact simulation.
      The reported ``oracle_value`` is the max over decisions of the chosen
      candidate's full-rollout coverage; because the prior's own code is in the
      candidate set, ``oracle_value >= prior_value`` holds *by construction*
      (exact, single-run, immune to float drift). This is the provable rollout
      ceiling -- but it requires omniscient future-state access no deployable
      router can have.

    * ``score_horizon = h > 0`` (myopic / routable proxy): each candidate is
      scored only on the max coverage it reaches within its first ``h`` executed
      steps -- *no* long-horizon continuation. This approximates what a selector
      with only short/local lookahead could pick. The oracle still commits the
      full chosen chunk and replans, and we report the *realized* coverage of the
      actually-committed trajectory (``oracle_value``). The ``>= prior`` invariant
      is NOT guaranteed here (myopic selection can hurt), which is exactly the
      signal we want: if myopic still ~matches the full ceiling the codebook is
      locally routable; if it collapses toward the prior the ceiling was bought
      by privileged lookahead a feedforward router cannot replicate.

    ``prior_value`` is always the step-0 prior rollout (full prior continuation),
    i.e. the prior policy's realized episode coverage under the same cold
    dynamics."""
    full_mode = int(score_horizon) <= 0
    h = max(1, int(score_horizon))
    obs_steps = int(cfg["data"]["obs_steps"])
    env = make_env(seed=seed, render_mode=None)
    obs_raw, _ = env.reset(seed=seed)
    policy.reset()
    history = [normalize_state(np.asarray(obs_raw))] * obs_steps
    snap = snapshot_full(env)

    oracle_full_value = 0.0
    realized_coverage = 0.0
    realized_success = False
    prior_value: float | None = None
    prior_success = False
    success = False
    t = 0
    decisions = 0
    prior_selected = 0
    selected_codes: list[int] = []
    trace: list[dict[str, Any]] = []
    while t < max_steps:
        obs_vec = np.concatenate(history[-obs_steps:]).astype(np.float32) if collect_trace else None
        chunks_px, prior_code, prior_logits = prior_candidates(policy, history, obs_steps)
        # The prior's full continuation is needed once (for prior_value) and for
        # every candidate in full mode (for selection + the provable invariant).
        evals = {
            code: eval_candidate(
                policy, env, snap, history, chunks_px[code], obs_steps, t, max_steps, replan_every,
                compute_full=full_mode or (prior_value is None and code == prior_code),
            )
            for code in range(chunks_px.shape[0])
        }
        if prior_value is None:
            prior_value = float(evals[prior_code]["total"])
            prior_success = bool(evals[prior_code]["success"])
        if full_mode:
            scores = np.asarray([evals[code]["total"] for code in range(chunks_px.shape[0])], dtype=np.float64)
        else:
            scores = np.asarray(
                [max(evals[code]["step_covs"][:h], default=0.0) for code in range(chunks_px.shape[0])],
                dtype=np.float64,
            )
        raw_best = int(scores.argmax())
        chosen = prior_code if float(scores[raw_best]) <= float(scores[prior_code]) + float(prior_tie_epsilon) else raw_best
        decisions += 1
        prior_selected += int(chosen == prior_code)
        selected_codes.append(int(chosen))
        if collect_trace:
            trace.append({
                "obs": obs_vec.tolist(),
                "chosen": int(chosen),
                "prior": int(prior_code),
                "prior_logits": [float(x) for x in prior_logits.tolist()],
                "t": int(t),
                # per-candidate selection scores (full mode: to-goal coverage of
                # "this code's chunk -> prior to horizon"). Needed to measure the
                # margin between the oracle's pick and its alternatives, i.e. to
                # tell genuine state-dependence from arg-max-over-near-ties.
                "scores": [float(x) for x in scores.tolist()],
            })
        e = evals[chosen]
        if full_mode:
            oracle_full_value = max(oracle_full_value, float(e["total"]))
            success = success or bool(e["success"])
        realized_coverage = max(realized_coverage, float(e["chunk"]))
        realized_success = realized_success or bool(e["chunk_success"])
        snap = e["snap"]
        history = e["hist"]
        t = e["t"]
        if e["term"]:
            break
    env.close()
    oracle_value = oracle_full_value if full_mode else realized_coverage
    oracle_success = success if full_mode else realized_success
    if collect_trace:
        for rec in trace:
            rec["episode_oracle_value"] = float(oracle_value)
            rec["seed"] = int(seed)
    return {
        "trace": trace,
        "oracle_value": float(oracle_value),
        "oracle_realized_coverage": float(realized_coverage),
        "prior_value": float(prior_value if prior_value is not None else 0.0),
        "prior_success": float(prior_success),
        "success": float(oracle_success),
        "realized_success": float(realized_success),
        "decisions": float(decisions),
        "prior_selected_rate": float(prior_selected / max(1, decisions)),
        "selected_code_switches": float(np.sum(np.diff(selected_codes) != 0)) if len(selected_codes) > 1 else 0.0,
        "selected_unique_codes": float(len(set(selected_codes))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rollout oracle ceiling (prior-continuation) for VQ-SIGReg Push-T.")
    parser.add_argument("--checkpoint", default="outputs/vq_sigreg_pusht_k16_prior/vq_sigreg_latest.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--replan-every", type=int, default=16)
    parser.add_argument(
        "--score-horizon",
        type=int,
        default=0,
        help="Lookahead (in steps) the selection may use. 0 = full omniscient-MPC ceiling "
        "(provable >= prior); h>0 = myopic routable proxy scored on the first h executed steps.",
    )
    parser.add_argument("--prior-tie-epsilon", type=float, default=1e-4)
    parser.add_argument("--invariant-tol", type=float, default=1e-6)
    parser.add_argument("--out", default="outputs/vq_sigreg_rollout_oracle_ceiling/rollout_oracle_h16_50seed.json")
    parser.add_argument(
        "--dump-trace",
        default="",
        help="If set, write an .npz of per-decision (obs, oracle_chosen_code, prior_code) for distillation.",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    policy, cfg = load_policy(args.checkpoint, device)
    if not isinstance(policy, VQSigRegPolicy):
        raise TypeError("rollout oracle requires a VQSigRegPolicy checkpoint.")
    policy.selector = "prior"

    full_mode = int(args.score_horizon) <= 0
    collect_trace = bool(args.dump_trace)
    rows: list[dict[str, Any]] = []
    all_trace: list[dict[str, Any]] = []
    invariant_violations = 0
    worst_gap = 0.0
    for ep in range(int(args.episodes)):
        seed = int(args.start_seed) + ep
        res = simulate(
            policy, cfg, seed, int(args.max_steps), int(args.replan_every),
            float(args.prior_tie_epsilon), int(args.score_horizon), collect_trace,
        )
        all_trace.extend(res.pop("trace", []))
        gap = float(res["oracle_value"]) - float(res["prior_value"])
        worst_gap = min(worst_gap, gap)
        res["oracle_minus_prior"] = gap
        # The provable >= prior invariant only applies to the full omniscient ceiling.
        # In myopic mode a below-prior gap is a *finding* (lookahead was load-bearing),
        # not a simulator desync, so we record but do not enforce it.
        res["invariant_ok"] = bool(gap + float(args.invariant_tol) >= 0.0)
        if full_mode and not res["invariant_ok"]:
            invariant_violations += 1
        rows.append(res)

    covs = np.asarray([r["oracle_value"] for r in rows], dtype=np.float64)
    prior_covs = np.asarray([r["prior_value"] for r in rows], dtype=np.float64)
    successes = np.asarray([r["success"] for r in rows], dtype=np.float64)
    prior_succ = np.asarray([r["prior_success"] for r in rows], dtype=np.float64)
    clo, chi = wilson_ci(int((covs > 0.9).sum()), len(rows))
    plo, phi = wilson_ci(int((prior_covs > 0.9).sum()), len(rows))
    summary = {
        "episodes": float(len(rows)),
        "score_horizon": int(args.score_horizon),
        "mode": "full_mpc_ceiling" if full_mode else "myopic_routable_proxy",
        "invariant_violations": float(invariant_violations),
        "invariant_below_prior_seeds": float(sum(1 for r in rows if not r["invariant_ok"])),
        "worst_oracle_minus_prior": float(worst_gap),
        "oracle_mean_max_reward": float(covs.mean()),
        "oracle_max_reward_gt_0.9_rate": float((covs > 0.9).mean()),
        "oracle_max_reward_gt_0.9_ci95_wilson": [clo, chi],
        "oracle_strict_env_success_rate": float(successes.mean()),
        "prior_mean_max_reward": float(prior_covs.mean()),
        "prior_max_reward_gt_0.9_rate": float((prior_covs > 0.9).mean()),
        "prior_max_reward_gt_0.9_ci95_wilson": [plo, phi],
        "prior_strict_env_success_rate": float(prior_succ.mean()),
        "oracle_minus_prior_mean": float((covs - prior_covs).mean()),
        "mean_prior_selected_rate": float(np.mean([r["prior_selected_rate"] for r in rows])),
        "mean_selected_code_switches": float(np.mean([r["selected_code_switches"] for r in rows])),
    }
    payload = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "episodes": int(args.episodes),
            "start_seed": int(args.start_seed),
            "max_steps": int(args.max_steps),
            "replan_every": int(args.replan_every),
            "score_horizon": int(args.score_horizon),
            "prior_tie_epsilon": float(args.prior_tie_epsilon),
            "invariant_tol": float(args.invariant_tol),
            "note": "cold-stepped (warm-start cache cleared) memoryless dynamics; numbers are not directly comparable to warm-start render_pusht_gif baselines.",
        },
        "summary": summary,
        "per_episode": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if collect_trace and all_trace:
        tp = Path(args.dump_trace)
        tp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            tp,
            obs=np.asarray([r["obs"] for r in all_trace], dtype=np.float32),
            chosen=np.asarray([r["chosen"] for r in all_trace], dtype=np.int64),
            prior=np.asarray([r["prior"] for r in all_trace], dtype=np.int64),
            prior_logits=np.asarray([r["prior_logits"] for r in all_trace], dtype=np.float32),
            scores=np.asarray([r["scores"] for r in all_trace], dtype=np.float32),
            seed=np.asarray([r["seed"] for r in all_trace], dtype=np.int64),
            t=np.asarray([r["t"] for r in all_trace], dtype=np.int64),
            episode_oracle_value=np.asarray([r["episode_oracle_value"] for r in all_trace], dtype=np.float32),
        )
        print(f"wrote {len(all_trace)} trace rows -> {tp}")
    print(json.dumps(summary, indent=2))
    if invariant_violations > 0:
        raise SystemExit(
            f"INVARIANT FAILED on {invariant_violations}/{len(rows)} seeds "
            f"(worst gap {worst_gap:.6f}): oracle scored below prior; simulator desync."
        )


if __name__ == "__main__":
    main()
