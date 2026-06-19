#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from vq_sigreg.pusht_policy import load_policy
from vq_sigreg.pusht_rollout import rollout_policy


def summarize(rollouts: list[dict]) -> dict[str, float]:
    covs = np.asarray([r["max_coverage"] for r in rollouts], dtype=np.float32)
    return {
        "episodes": float(len(rollouts)),
        "mean_max_coverage": float(covs.mean()),
        "success_rate": float(np.mean([r["success"] for r in rollouts])),
        "frac_solved_above_0.9": float((covs > 0.9).mean()),
        "frac_stalled_below_0.1": float((covs < 0.1).mean()),
        "mean_when_not_stalled": float(covs[covs > 0.1].mean()) if (covs > 0.1).any() else 0.0,
        "mean_selected_code_switches": float(np.mean([r["selected_code_switches"] for r in rollouts])),
        "mean_selected_unique_codes": float(np.mean([r["selected_unique_codes"] for r in rollouts])),
        "mean_recovery_uses": float(np.mean([r.get("recovery_uses", 0.0) for r in rollouts])),
    }


def write_gif(rollouts: list[dict], out: Path, keep: int, fps: int, every: int) -> None:
    kept = sorted(rollouts, key=lambda r: r["max_coverage"], reverse=True)[:keep]
    if not kept or "frames" not in kept[0]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Render/evaluate closed-loop Push-T rollouts for current VQ-SIGReg lane.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--start-seed", type=int, default=10_000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--replan-every", type=int, default=None)
    parser.add_argument("--out", default=None, help="GIF output path. Omit or set empty to skip GIF.")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--keep", type=int, default=3)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-episode logs and omit per_episode from stdout.")
    parser.add_argument(
        "--selector",
        default=None,
        choices=["prior", "reranker", "reranker_margin", "cycle", "oracle", "temporal", "recovery_prior"],
    )
    parser.add_argument("--reranker-switch-margin", type=float, default=None)
    parser.add_argument("--oracle-top-m", type=int, default=None)
    parser.add_argument("--oracle-score-steps", type=int, default=None)
    parser.add_argument("--oracle-min-advantage", type=float, default=None)
    parser.add_argument("--oracle-terminal-score", action="store_true")
    parser.add_argument("--mode-dwell-steps", type=int, default=None)
    parser.add_argument("--mode-hysteresis-margin", type=float, default=None)
    parser.add_argument("--sticky-logit-bonus", type=float, default=None)
    parser.add_argument("--sticky-switch-margin", type=float, default=None)
    parser.add_argument("--sticky-macro-chunks", type=int, default=None)
    parser.add_argument("--recovery-min-plans", type=int, default=None)
    parser.add_argument("--recovery-max-plans", type=int, default=None)
    parser.add_argument("--recovery-max-uses", type=int, default=None)
    parser.add_argument("--recovery-block-disp-px", type=float, default=None)
    parser.add_argument("--recovery-recent-disp-px", type=float, default=None)
    parser.add_argument("--recovery-goal-progress-px", type=float, default=None)
    parser.add_argument("--recovery-standoff-px", type=float, default=None)
    parser.add_argument("--recovery-contact-px", type=float, default=None)
    parser.add_argument("--recovery-push-px", type=float, default=None)
    parser.add_argument("--recovery-min-goal-dist-px", type=float, default=None)
    parser.add_argument("--temporal-ensemble", action="store_true")
    parser.add_argument("--ensemble-replan-every", type=int, default=1)
    parser.add_argument("--ensemble-decay", type=float, default=0.6)
    parser.add_argument("--ensemble-near-goal-only", action="store_true")
    parser.add_argument("--ensemble-goal-dist-px", type=float, default=64.0)
    parser.add_argument("--ensemble-goal-angle-rad", type=float, default=0.5)
    parser.add_argument("--ensemble-within-code", action="store_true")
    parser.add_argument("--hier-flow-noise-scale", type=float, default=None)
    parser.add_argument("--hier-flow-samples", type=int, default=None)
    parser.add_argument("--hier-flow-steps", type=int, default=None, help="Override ODE solver steps at inference.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    torch.manual_seed(int(args.start_seed))
    policy, cfg = load_policy(args.checkpoint, device)
    model = getattr(policy, "model", None)
    if model is not None and args.hier_flow_noise_scale is not None and hasattr(model, "hierarchical_flow_noise_scale"):
        model.hierarchical_flow_noise_scale = float(args.hier_flow_noise_scale)
    if model is not None and args.hier_flow_samples is not None and hasattr(model, "hierarchical_flow_samples"):
        model.hierarchical_flow_samples = max(1, int(args.hier_flow_samples))
    if model is not None and args.hier_flow_steps is not None and hasattr(model, "hierarchical_flow_steps"):
        model.hierarchical_flow_steps = max(1, int(args.hier_flow_steps))
    if hasattr(policy, "selector") and args.selector is not None:
        policy.selector = str(args.selector)
    if hasattr(policy, "sticky_logit_bonus") and args.sticky_logit_bonus is not None:
        policy.sticky_logit_bonus = float(args.sticky_logit_bonus)
    if hasattr(policy, "sticky_switch_margin") and args.sticky_switch_margin is not None:
        policy.sticky_switch_margin = float(args.sticky_switch_margin)
    if hasattr(policy, "sticky_macro_chunks") and args.sticky_macro_chunks is not None:
        policy.sticky_macro_chunks = max(0, int(args.sticky_macro_chunks))
    for attr in [
        "recovery_min_plans",
        "recovery_max_plans",
        "recovery_max_uses",
        "recovery_block_disp_px",
        "recovery_recent_disp_px",
        "recovery_goal_progress_px",
        "recovery_standoff_px",
        "recovery_contact_px",
        "recovery_push_px",
        "recovery_min_goal_dist_px",
    ]:
        value = getattr(args, attr, None)
        if hasattr(policy, attr) and value is not None:
            setattr(policy, attr, type(getattr(policy, attr))(value))
    if hasattr(policy, "reranker_switch_margin") and args.reranker_switch_margin is not None:
        policy.reranker_switch_margin = float(args.reranker_switch_margin)
    if hasattr(policy, "oracle_top_m") and args.oracle_top_m is not None:
        policy.oracle_top_m = max(0, int(args.oracle_top_m))
    if hasattr(policy, "oracle_score_steps") and args.oracle_score_steps is not None:
        policy.oracle_score_steps = max(1, int(args.oracle_score_steps))
    if hasattr(policy, "oracle_min_advantage") and args.oracle_min_advantage is not None:
        policy.oracle_min_advantage = float(args.oracle_min_advantage)
    if hasattr(policy, "oracle_terminal_score") and bool(args.oracle_terminal_score):
        policy.oracle_terminal_score = True
    if hasattr(policy, "mode_dwell_steps") and args.mode_dwell_steps is not None:
        policy.mode_dwell_steps = max(0, int(args.mode_dwell_steps))
    if hasattr(policy, "mode_hysteresis_margin") and args.mode_hysteresis_margin is not None:
        policy.mode_hysteresis_margin = float(args.mode_hysteresis_margin)
    rollouts = []
    for ep in range(int(args.episodes)):
        seed = int(args.start_seed) + ep
        result = rollout_policy(
            policy,
            cfg,
            seed=seed,
            max_steps=args.max_steps,
            replan_every=args.replan_every,
            capture_frames=not args.no_gif,
            temporal_ensemble=bool(args.temporal_ensemble),
            ensemble_replan_every=int(args.ensemble_replan_every),
            ensemble_decay=float(args.ensemble_decay),
            ensemble_near_goal_only=bool(args.ensemble_near_goal_only),
            ensemble_goal_dist_px=float(args.ensemble_goal_dist_px),
            ensemble_goal_angle_rad=float(args.ensemble_goal_angle_rad),
            ensemble_within_code=bool(args.ensemble_within_code),
        )
        if not args.quiet:
            print(f"episode seed {seed}: max coverage {result['max_coverage']:.3f}")
        rollouts.append(result)

    metrics = summarize(rollouts)
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["model_type"] = getattr(policy, "model_type", "unknown")
    metrics["per_episode"] = [
        {
            "seed": float(int(args.start_seed) + idx),
            "max_coverage": float(result["max_coverage"]),
            "success": float(result["success"]),
            "selected_code_switches": float(result["selected_code_switches"]),
            "selected_unique_codes": float(result["selected_unique_codes"]),
            "recovery_uses": float(result.get("recovery_uses", 0.0)),
        }
        for idx, result in enumerate(rollouts)
    ]
    if hasattr(policy, "sticky_logit_bonus"):
        metrics["selector"] = str(policy.selector)
        metrics["sticky_logit_bonus"] = float(policy.sticky_logit_bonus)
        metrics["sticky_switch_margin"] = float(policy.sticky_switch_margin)
        metrics["sticky_macro_chunks"] = float(policy.sticky_macro_chunks)
        metrics["reranker_switch_margin"] = float(getattr(policy, "reranker_switch_margin", 0.0))
        metrics["oracle_top_m"] = float(getattr(policy, "oracle_top_m", 0.0))
        metrics["oracle_score_steps"] = float(getattr(policy, "oracle_score_steps", 0.0))
        metrics["oracle_min_advantage"] = float(getattr(policy, "oracle_min_advantage", 0.0))
        metrics["oracle_terminal_score"] = float(bool(getattr(policy, "oracle_terminal_score", False)))
        metrics["mode_dwell_steps"] = float(getattr(policy, "mode_dwell_steps", 0.0))
        metrics["mode_hysteresis_margin"] = float(getattr(policy, "mode_hysteresis_margin", 0.0))
        metrics["recovery_min_plans"] = float(getattr(policy, "recovery_min_plans", 0.0))
        metrics["recovery_max_plans"] = float(getattr(policy, "recovery_max_plans", 0.0))
        metrics["recovery_max_uses"] = float(getattr(policy, "recovery_max_uses", 0.0))
        metrics["recovery_block_disp_px"] = float(getattr(policy, "recovery_block_disp_px", 0.0))
        metrics["recovery_recent_disp_px"] = float(getattr(policy, "recovery_recent_disp_px", 0.0))
        metrics["recovery_goal_progress_px"] = float(getattr(policy, "recovery_goal_progress_px", 0.0))
        metrics["recovery_standoff_px"] = float(getattr(policy, "recovery_standoff_px", 0.0))
        metrics["recovery_contact_px"] = float(getattr(policy, "recovery_contact_px", 0.0))
        metrics["recovery_push_px"] = float(getattr(policy, "recovery_push_px", 0.0))
        metrics["temporal_ensemble"] = float(bool(args.temporal_ensemble))
        metrics["ensemble_replan_every"] = float(args.ensemble_replan_every)
        metrics["ensemble_decay"] = float(args.ensemble_decay)
        metrics["ensemble_near_goal_only"] = float(bool(args.ensemble_near_goal_only))
        metrics["ensemble_goal_dist_px"] = float(args.ensemble_goal_dist_px)
        metrics["ensemble_goal_angle_rad"] = float(args.ensemble_goal_angle_rad)
        metrics["ensemble_within_code"] = float(bool(args.ensemble_within_code))
        metrics["hier_flow_noise_scale"] = (
            float(args.hier_flow_noise_scale) if args.hier_flow_noise_scale is not None else -1.0
        )
        metrics["hier_flow_samples"] = float(args.hier_flow_samples) if args.hier_flow_samples is not None else -1.0
        metrics["hier_flow_steps"] = float(args.hier_flow_steps) if args.hier_flow_steps is not None else -1.0
    json_out = Path(args.json_out) if args.json_out else Path(args.checkpoint).with_suffix(".closed_loop.json")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if not args.no_gif:
        gif_out = Path(args.out) if args.out else Path(args.checkpoint).with_suffix(".gif")
        write_gif(rollouts, gif_out, int(args.keep), int(args.fps), int(args.every))
        metrics["gif"] = str(gif_out)
        json_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    stdout_metrics = dict(metrics)
    if args.quiet:
        stdout_metrics.pop("per_episode", None)
    print(json.dumps(stdout_metrics, indent=2))


if __name__ == "__main__":
    main()
