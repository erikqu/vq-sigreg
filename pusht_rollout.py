from __future__ import annotations

from typing import Any

import numpy as np
import torch

from vq_sigreg.pusht_data import denormalize_action, normalize_state
from vq_sigreg.pusht_env import make_env
from vq_sigreg.pusht_policy import LeJEPAPolicy, VQSigRegPolicy


@torch.no_grad()
def rollout_policy(
    policy: LeJEPAPolicy | VQSigRegPolicy,
    cfg: dict[str, Any],
    seed: int,
    max_steps: int | None = None,
    replan_every: int | None = None,
    capture_frames: bool = False,
    temporal_ensemble: bool = False,
    ensemble_replan_every: int = 1,
    ensemble_decay: float = 0.6,
    ensemble_near_goal_only: bool = False,
    ensemble_goal_dist_px: float = 64.0,
    ensemble_goal_angle_rad: float = 0.5,
    ensemble_within_code: bool = False,
) -> dict[str, Any]:
    env = make_env(render_mode="rgb_array" if capture_frames else None)
    obs_raw, _ = env.reset(seed=seed)
    score_env = None
    if isinstance(policy, VQSigRegPolicy) and getattr(policy, "selector", "") == "oracle":
        score_env = make_env(render_mode=None)
        score_env.reset(seed=seed)
    policy.reset()

    obs_steps = int(cfg["data"]["obs_steps"])
    max_steps = int(max_steps or cfg.get("eval", {}).get("rollout_max_steps", 300))
    replan_every = int(
        replan_every
        or cfg.get("eval", {}).get("execution_horizon", cfg.get("eval", {}).get("replan_every", 8))
    )
    history = [normalize_state(np.asarray(obs_raw))] * obs_steps
    frames: list[np.ndarray] = []
    if capture_frames:
        frames.append(np.asarray(env.render()))

    max_coverage = 0.0
    success = False
    steps_taken = 0
    winding = 0.0
    prev_angle = None
    selected_codes: list[int] = []

    def near_goal(raw: np.ndarray) -> bool:
        block = np.asarray(raw[2:4], dtype=np.float32)
        dist = float(np.linalg.norm(block - np.asarray([256.0, 256.0], dtype=np.float32)))
        angle = (float(raw[4]) - np.pi / 4.0 + np.pi) % (2.0 * np.pi) - np.pi
        return dist <= float(ensemble_goal_dist_px) and abs(angle) <= float(ensemble_goal_angle_rad)

    def make_plan() -> tuple[Any, np.ndarray]:
        obs = torch.from_numpy(np.concatenate(history[-obs_steps:])).float()[None]
        if score_env is not None and isinstance(policy, VQSigRegPolicy):
            plan_inner = policy.plan_oracle(
                obs,
                np.asarray(obs_raw),
                score_env,
                obs_steps=obs_steps,
                replan_every=replan_every,
            )
        else:
            plan_inner = policy.plan(obs, np.asarray(obs_raw))
        selected_codes.append(plan_inner.selected)
        actions_inner = denormalize_action(plan_inner.chunk.reshape(-1, 2).detach().cpu().numpy())
        return plan_inner, actions_inner

    def step_env(action: np.ndarray) -> tuple[bool, bool]:
        nonlocal obs_raw, steps_taken, max_coverage, success, winding, prev_angle
        obs_raw, reward, terminated, truncated, info = env.step(
            np.clip(action, 0.0, 512.0).astype(np.float32)
        )
        steps_taken += 1
        history.append(normalize_state(np.asarray(obs_raw)))
        if capture_frames:
            frames.append(np.asarray(env.render()))
        max_coverage = max(max_coverage, float(reward))
        success = success or bool(info.get("is_success", False))

        agent = np.asarray(obs_raw[:2], dtype=np.float64)
        block = np.asarray(obs_raw[2:4], dtype=np.float64)
        angle = float(np.arctan2(*(agent - block)[::-1]))
        if prev_angle is not None:
            delta = angle - prev_angle
            while delta > np.pi:
                delta -= 2 * np.pi
            while delta < -np.pi:
                delta += 2 * np.pi
            winding += delta
        prev_angle = angle
        return bool(terminated), bool(truncated)

    while steps_taken < max_steps:
        if not temporal_ensemble or (ensemble_near_goal_only and not near_goal(np.asarray(obs_raw))):
            _, actions = make_plan()
            terminated = truncated = False
            for action in actions[:replan_every]:
                terminated, truncated = step_env(action)
                if terminated or truncated or steps_taken >= max_steps:
                    break
            if terminated or truncated:
                break
            continue

        action_buffer: dict[int, np.ndarray] = {}
        buffer_code: int | None = None
        terminated = truncated = False
        while steps_taken < max_steps:
            active = (not ensemble_near_goal_only) or near_goal(np.asarray(obs_raw))
            if not active:
                break
            if steps_taken % max(1, int(ensemble_replan_every)) == 0 or steps_taken not in action_buffer:
                plan_inner, actions = make_plan()
                # Within-code ensembling: only average overlapping chunks that share
                # the same VQ macro-mode. On a code switch, drop stale predictions so
                # we never average actions from two different discrete routes.
                if ensemble_within_code and plan_inner.selected != buffer_code:
                    action_buffer.clear()
                    buffer_code = plan_inner.selected
                for offset, action in enumerate(actions):
                    step_idx = steps_taken + offset
                    if step_idx in action_buffer:
                        action_buffer[step_idx] = (
                            float(ensemble_decay) * action_buffer[step_idx]
                            + (1.0 - float(ensemble_decay)) * action
                        )
                    else:
                        action_buffer[step_idx] = action.copy()
            action = action_buffer.pop(steps_taken)
            terminated, truncated = step_env(action)
            if terminated or truncated:
                break
        if terminated or truncated:
            break

    env.close()
    if score_env is not None:
        score_env.close()
    result: dict[str, Any] = {
        "max_coverage": max_coverage,
        "success": float(success),
        "winding": winding,
        "steps": float(steps_taken),
        "selected_code_switches": float(np.sum(np.diff(selected_codes) != 0)) if len(selected_codes) > 1 else 0.0,
        "selected_unique_codes": float(len(set(selected_codes))),
        "recovery_uses": float(getattr(policy, "recovery_uses", 0)),
    }
    if capture_frames:
        result["frames"] = frames
    return result

