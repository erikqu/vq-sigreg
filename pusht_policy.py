from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vq_sigreg.models import LeJEPAOpenLoop, VQSigRegOpenLoop
from vq_sigreg.pusht_data import denormalize_action, normalize_action, normalize_state
from vq_sigreg.pusht_env import set_env_state_exact


@dataclass
class ChunkPlan:
    chunk: torch.Tensor
    candidates: torch.Tensor
    selected: int


class LeJEPAPolicy:
    """Closed-loop wrapper for the deterministic official-SIGReg LeJEPA head."""

    model_type = "lejepa"

    def __init__(self, model: LeJEPAOpenLoop, cfg: dict[str, Any], device: torch.device):
        self.model = model.to(device).eval()
        self.cfg = cfg
        self.device = device

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def plan(self, obs: torch.Tensor, raw_state: np.ndarray) -> ChunkPlan:
        chunk = self.model.predict_chunk(obs.to(self.device))[0].clamp(-1.0, 1.0)
        return ChunkPlan(chunk=chunk, candidates=chunk[None], selected=0)


class VQSigRegPolicy:
    """Closed-loop wrapper for VQ-SIGReg candidates.

    The current VQ-SIGReg model is an open-loop candidate generator, not an EBM.
    At inference the default selector can use either a learned prior P(z | s_x)
    or a candidate-conditioned reranker over decoded VQ chunks.
    The older temporal-consistency selector remains available via config for ablations.
    """

    model_type = "vq_sigreg"

    def __init__(
        self,
        model: VQSigRegOpenLoop,
        cfg: dict[str, Any],
        device: torch.device,
        switch_penalty_px: float = 15.0,
    ):
        self.model = model.to(device).eval()
        self.cfg = cfg
        self.device = device
        self.switch_penalty_px = float(cfg.get("eval", {}).get("z_switch_penalty_px", switch_penalty_px))
        self.selector = str(cfg.get("eval", {}).get("selector", "prior"))
        self.sticky_logit_bonus = float(cfg.get("eval", {}).get("sticky_logit_bonus", 0.0))
        self.sticky_switch_margin = float(cfg.get("eval", {}).get("sticky_switch_margin", 0.5))
        self.sticky_macro_chunks = max(0, int(cfg.get("eval", {}).get("sticky_macro_chunks", 0)))
        self.reranker_switch_margin = float(cfg.get("eval", {}).get("reranker_switch_margin", 1.0))
        self.oracle_top_m = max(0, int(cfg.get("eval", {}).get("oracle_top_m", 16)))
        self.oracle_score_steps = max(1, int(cfg.get("eval", {}).get("oracle_score_steps", 16)))
        self.oracle_min_advantage = float(cfg.get("eval", {}).get("oracle_min_advantage", 1e-6))
        self.oracle_terminal_score = bool(cfg.get("eval", {}).get("oracle_terminal_score", False))
        self.mode_dwell_steps = max(0, int(cfg.get("eval", {}).get("mode_dwell_steps", 0)))
        self.mode_hysteresis_margin = float(cfg.get("eval", {}).get("mode_hysteresis_margin", 0.0))
        self.recovery_min_plans = max(0, int(cfg.get("eval", {}).get("recovery_min_plans", 2)))
        self.recovery_max_plans = max(1, int(cfg.get("eval", {}).get("recovery_max_plans", 5)))
        self.recovery_max_uses = max(0, int(cfg.get("eval", {}).get("recovery_max_uses", 3)))
        self.recovery_block_disp_px = float(cfg.get("eval", {}).get("recovery_block_disp_px", 6.0))
        self.recovery_recent_disp_px = float(cfg.get("eval", {}).get("recovery_recent_disp_px", 1.5))
        self.recovery_goal_progress_px = float(cfg.get("eval", {}).get("recovery_goal_progress_px", 8.0))
        self.recovery_standoff_px = float(cfg.get("eval", {}).get("recovery_standoff_px", 70.0))
        self.recovery_contact_px = float(cfg.get("eval", {}).get("recovery_contact_px", 35.0))
        self.recovery_push_px = float(cfg.get("eval", {}).get("recovery_push_px", 90.0))
        self.recovery_min_goal_dist_px = float(cfg.get("eval", {}).get("recovery_min_goal_dist_px", 0.0))
        self.macro_remaining = 0
        self.prev_action_px: np.ndarray | None = None
        self.prev_code: int | None = None
        self.initial_block_px: np.ndarray | None = None
        self.initial_goal_dist_px: float | None = None
        self.plan_calls = 0
        self.recovery_uses = 0

    def reset(self) -> None:
        self.prev_action_px = None
        self.prev_code = None
        self.macro_remaining = 0
        self.initial_block_px = None
        self.initial_goal_dist_px = None
        self.plan_calls = 0
        self.recovery_uses = 0

    def _obs_history_raw(self, obs: torch.Tensor) -> np.ndarray:
        hist = obs.detach().cpu().numpy()[0].reshape(-1, 6)
        positions = (hist[:, :4] + 1.0) * 0.5 * 512.0
        angles = np.arctan2(hist[:, 5:6], hist[:, 4:5])
        return np.concatenate([positions, angles], axis=-1)

    def _recovery_direction(self, raw_state: np.ndarray) -> np.ndarray:
        block = np.asarray(raw_state[2:4], dtype=np.float32)
        goal = np.asarray([256.0, 256.0], dtype=np.float32)
        direction = goal - block
        norm = float(np.linalg.norm(direction))
        if norm < 1e-4:
            angle = float(raw_state[4]) if raw_state.shape[0] > 4 else 0.0
            direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
            norm = float(np.linalg.norm(direction))
        return direction / max(norm, 1e-4)

    def _should_recover(self, obs: torch.Tensor, raw_state: np.ndarray) -> bool:
        if self.selector != "recovery_prior":
            return False
        if self.recovery_uses >= self.recovery_max_uses:
            return False
        if self.plan_calls < self.recovery_min_plans or self.plan_calls > self.recovery_max_plans:
            return False
        block = np.asarray(raw_state[2:4], dtype=np.float32)
        goal = np.asarray([256.0, 256.0], dtype=np.float32)
        if self.initial_block_px is None:
            self.initial_block_px = block.copy()
        if self.initial_goal_dist_px is None:
            self.initial_goal_dist_px = float(np.linalg.norm(goal - block))
        total_disp = float(np.linalg.norm(block - self.initial_block_px))
        hist = self._obs_history_raw(obs)
        recent_disp = float(np.linalg.norm(hist[-1, 2:4] - hist[0, 2:4]))
        current_goal_dist = float(np.linalg.norm(goal - block))
        # Spatial guard: only recover when the block is genuinely far from the
        # goal (where the prior tends to be stalling). Near the goal the prior is
        # usually mid-insertion, and a blind geometric push derails it.
        if current_goal_dist < self.recovery_min_goal_dist_px:
            return False
        goal_progress = self.initial_goal_dist_px - current_goal_dist
        no_motion = total_disp <= self.recovery_block_disp_px and recent_disp <= self.recovery_recent_disp_px
        no_goal_progress = goal_progress <= self.recovery_goal_progress_px
        return no_motion or no_goal_progress

    def _recovery_plan(self, raw_state: np.ndarray) -> ChunkPlan:
        agent = np.asarray(raw_state[:2], dtype=np.float32)
        block = np.asarray(raw_state[2:4], dtype=np.float32)
        direction = self._recovery_direction(raw_state)
        behind = block - direction * self.recovery_standoff_px
        contact = block - direction * self.recovery_contact_px
        push_through = block + direction * self.recovery_push_px
        waypoints = np.stack([agent, behind, contact, push_through], axis=0)
        fractions = np.linspace(0.0, 1.0, self.model.horizon, dtype=np.float32)
        actions = np.empty((self.model.horizon, self.model.action_dim), dtype=np.float32)
        segments = np.asarray([0.0, 0.35, 0.55, 1.0], dtype=np.float32)
        for idx, frac in enumerate(fractions):
            seg = int(np.searchsorted(segments, frac, side="right") - 1)
            seg = min(max(seg, 0), len(segments) - 2)
            local = (frac - segments[seg]) / max(float(segments[seg + 1] - segments[seg]), 1e-6)
            actions[idx] = (1.0 - local) * waypoints[seg] + local * waypoints[seg + 1]
        actions = np.clip(actions, 0.0, 512.0)
        chunk = torch.from_numpy(normalize_action(actions).reshape(-1)).to(self.device)
        self.recovery_uses += 1
        self.prev_code = None
        self.prev_action_px = actions[0].copy()
        return ChunkPlan(chunk=chunk.clamp(-1.0, 1.0), candidates=chunk[None].clamp(-1.0, 1.0), selected=-1)

    @torch.no_grad()
    def _prior_chunk_px(self, obs_vec: np.ndarray) -> np.ndarray:
        obs = torch.from_numpy(obs_vec.astype(np.float32))[None].to(self.device)
        out = self.model.candidate_outputs(obs)
        selected = int(out["prior_logits"][0].argmax().detach().cpu())
        chunk = out["chunk_hat"][0, selected].clamp(-1.0, 1.0)
        return denormalize_action(chunk.reshape(-1, 2)).detach().cpu().numpy()

    @torch.no_grad()
    def _code_chunk_px(self, obs_vec: np.ndarray, code_id: int) -> np.ndarray:
        obs = torch.from_numpy(obs_vec.astype(np.float32))[None].to(self.device)
        out = self.model.candidate_outputs(obs)
        code_id = int(np.clip(code_id, 0, out["chunk_hat"].shape[1] - 1))
        chunk = out["chunk_hat"][0, code_id].clamp(-1.0, 1.0)
        return denormalize_action(chunk.reshape(-1, 2)).detach().cpu().numpy()

    @torch.no_grad()
    def _score_oracle_candidate(
        self,
        score_env,
        raw_state: np.ndarray,
        obs_vec: np.ndarray,
        candidate_actions_px: np.ndarray,
        obs_steps: int,
        replan_every: int,
        score_steps: int,
    ) -> float:
        set_env_state_exact(score_env, np.asarray(raw_state, dtype=np.float64), reset_space=True)
        history = [row.copy() for row in obs_vec.reshape(obs_steps, -1)]
        max_reward = 0.0
        final_reward = 0.0
        steps_taken = 0
        terminated = False
        truncated = False

        def step_actions(actions_px: np.ndarray) -> None:
            nonlocal max_reward, final_reward, steps_taken, terminated, truncated
            for action in actions_px:
                obs_raw, reward, terminated, truncated, _ = score_env.step(
                    np.clip(action, 0.0, 512.0).astype(np.float32)
                )
                steps_taken += 1
                final_reward = float(reward)
                max_reward = max(max_reward, final_reward)
                history.append(normalize_state(np.asarray(obs_raw)))
                if terminated or truncated or steps_taken >= score_steps:
                    break

        step_actions(candidate_actions_px[: min(replan_every, score_steps)])
        while steps_taken < score_steps and not (terminated or truncated):
            obs_next = np.concatenate(history[-obs_steps:]).astype(np.float32)
            step_actions(self._prior_chunk_px(obs_next)[: min(replan_every, score_steps - steps_taken)])
        return float(max_reward + 0.02 * final_reward)

    @torch.no_grad()
    def _score_oracle_code(
        self,
        score_env,
        raw_state: np.ndarray,
        obs_vec: np.ndarray,
        code_id: int,
        obs_steps: int,
        score_steps: int,
        terminal_score: bool = False,
    ) -> float:
        set_env_state_exact(score_env, np.asarray(raw_state, dtype=np.float64), reset_space=True)
        history = [row.copy() for row in obs_vec.reshape(obs_steps, -1)]
        max_reward = 0.0
        final_reward = 0.0
        final_coverage = 0.0
        terminated = False
        truncated = False
        for _ in range(max(1, int(score_steps))):
            obs_next = np.concatenate(history[-obs_steps:]).astype(np.float32)
            actions_px = self._code_chunk_px(obs_next, int(code_id))
            obs_raw, reward, terminated, truncated, info = score_env.step(
                np.clip(actions_px[0], 0.0, 512.0).astype(np.float32)
            )
            final_reward = float(reward)
            final_coverage = float(
                info.get("coverage", final_reward * getattr(score_env.unwrapped, "success_threshold", 0.95))
            )
            max_reward = max(max_reward, final_reward)
            history.append(normalize_state(np.asarray(obs_raw)))
            if terminated or truncated:
                break
        if terminal_score:
            return float(final_coverage)
        return float(max_reward + 0.02 * final_reward)

    @torch.no_grad()
    def plan_oracle(
        self,
        obs: torch.Tensor,
        raw_state: np.ndarray,
        score_env,
        obs_steps: int,
        replan_every: int,
    ) -> ChunkPlan:
        obs = obs.to(self.device)
        out = self.model.candidate_outputs(obs)
        candidates = out["chunk_hat"][0].clamp(-1.0, 1.0)
        chunks_px = denormalize_action(candidates.reshape(candidates.shape[0], -1, 2)).detach().cpu().numpy()
        if self.mode_dwell_steps > 0 and self.prev_code is not None and self.macro_remaining > 0:
            selected = int(np.clip(self.prev_code, 0, candidates.shape[0] - 1))
            self.macro_remaining -= 1
            self.prev_action_px = chunks_px[selected, 0]
            return ChunkPlan(chunk=candidates[selected], candidates=candidates, selected=selected)
        prior_logits = out["prior_logits"][0]
        prior_selected = int(prior_logits.argmax().detach().cpu())
        if self.oracle_top_m <= 0 or self.oracle_top_m >= candidates.shape[0]:
            candidate_ids = np.arange(candidates.shape[0])
        else:
            candidate_ids = (
                torch.topk(prior_logits, k=min(self.oracle_top_m, candidates.shape[0])).indices.detach().cpu().numpy()
            )
        if prior_selected not in set(int(idx) for idx in candidate_ids):
            candidate_ids = np.concatenate([candidate_ids, np.asarray([prior_selected], dtype=candidate_ids.dtype)])
        obs_vec = obs.detach().cpu().numpy()[0]
        score_steps = max(1, int(self.oracle_score_steps))
        if self.mode_dwell_steps > 0:
            scores = np.asarray(
                [
                    self._score_oracle_code(
                        score_env,
                        raw_state,
                        obs_vec,
                        int(idx),
                        obs_steps,
                        score_steps,
                        terminal_score=self.oracle_terminal_score,
                    )
                    for idx in candidate_ids
                ],
                dtype=np.float32,
            )
        else:
            scores = np.asarray(
                [
                    self._score_oracle_candidate(
                        score_env,
                        raw_state,
                        obs_vec,
                        chunks_px[int(idx)],
                        obs_steps,
                        replan_every,
                        score_steps,
                    )
                    for idx in candidate_ids
                ],
                dtype=np.float32,
            )
        if not np.isfinite(scores).all():
            raise RuntimeError(f"oracle scoring produced non-finite scores: {scores.tolist()}")
        best_pos = int(scores.argmax())
        best_selected = int(candidate_ids[best_pos])
        prior_pos = int(np.where(candidate_ids == prior_selected)[0][0])
        prior_score = float(scores[prior_pos])
        best_score = float(scores[best_pos])
        if best_score + 1e-9 < prior_score:
            raise RuntimeError(
                "oracle prior-inclusion invariant failed: "
                f"best_score={best_score:.6f} < prior_score={prior_score:.6f}; "
                f"best_selected={best_selected}, prior_selected={prior_selected}, "
                f"candidate_ids={candidate_ids.tolist()}, scores={scores.tolist()}"
            )
        if best_selected != prior_selected and best_score <= prior_score + self.oracle_min_advantage:
            selected = prior_selected
        else:
            selected = best_selected
        self.prev_code = selected
        self.prev_action_px = chunks_px[selected, 0]
        if self.mode_dwell_steps > 0:
            self.macro_remaining = max(0, self.mode_dwell_steps - 1)
        return ChunkPlan(chunk=candidates[selected], candidates=candidates, selected=selected)

    @torch.no_grad()
    def plan(self, obs: torch.Tensor, raw_state: np.ndarray) -> ChunkPlan:
        obs = obs.to(self.device)
        if self.initial_block_px is None:
            block = np.asarray(raw_state[2:4], dtype=np.float32).copy()
            self.initial_block_px = block
            self.initial_goal_dist_px = float(np.linalg.norm(np.asarray([256.0, 256.0], dtype=np.float32) - block))
        if self._should_recover(obs, raw_state):
            self.plan_calls += 1
            return self._recovery_plan(raw_state)
        self.plan_calls += 1
        out = self.model.candidate_outputs(obs)
        candidates = out["chunk_hat"][0].clamp(-1.0, 1.0)
        if self.selector == "reranker_margin":
            prior_logits = out["prior_logits"][0]
            reranker_logits = out["reranker_logits"][0]
            prior_selected = int(prior_logits.argmax().detach().cpu())
            reranker_selected = int(reranker_logits.argmax().detach().cpu())
            gap = reranker_logits[reranker_selected] - reranker_logits[prior_selected]
            if reranker_selected != prior_selected and float(gap.detach().cpu()) > self.reranker_switch_margin:
                selected = reranker_selected
            else:
                selected = prior_selected
            self.prev_code = selected
            first_px = denormalize_action(candidates[selected].reshape(-1, 2))[0].detach().cpu().numpy()
            self.prev_action_px = first_px
            return ChunkPlan(chunk=candidates[selected], candidates=candidates, selected=selected)
        if self.selector in {"prior", "recovery_prior", "reranker", "cycle"}:
            if self.selector == "cycle":
                raw_logits = out["cycle_logits"][0]
            elif self.selector == "reranker":
                raw_logits = out["reranker_logits"][0]
            else:
                raw_logits = out["prior_logits"][0]
            logits = raw_logits.clone()
            if self.mode_dwell_steps > 0:
                if self.prev_code is not None and self.macro_remaining > 0:
                    selected = self.prev_code
                    self.macro_remaining -= 1
                else:
                    raw_selected = int(raw_logits.argmax().detach().cpu())
                    if self.prev_code is None:
                        selected = raw_selected
                    else:
                        switch_gap = raw_logits[raw_selected] - raw_logits[self.prev_code]
                        if raw_selected != self.prev_code and float(switch_gap.detach().cpu()) > self.mode_hysteresis_margin:
                            selected = raw_selected
                        else:
                            selected = self.prev_code
                    self.macro_remaining = max(0, self.mode_dwell_steps - 1)
            elif self.prev_code is not None and self.macro_remaining > 0:
                selected = self.prev_code
                self.macro_remaining -= 1
            else:
                if self.prev_code is not None and self.sticky_logit_bonus != 0.0:
                    raw_selected = int(raw_logits.argmax().detach().cpu())
                    switch_gap = raw_logits[raw_selected] - raw_logits[self.prev_code]
                    if float(switch_gap.detach().cpu()) > self.sticky_switch_margin:
                        selected = raw_selected
                    else:
                        logits[self.prev_code] = logits[self.prev_code] + self.sticky_logit_bonus
                        selected = int(logits.argmax().detach().cpu())
                else:
                    selected = int(logits.argmax().detach().cpu())
                if self.sticky_macro_chunks > 0 and selected != self.prev_code:
                    self.macro_remaining = self.sticky_macro_chunks - 1
            self.prev_code = selected
            first_px = denormalize_action(candidates[selected].reshape(-1, 2))[0].detach().cpu().numpy()
            self.prev_action_px = first_px
            return ChunkPlan(chunk=candidates[selected], candidates=candidates, selected=selected)
        chunks_px = denormalize_action(candidates.reshape(candidates.shape[0], -1, 2)).detach().cpu().numpy()
        reference = np.asarray(raw_state[:2], dtype=np.float32)
        if self.prev_action_px is not None:
            reference = self.prev_action_px.astype(np.float32)
        first_px = chunks_px[:, 0, :]
        score = np.linalg.norm(first_px - reference[None], axis=-1)
        if self.prev_code is not None:
            code_ids = np.arange(candidates.shape[0])
            score = score + self.switch_penalty_px * (code_ids != self.prev_code)
        selected = int(np.argmin(score))
        self.prev_code = selected
        self.prev_action_px = first_px[selected]
        return ChunkPlan(chunk=candidates[selected], candidates=candidates, selected=selected)


def _dims_from_checkpoint(checkpoint: dict[str, Any]) -> tuple[int, int]:
    cfg = checkpoint["cfg"]
    obs_dim = int(checkpoint.get("obs_dim", 6 * int(cfg["data"]["obs_steps"])))
    chunk_dim = int(checkpoint.get("chunk_dim", 2 * int(cfg["data"]["horizon"])))
    return obs_dim, chunk_dim


def _vq_decoder_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    model_cfg = cfg.get("model", {})
    return {
        "decoder_type": str(model_cfg.get("decoder_type", "mlp")),
        "action_dim": int(model_cfg.get("action_dim", 2)),
        "transformer_layers": int(model_cfg.get("transformer_layers", 2)),
        "transformer_heads": int(model_cfg.get("transformer_heads", 4)),
        "transformer_dim": model_cfg.get("transformer_dim", None),
        "transformer_ff_mult": int(model_cfg.get("transformer_ff_mult", 4)),
        "transformer_dropout": float(model_cfg.get("transformer_dropout", 0.0)),
        "transformer_causal": bool(model_cfg.get("transformer_causal", False)),
        "final_align_residual": bool(model_cfg.get("final_align_residual", False)),
        "final_align_residual_scale": float(model_cfg.get("final_align_residual_scale", 0.12)),
        "final_align_gate_scale_px": float(model_cfg.get("final_align_gate_scale_px", 96.0)),
        "delta_action_residual": bool(model_cfg.get("delta_action_residual", False)),
        "delta_action_residual_scale": float(model_cfg.get("delta_action_residual_scale", 0.12)),
        "delta_action_gate_scale_px": float(model_cfg.get("delta_action_gate_scale_px", 96.0)),
        "delta_action_gate_angle_rad": float(model_cfg.get("delta_action_gate_angle_rad", 0.75)),
        "continuous_action_residual": bool(model_cfg.get("continuous_action_residual", False)),
        "continuous_action_residual_scale": float(model_cfg.get("continuous_action_residual_scale", 0.06)),
        "continuous_action_gate_scale_px": float(model_cfg.get("continuous_action_gate_scale_px", 48.0)),
        "continuous_action_gate_angle_rad": float(model_cfg.get("continuous_action_gate_angle_rad", 0.35)),
        "continuous_action_steps": int(model_cfg.get("continuous_action_steps", 1)),
        "local_continuous_action_residual": bool(model_cfg.get("local_continuous_action_residual", False)),
        "local_continuous_action_residual_scale": float(model_cfg.get("local_continuous_action_residual_scale", 0.06)),
        "local_continuous_action_gate_scale_px": float(model_cfg.get("local_continuous_action_gate_scale_px", 48.0)),
        "local_continuous_action_gate_angle_rad": float(model_cfg.get("local_continuous_action_gate_angle_rad", 0.35)),
        "local_continuous_action_steps": int(model_cfg.get("local_continuous_action_steps", 1)),
        "local_continuous_action_last_steps": int(model_cfg.get("local_continuous_action_last_steps", 0)),
        "local_spline_action_residual": bool(model_cfg.get("local_spline_action_residual", False)),
        "local_spline_action_residual_scale": float(model_cfg.get("local_spline_action_residual_scale", 0.06)),
        "local_spline_action_gate_scale_px": float(model_cfg.get("local_spline_action_gate_scale_px", 48.0)),
        "local_spline_action_gate_angle_rad": float(model_cfg.get("local_spline_action_gate_angle_rad", 0.35)),
        "contact_action_residual": bool(model_cfg.get("contact_action_residual", False)),
        "contact_action_residual_scale": float(model_cfg.get("contact_action_residual_scale", 0.06)),
        "contact_action_gate_scale_px": float(model_cfg.get("contact_action_gate_scale_px", 48.0)),
        "contact_action_gate_angle_rad": float(model_cfg.get("contact_action_gate_angle_rad", 0.35)),
        "contact_action_steps": int(model_cfg.get("contact_action_steps", 1)),
        "multi_contact_action_residual": bool(model_cfg.get("multi_contact_action_residual", False)),
        "multi_contact_action_residual_scale": float(model_cfg.get("multi_contact_action_residual_scale", 0.06)),
        "multi_contact_action_gate_scale_px": float(model_cfg.get("multi_contact_action_gate_scale_px", 48.0)),
        "multi_contact_action_gate_angle_rad": float(model_cfg.get("multi_contact_action_gate_angle_rad", 0.35)),
        "multi_contact_action_samples": int(model_cfg.get("multi_contact_action_samples", 4)),
        "flow_action_residual": bool(model_cfg.get("flow_action_residual", False)),
        "flow_action_residual_scale": float(model_cfg.get("flow_action_residual_scale", 0.06)),
        "flow_action_gate_scale_px": float(model_cfg.get("flow_action_gate_scale_px", 48.0)),
        "flow_action_gate_angle_rad": float(model_cfg.get("flow_action_gate_angle_rad", 0.35)),
        "flow_action_steps": int(model_cfg.get("flow_action_steps", 1)),
        "hierarchical_flow_decoder": bool(model_cfg.get("hierarchical_flow_decoder", False)),
        "hierarchical_flow_steps": int(model_cfg.get("hierarchical_flow_steps", 2)),
        "hierarchical_flow_noise_scale": float(model_cfg.get("hierarchical_flow_noise_scale", 0.0)),
        "hierarchical_flow_samples": int(model_cfg.get("hierarchical_flow_samples", 1)),
        "hierarchical_flow_relative": bool(model_cfg.get("hierarchical_flow_relative", False)),
        "hierarchical_flow_no_z": bool(model_cfg.get("hierarchical_flow_no_z", False)),
        "fine_vq_residual": bool(model_cfg.get("fine_vq_residual", False)),
        "fine_vq_codebook_size": int(model_cfg.get("fine_vq_codebook_size", 8)),
        "fine_vq_residual_scale": float(model_cfg.get("fine_vq_residual_scale", 0.06)),
        "fine_vq_gate_scale_px": float(model_cfg.get("fine_vq_gate_scale_px", 48.0)),
        "fine_vq_gate_angle_rad": float(model_cfg.get("fine_vq_gate_angle_rad", 0.35)),
    }


def load_policy(checkpoint_path: str | Path, device: torch.device | str):
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["cfg"]
    obs_dim, chunk_dim = _dims_from_checkpoint(checkpoint)
    model_type = str(checkpoint["model_type"])
    hidden_dim = int(cfg["model"]["hidden_dim"])
    embedding_dim = int(cfg["model"]["embedding_dim"])
    if model_type == "lejepa":
        model = LeJEPAOpenLoop(obs_dim, chunk_dim, hidden_dim, embedding_dim)
        model.load_state_dict(checkpoint["model"])
        return LeJEPAPolicy(model, cfg, device), cfg
    if model_type == "vq_sigreg":
        model = VQSigRegOpenLoop(
            obs_dim,
            chunk_dim,
            hidden_dim,
            embedding_dim,
            int(cfg["model"]["codebook_size"]),
            **_vq_decoder_kwargs(cfg),
        )
        state = checkpoint["model"]
        has_prior = any(str(key).startswith("prior_head.") for key in state)
        has_reranker = any(str(key).startswith("reranker_head.") for key in state)
        if has_prior and has_reranker:
            model.load_state_dict(state)
        else:
            model.load_state_dict(state, strict=False)
            cfg = deepcopy(cfg)
            if has_prior:
                cfg.setdefault("eval", {})["selector"] = "prior"
            else:
                cfg.setdefault("eval", {})["selector"] = "temporal"
        return VQSigRegPolicy(model, cfg, device), cfg
    raise ValueError(f"unknown model_type={model_type!r}")

