#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from matplotlib.patches import Polygon
from torch.utils.data import DataLoader

from vq_sigreg.models import LeJEPAOpenLoop, VQSigRegOpenLoop
from vq_sigreg.official_lejepa import OfficialLeJEPASIGReg
from vq_sigreg.pusht_data import denormalize_action, make_openloop_datasets


def vq_decoder_kwargs(cfg: dict) -> dict:
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
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def final_alignment_weights(batch: dict[str, torch.Tensor], cfg: dict) -> torch.Tensor:
    weight = float(cfg["train"].get("final_align_weight", 0.0))
    raw_state = batch.get("raw_state")
    if weight <= 0.0 or raw_state is None:
        return torch.ones(batch["chunk"].shape[0], device=batch["chunk"].device)
    goal_xy = torch.tensor([256.0, 256.0], device=raw_state.device, dtype=raw_state.dtype)
    block_xy = raw_state[:, 2:4]
    dist = torch.linalg.norm(block_xy - goal_xy[None], dim=-1)
    scale = float(cfg["train"].get("final_align_scale_px", 96.0))
    proximity = torch.exp(-((dist / scale) ** 2))
    return 1.0 + weight * proximity


def train_lejepa(
    model: LeJEPAOpenLoop,
    loader: DataLoader,
    cfg: dict,
    device: torch.device,
    sigreg_loss: torch.nn.Module,
) -> list[dict[str, float]]:
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    it = iter(loader)
    logs: list[dict[str, float]] = []
    for step in range(1, int(cfg["train"]["steps"]) + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = to_device(batch, device)
        out = model(batch["obs"], batch["chunk"])
        pred = F.mse_loss(out["shat"], out["sy"].detach())
        readout = F.mse_loss(model.decoder(out["sy"]), batch["chunk"])
        pred_readout = F.mse_loss(out["chunk_hat"], batch["chunk"])
        sigreg = sigreg_loss(torch.cat([out["sx"], out["sy"]], dim=0))
        loss = (
            pred
            + float(cfg["train"]["lambda_readout"]) * readout
            + float(cfg["train"]["lambda_pred_readout"]) * pred_readout
            + float(cfg["train"]["lambda_sigreg"]) * sigreg
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip_norm"]))
        opt.step()
        if step == 1 or step % int(cfg["train"]["log_every"]) == 0 or step == int(cfg["train"]["steps"]):
            logs.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "pred": float(pred.detach().cpu()),
                    "readout": float(readout.detach().cpu()),
                    "pred_readout": float(pred_readout.detach().cpu()),
                    "sigreg": float(sigreg.detach().cpu()),
                }
            )
    return logs


def train_vq_sigreg(
    model: VQSigRegOpenLoop,
    loader: DataLoader,
    cfg: dict,
    device: torch.device,
    sigreg_loss: torch.nn.Module,
) -> list[dict[str, float]]:
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    it = iter(loader)
    logs: list[dict[str, float]] = []
    temp = float(cfg["train"]["select_temp"])
    codebook_size = int(cfg["model"]["codebook_size"])
    for step in range(1, int(cfg["train"]["steps"]) + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = to_device(batch, device)
        out = model(batch["obs"], batch["chunk"], select_temp=temp)
        latent_dist = (out["shat"] - out["sy"].detach()[:, None, :]).square().sum(dim=-1)
        pred = (-torch.logsumexp(-temp * latent_dist, dim=-1) / temp).mean()
        align_weight = final_alignment_weights(batch, cfg)
        readout_per_example = (model.decoder(out["sy"]) - batch["chunk"]).square().mean(dim=-1)
        readout = (readout_per_example * align_weight).sum() / align_weight.sum()
        decoded_dist = (out["chunk_hat"] - batch["chunk"][:, None, :]).square().mean(dim=-1)
        decoded_winner = decoded_dist.argmin(dim=-1)
        pred_readout_per_example = decoded_dist.gather(1, decoded_winner[:, None]).squeeze(1)
        pred_readout = (pred_readout_per_example * align_weight).sum() / align_weight.sum()
        usage = torch.softmax(-temp * decoded_dist, dim=-1).mean(dim=0)
        usage_kl = (usage * (usage * codebook_size + 1e-12).log()).sum()
        prior_per_example = F.cross_entropy(out["prior_logits"], decoded_winner.detach(), reduction="none")
        prior = (prior_per_example * align_weight).sum() / align_weight.sum()
        prior_acc = (out["prior_logits"].argmax(dim=-1) == decoded_winner).float().mean()
        reranker = F.cross_entropy(out["reranker_logits"], decoded_winner.detach())
        reranker_acc = (out["reranker_logits"].argmax(dim=-1) == decoded_winner).float().mean()
        sigreg = sigreg_loss(torch.cat([out["sx"], out["sy"]], dim=0))
        loss = (
            float(cfg["train"].get("lambda_latent_pred", 0.0)) * pred
            + float(cfg["train"]["lambda_readout"]) * readout
            + float(cfg["train"]["lambda_pred_readout"]) * pred_readout
            + float(cfg["train"]["lambda_sigreg"]) * sigreg
            + float(cfg["train"]["lambda_code_entropy"]) * usage_kl
            + float(cfg["train"].get("lambda_prior", 1.0)) * prior
            + float(cfg["train"].get("lambda_reranker", 1.0)) * reranker
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip_norm"]))
        opt.step()
        if step == 1 or step % int(cfg["train"]["log_every"]) == 0 or step == int(cfg["train"]["steps"]):
            logs.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "pred": float(pred.detach().cpu()),
                    "readout": float(readout.detach().cpu()),
                    "pred_readout": float(pred_readout.detach().cpu()),
                    "sigreg": float(sigreg.detach().cpu()),
                    "usage_kl": float(usage_kl.detach().cpu()),
                    "code_perplexity": float(torch.exp(-(usage * (usage + 1e-12).log()).sum()).detach().cpu()),
                    "prior": float(prior.detach().cpu()),
                    "prior_acc": float(prior_acc.detach().cpu()),
                    "reranker": float(reranker.detach().cpu()),
                    "reranker_acc": float(reranker_acc.detach().cpu()),
                }
            )
    return logs


def chunks_to_pixels(chunks: torch.Tensor | np.ndarray, horizon: int) -> np.ndarray:
    if isinstance(chunks, torch.Tensor):
        chunks = chunks.detach().cpu()
        chunks = denormalize_action(chunks).numpy()
    else:
        chunks = denormalize_action(chunks)
    return np.asarray(chunks, dtype=np.float32).reshape(*chunks.shape[:-1], horizon, 2)


def _block_centers(raw_state: np.ndarray) -> np.ndarray:
    return np.asarray(raw_state, dtype=np.float32)[..., 2:4]


def _path_block_distance(paths: np.ndarray, centers: np.ndarray) -> np.ndarray:
    if paths.ndim == 3:
        return np.linalg.norm(paths - centers[:, None, :], axis=-1)
    if paths.ndim == 4:
        return np.linalg.norm(paths - centers[:, None, None, :], axis=-1)
    raise ValueError(f"expected paths with rank 3 or 4, got shape {paths.shape}")


def collision_rate(paths: np.ndarray, centers: np.ndarray, radius: float) -> float:
    dist = _path_block_distance(paths, centers)
    return float((dist < radius).mean())


def side_score(paths: np.ndarray, centers: np.ndarray) -> np.ndarray:
    if paths.ndim == 3:
        return (paths[..., 1] - centers[:, None, 1]).mean(axis=-1)
    if paths.ndim == 4:
        return (paths[..., 1] - centers[:, None, None, 1]).mean(axis=-1)
    raise ValueError(f"expected paths with rank 3 or 4, got shape {paths.shape}")


def nearest_demo_distance(paths: np.ndarray, demo_paths: np.ndarray, query_block: int = 32, demo_block: int = 512) -> np.ndarray:
    """Nearest demo-future distance without materializing the full N x K x N tensor."""
    paths = np.asarray(paths, dtype=np.float32)
    demo_paths = np.asarray(demo_paths, dtype=np.float32)
    if paths.ndim == 3:
        paths = paths[:, None, :, :]
    if paths.ndim != 4:
        raise ValueError(f"expected paths rank 3 or 4, got {paths.shape}")
    n = paths.shape[0]
    best = np.full((n,), np.inf, dtype=np.float32)
    for lo in range(0, n, query_block):
        hi = min(n, lo + query_block)
        q = paths[lo:hi]
        q_best = np.full((hi - lo,), np.inf, dtype=np.float32)
        for dlo in range(0, demo_paths.shape[0], demo_block):
            dhi = min(demo_paths.shape[0], dlo + demo_block)
            d = demo_paths[dlo:dhi]
            dist = np.linalg.norm(q[:, :, None] - d[None, None], axis=-1).mean(axis=-1)
            q_best = np.minimum(q_best, dist.min(axis=-1).min(axis=-1))
        best[lo:hi] = q_best
    return best


def mine_ambiguous_contexts(payload: dict[str, np.ndarray], cfg: dict) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    obs = np.asarray(payload["obs"], dtype=np.float32)
    centers = _block_centers(payload["raw_state"])
    demo_paths = np.asarray(payload["demo_paths"], dtype=np.float32)
    n = obs.shape[0]
    k = min(int(cfg["eval"].get("nn_k", 64)), max(1, n - 1))
    norms = (obs * obs).sum(axis=1, keepdims=True)
    d2 = norms + norms.T - 2.0 * (obs @ obs.T)
    np.fill_diagonal(d2, np.inf)
    nn = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]

    demo_side = side_score(demo_paths, centers)
    agent = payload["raw_state"][:, :2]
    agent_block_dist = np.linalg.norm(agent - centers, axis=-1)
    min_dist = float(cfg["eval"].get("approach_min_agent_block_dist_px", 55.0))
    min_spread = float(cfg["eval"].get("min_route_spread_px", 45.0))
    sides = demo_side[nn]
    q10 = np.quantile(sides, 0.1, axis=1)
    q90 = np.quantile(sides, 0.9, axis=1)
    spread = q90 - q10
    has_both_sides = (sides.min(axis=1) < -8.0) & (sides.max(axis=1) > 8.0)
    approach = agent_block_dist >= min_dist
    candidates = np.where(approach & has_both_sides & (spread >= min_spread))[0]
    if len(candidates) == 0:
        candidates = np.where(approach & (spread >= min_spread))[0]
    if len(candidates) == 0:
        candidates = np.where(approach)[0]
    score = spread + 0.25 * agent_block_dist + 80.0 * has_both_sides.astype(np.float32)
    order = candidates[np.argsort(-score[candidates])]
    metrics = {
        "mined_contexts": float(len(order)),
        "mined_mean_demo_route_spread_px": float(spread[order].mean()) if len(order) else 0.0,
        "mined_mean_agent_block_dist_px": float(agent_block_dist[order].mean()) if len(order) else 0.0,
        "mined_both_sides_rate": float(has_both_sides[order].mean()) if len(order) else 0.0,
    }
    return order, nn, metrics


def mined_demo_metrics(payload: dict[str, np.ndarray], mined: np.ndarray, nn: np.ndarray, cfg: dict) -> dict[str, dict[str, float]]:
    if len(mined) == 0:
        return {"lejepa": {}, "vq_sigreg": {}}
    centers = _block_centers(payload["raw_state"])
    demo = payload["demo_paths"]
    le = payload["lejepa_paths"]
    vq = payload["vq_paths"]
    k = min(int(cfg["eval"].get("nn_k", 64)), nn.shape[1])
    side_thresh = float(cfg["eval"].get("route_side_threshold_px", 8.0))

    le_dists, vq_dists, route_counts, demo_spreads, vq_spreads = [], [], [], [], []
    for i in mined:
        neigh = nn[i, :k]
        neigh_demo = demo[neigh]
        le_d = np.linalg.norm(le[i][None] - neigh_demo, axis=-1).mean(axis=-1).min()
        vq_d = np.linalg.norm(vq[i, :, None] - neigh_demo[None], axis=-1).mean(axis=-1).min(axis=-1).min()
        le_dists.append(float(le_d))
        vq_dists.append(float(vq_d))

        neigh_sides = side_score(neigh_demo, _block_centers(payload["raw_state"][neigh]))
        vq_sides = side_score(vq[i : i + 1], centers[i : i + 1])[0]
        demo_routes = set()
        if np.any(neigh_sides < -side_thresh):
            demo_routes.add("below")
        if np.any(neigh_sides > side_thresh):
            demo_routes.add("above")
        vq_routes = set()
        if np.any(vq_sides < -side_thresh):
            vq_routes.add("below")
        if np.any(vq_sides > side_thresh):
            vq_routes.add("above")
        route_counts.append(float(len(demo_routes & vq_routes)))
        demo_spreads.append(float(np.quantile(neigh_sides, 0.9) - np.quantile(neigh_sides, 0.1)))
        vq_spreads.append(float(vq_sides.max() - vq_sides.min()))

    return {
        "lejepa": {
            "mined_nearest_demo_dist_px": float(np.mean(le_dists)),
        },
        "vq_sigreg": {
            "mined_nearest_demo_dist_px": float(np.mean(vq_dists)),
            "mined_demo_routes_covered": float(np.mean(route_counts)),
            "mined_demo_route_spread_px": float(np.mean(demo_spreads)),
            "mined_vq_route_spread_px": float(np.mean(vq_spreads)),
        },
    }


@torch.no_grad()
def evaluate(
    lejepa: LeJEPAOpenLoop,
    vq: VQSigRegOpenLoop,
    loader: DataLoader,
    cfg: dict,
    device: torch.device,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    batch = to_device(next(iter(loader)), device)
    horizon = int(cfg["data"]["horizon"])
    le_chunk = lejepa.predict_chunk(batch["obs"])
    vq_chunks = vq.predict_all_chunks(batch["obs"])
    demo_paths = chunks_to_pixels(batch["chunk"], horizon)
    le_paths = chunks_to_pixels(le_chunk, horizon)
    vq_paths = chunks_to_pixels(vq_chunks, horizon)
    radius = float(cfg["eval"]["collision_radius_px"])
    centers = _block_centers(batch["raw_state"].cpu().numpy())
    le_demo_dist = nearest_demo_distance(le_paths, demo_paths)
    vq_demo_dist = nearest_demo_distance(vq_paths, demo_paths)
    le_sides = side_score(le_paths, centers)
    vq_sides = side_score(vq_paths, centers)
    metrics = {
        "lejepa": {
            "collision_rate": collision_rate(le_paths, centers, radius),
            "mean_nearest_demo_dist_px": float(le_demo_dist.mean()),
            "mean_abs_side_score_px": float(np.abs(le_sides).mean()),
            "num_predictions_per_context": 1.0,
        },
        "vq_sigreg": {
            "collision_rate": collision_rate(vq_paths, centers, radius),
            "mean_nearest_demo_dist_px": float(vq_demo_dist.mean()),
            "mean_abs_side_score_px": float(np.abs(vq_sides).mean()),
            "fork_spread_px": float((vq_sides.max(axis=1) - vq_sides.min(axis=1)).mean()),
            "num_predictions_per_context": float(vq_paths.shape[1]),
        },
    }
    payload = {
        "obs": batch["obs"].cpu().numpy(),
        "raw_state": batch["raw_state"].cpu().numpy(),
        "demo_paths": demo_paths,
        "lejepa_paths": le_paths,
        "vq_paths": vq_paths,
        "route": batch.get("route", torch.zeros(batch["obs"].shape[0], device=device)).cpu().numpy(),
    }
    if str(cfg["eval"].get("selection", "ambiguous_nn")) == "ambiguous_nn":
        mined, nn, mined_metrics = mine_ambiguous_contexts(payload, cfg)
        payload["mined_indices"] = mined
        payload["neighbor_indices"] = nn
        extra = mined_demo_metrics(payload, mined, nn, cfg)
        metrics["selection"] = mined_metrics
        metrics["lejepa"].update(extra["lejepa"])
        metrics["vq_sigreg"].update(extra["vq_sigreg"])
    return metrics, payload


def tee_polygons(raw_state: np.ndarray, scale: float = 30.0) -> list[np.ndarray]:
    """Push-T tee polygons matching `gym_pusht.envs.PushTEnv.add_tee`."""
    raw_state = np.asarray(raw_state, dtype=np.float32)
    center = raw_state[2:4]
    angle = float(raw_state[4])
    length = 4.0
    rect_bar = np.array(
        [
            (-length * scale / 2.0, scale),
            (length * scale / 2.0, scale),
            (length * scale / 2.0, 0.0),
            (-length * scale / 2.0, 0.0),
        ],
        dtype=np.float32,
    )
    rect_stem = np.array(
        [
            (-scale / 2.0, scale),
            (-scale / 2.0, length * scale),
            (scale / 2.0, length * scale),
            (scale / 2.0, scale),
        ],
        dtype=np.float32,
    )
    rot = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    return [rect_bar @ rot.T + center, rect_stem @ rot.T + center]


def draw_block(ax, raw_state: np.ndarray) -> None:
    for idx, poly in enumerate(tee_polygons(raw_state)):
        ax.add_patch(
            Polygon(
                poly,
                closed=True,
                facecolor="0.40",
                edgecolor="0.18",
                linewidth=1.5,
                alpha=0.82,
                label="T-block" if idx == 0 else None,
            )
        )


def select_plot_indices(payload: dict[str, np.ndarray], cfg: dict) -> np.ndarray:
    n = min(int(cfg["eval"]["num_plot_contexts"]), payload["demo_paths"].shape[0])
    if str(cfg["eval"].get("selection", "ambiguous_nn")) == "ambiguous_nn" and "mined_indices" in payload:
        return np.asarray(payload["mined_indices"][:n], dtype=np.int64)
    centers = _block_centers(payload["raw_state"])
    radius = float(cfg["eval"]["collision_radius_px"])
    le_dist = _path_block_distance(payload["lejepa_paths"], centers).min(axis=-1)
    vq_dist = _path_block_distance(payload["vq_paths"], centers).min(axis=-1)
    vq_sides = side_score(payload["vq_paths"], centers)
    vq_spread = vq_sides.max(axis=1) - vq_sides.min(axis=1)
    agent = payload["raw_state"][:, :2]
    agent_left = (centers[:, 0] - agent[:, 0]).clip(min=0.0)
    # Prefer the visual that answers the question: deterministic average near
    # the block, code-conditioned predictions separated around the block.
    score = 4.0 * (le_dist < radius).astype(np.float32)
    score += 2.0 * (vq_dist.min(axis=1) >= radius).astype(np.float32)
    score += vq_spread / 80.0
    score += agent_left / 240.0
    return np.argsort(-score)[:n]


def plot_trajectories(payload: dict[str, np.ndarray], cfg: dict, out_path: Path) -> None:
    indices = select_plot_indices(payload, cfg)
    n = len(indices)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4.1), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, payload["vq_paths"].shape[1]))
    all_x, all_y = [], []
    for panel, ax in enumerate(axes):
        i = int(indices[panel])
        raw = payload["raw_state"][i]
        center = _block_centers(raw)
        draw_block(ax, raw)
        demo = payload["demo_paths"][i]
        le = payload["lejepa_paths"][i]
        vq = payload["vq_paths"][i]
        if "neighbor_indices" in payload:
            neighbors = payload["neighbor_indices"][i, : min(12, payload["neighbor_indices"].shape[1])]
            for j, neighbor in enumerate(neighbors):
                neigh_demo = payload["demo_paths"][int(neighbor)]
                ax.plot(
                    neigh_demo[:, 0],
                    neigh_demo[:, 1],
                    color="0.80",
                    lw=1.4,
                    alpha=0.45,
                    label="NN demo futures" if j == 0 else None,
                )
        ax.plot(demo[:, 0], demo[:, 1], color="0.45", lw=2.5, label="held-out demo target")
        ax.plot(le[:, 0], le[:, 1], color="tab:red", lw=3.0, label="LeJEPA")
        ax.scatter(le[:, 0], le[:, 1], color="tab:red", s=10)
        for k in range(vq.shape[0]):
            ax.plot(vq[k, :, 0], vq[k, :, 1], color=colors[k], lw=2.2, label=f"VQ z={k}")
        ax.scatter([raw[0]], [raw[1]], marker="o", s=70, color="black", label="start")
        ax.scatter([center[0]], [center[1]], marker="x", s=70, color="black", label="block center")
        ax.set_title(f"context {i}")
        tee = np.concatenate(tee_polygons(raw), axis=0)
        xs = np.concatenate([demo[:, 0], le[:, 0], vq.reshape(-1, 2)[:, 0], tee[:, 0], [raw[0], center[0]]])
        ys = np.concatenate([demo[:, 1], le[:, 1], vq.reshape(-1, 2)[:, 1], tee[:, 1], [raw[1], center[1]]])
        all_x.append(xs)
        all_y.append(ys)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
    xmin, xmax = float(np.concatenate(all_x).min() - 55), float(np.concatenate(all_x).max() + 55)
    ymin, ymax = float(np.concatenate(all_y).min() - 55), float(np.concatenate(all_y).max() + 55)
    for ax in axes:
        ax.set_xlim(max(0.0, xmin), min(512.0, xmax))
        ax.set_ylim(max(0.0, ymin), min(512.0, ymax))
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Push-T Open-Loop: deterministic LeJEPA average vs VQ-SIGReg forks")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_training(logs: dict[str, list[dict[str, float]]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for name, rows in logs.items():
        steps = [r["step"] for r in rows]
        axes[0].plot(steps, [r["loss"] for r in rows], label=name)
        axes[1].plot(steps, [r["pred_readout"] for r in rows], label=name)
        if name == "vq_sigreg":
            axes[2].plot(steps, [r["code_perplexity"] for r in rows], label=name)
    axes[0].set_title("loss")
    axes[1].set_title("decoded prediction loss")
    axes[2].set_title("VQ code perplexity")
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="vq_sigreg/configs/pusht_openloop.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--codebook-size", type=int, default=None)
    parser.add_argument("--final-align-weight", type=float, default=None)
    parser.add_argument("--final-align-residual", action="store_true")
    parser.add_argument("--final-align-residual-scale", type=float, default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.steps is not None:
        cfg["train"]["steps"] = int(args.steps)
    if args.horizon is not None:
        cfg["data"]["horizon"] = int(args.horizon)
        cfg.setdefault("eval", {})["execution_horizon"] = int(args.horizon)
    if args.codebook_size is not None:
        cfg["model"]["codebook_size"] = int(args.codebook_size)
    if args.final_align_weight is not None:
        cfg["train"]["final_align_weight"] = float(args.final_align_weight)
    if args.final_align_residual:
        cfg["model"]["final_align_residual"] = True
    if args.final_align_residual_scale is not None:
        cfg["model"]["final_align_residual_scale"] = float(args.final_align_residual_scale)
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir
    device = torch.device(args.device or cfg.get("device", "cpu"))
    set_seed(int(cfg["seed"]))
    train_ds, val_ds, source = make_openloop_datasets(cfg, int(cfg["seed"]))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=min(4096, len(val_ds)), shuffle=False, num_workers=0)
    sample = train_ds[0]
    obs_dim = int(sample["obs"].numel())
    chunk_dim = int(sample["chunk"].numel())

    lejepa = LeJEPAOpenLoop(obs_dim, chunk_dim, int(cfg["model"]["hidden_dim"]), int(cfg["model"]["embedding_dim"])).to(device)
    vq = VQSigRegOpenLoop(
        obs_dim,
        chunk_dim,
        int(cfg["model"]["hidden_dim"]),
        int(cfg["model"]["embedding_dim"]),
        int(cfg["model"]["codebook_size"]),
        **vq_decoder_kwargs(cfg),
    ).to(device)
    sigreg_cfg = cfg.get("sigreg", {})
    lejepa_sigreg = OfficialLeJEPASIGReg(
        num_slices=int(sigreg_cfg.get("num_slices", 64)),
        n_points=int(sigreg_cfg.get("n_points", 17)),
        t_max=float(sigreg_cfg.get("t_max", 3.0)),
    ).to(device)
    vq_sigreg = OfficialLeJEPASIGReg(
        num_slices=int(sigreg_cfg.get("num_slices", 64)),
        n_points=int(sigreg_cfg.get("n_points", 17)),
        t_max=float(sigreg_cfg.get("t_max", 3.0)),
    ).to(device)
    lejepa_logs = train_lejepa(lejepa, train_loader, cfg, device, lejepa_sigreg)
    vq_logs = train_vq_sigreg(vq, train_loader, cfg, device, vq_sigreg)
    metrics, payload = evaluate(lejepa, vq, val_loader, cfg, device)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_trajectories(payload, cfg, out_dir / "trajectories.png")
    plot_training({"lejepa": lejepa_logs, "vq_sigreg": vq_logs}, out_dir / "training.png")
    lejepa_ckpt = out_dir / "lejepa_latest.pt"
    vq_ckpt = out_dir / "vq_sigreg_latest.pt"
    torch.save(
        {
            "model_type": "lejepa",
            "model": lejepa.state_dict(),
            "cfg": cfg,
            "obs_dim": obs_dim,
            "chunk_dim": chunk_dim,
        },
        lejepa_ckpt,
    )
    torch.save(
        {
            "model_type": "vq_sigreg",
            "model": vq.state_dict(),
            "cfg": cfg,
            "obs_dim": obs_dim,
            "chunk_dim": chunk_dim,
        },
        vq_ckpt,
    )
    summary = {
        "source": source,
        "lejepa_impl": "galilai-group/lejepa SIGReg (EppsPulley + SlicingUnivariateTest)",
        "config": cfg,
        "metrics": metrics,
        "logs": {"lejepa": lejepa_logs, "vq_sigreg": vq_logs},
        "artifacts": {
            "trajectories": str(out_dir / "trajectories.png"),
            "training": str(out_dir / "training.png"),
            "lejepa_checkpoint": str(lejepa_ckpt),
            "vq_sigreg_checkpoint": str(vq_ckpt),
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
