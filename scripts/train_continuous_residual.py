#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from vq_sigreg.models import VQSigRegOpenLoop
from vq_sigreg.pusht_data import PushTChunkDataset
from vq_sigreg.pusht_policy import _dims_from_checkpoint, _vq_decoder_kwargs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def near_goal_indices(dataset: PushTChunkDataset, near_goal_dist_px: float, near_goal_angle_rad: float) -> np.ndarray:
    raw = dataset.raw_states.numpy()
    dist = np.linalg.norm(raw[:, 2:4] - np.asarray([256.0, 256.0], dtype=np.float32)[None], axis=-1)
    angle = (raw[:, 4] - np.pi / 4.0 + np.pi) % (2.0 * np.pi) - np.pi
    idx = np.where((dist <= float(near_goal_dist_px)) & (np.abs(angle) <= float(near_goal_angle_rad)))[0]
    if idx.size == 0:
        raise RuntimeError("No near-goal samples matched the continuous residual filter.")
    return idx.astype(np.int64)


def load_continuous_model(
    checkpoint_path: str | Path,
    device: torch.device,
    residual_scale: float,
    gate_scale_px: float,
    gate_angle_rad: float,
    residual_steps: int,
) -> tuple[VQSigRegOpenLoop, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = dict(checkpoint["cfg"])
    cfg["model"] = dict(cfg.get("model", {}))
    cfg["model"]["continuous_action_residual"] = True
    cfg["model"]["continuous_action_residual_scale"] = float(residual_scale)
    cfg["model"]["continuous_action_gate_scale_px"] = float(gate_scale_px)
    cfg["model"]["continuous_action_gate_angle_rad"] = float(gate_angle_rad)
    cfg["model"]["continuous_action_steps"] = int(residual_steps)
    obs_dim, chunk_dim = _dims_from_checkpoint(checkpoint)
    model = VQSigRegOpenLoop(
        obs_dim,
        chunk_dim,
        int(cfg["model"]["hidden_dim"]),
        int(cfg["model"]["embedding_dim"]),
        int(cfg["model"]["codebook_size"]),
        **_vq_decoder_kwargs(cfg),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    return model, cfg


def train_continuous_head(
    model: VQSigRegOpenLoop,
    loader: DataLoader,
    device: torch.device,
    steps: int,
    lr: float,
    grad_clip_norm: float,
    train_chunk_steps: int | None,
) -> list[dict[str, float]]:
    for param in model.parameters():
        param.requires_grad_(False)
    if model.continuous_action_head is None:
        raise RuntimeError("continuous_action_head is not enabled.")
    for param in model.continuous_action_head.parameters():
        param.requires_grad_(True)
    model.train()
    opt = torch.optim.AdamW(model.continuous_action_head.parameters(), lr=float(lr), weight_decay=1e-5)
    it = iter(loader)
    logs: list[dict[str, float]] = []
    train_steps = max(1, min(int(train_chunk_steps or model.horizon), model.horizon))
    for step in range(1, int(steps) + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        obs = batch["obs"].to(device)
        target = batch["chunk"].to(device).reshape(obs.shape[0], model.horizon, model.action_dim)
        out = model.candidate_outputs(obs)
        selected = out["prior_logits"].argmax(dim=-1)
        pred = out["chunk_hat"][torch.arange(obs.shape[0], device=device), selected]
        pred = pred.reshape(obs.shape[0], model.horizon, model.action_dim)
        loss = F.mse_loss(pred[:, :train_steps], target[:, :train_steps])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.continuous_action_head.parameters(), float(grad_clip_norm))
        opt.step()
        if step == 1 or step % 100 == 0 or step == int(steps):
            with torch.no_grad():
                err_px = (pred[:, :train_steps] - target[:, :train_steps]).square().sum(dim=-1).sqrt().mean() * 256.0
            logs.append(
                {
                    "step": float(step),
                    "loss": float(loss.detach().cpu()),
                    "mean_action_error_px": float(err_px.detach().cpu()),
                }
            )
    model.eval()
    return logs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an isolated VQ-conditioned continuous chunk residual.")
    parser.add_argument("--checkpoint", default="outputs/vq_sigreg_pusht_k16_prior/vq_sigreg_latest.pt")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--out-dir", default="outputs/vq_sigreg_pusht_k16_continuous_residual")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--near-goal-dist-px", type=float, default=48.0)
    parser.add_argument("--near-goal-angle-rad", type=float, default=0.35)
    parser.add_argument("--residual-scale", type=float, default=0.06)
    parser.add_argument("--gate-scale-px", type=float, default=48.0)
    parser.add_argument("--gate-angle-rad", type=float, default=0.35)
    parser.add_argument("--residual-steps", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--train-chunk-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model, cfg = load_continuous_model(
        args.checkpoint,
        device,
        residual_scale=float(args.residual_scale),
        gate_scale_px=float(args.gate_scale_px),
        gate_angle_rad=float(args.gate_angle_rad),
        residual_steps=int(args.residual_steps),
    )
    data_cfg = cfg["data"]
    dataset = PushTChunkDataset(
        zarr_path=data_cfg["zarr_path"],
        horizon=int(data_cfg["horizon"]),
        obs_steps=int(data_cfg["obs_steps"]),
        split="train",
        val_fraction=float(data_cfg.get("val_fraction", 0.1)),
        seed=int(cfg.get("seed", args.seed)),
        max_samples=data_cfg.get("max_samples"),
    )
    idx = near_goal_indices(dataset, float(args.near_goal_dist_px), float(args.near_goal_angle_rad))
    subset = Subset(dataset, idx.tolist())
    loader = DataLoader(
        subset,
        batch_size=min(int(args.batch_size), len(subset)),
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    logs = train_continuous_head(
        model,
        loader,
        device,
        steps=int(args.train_steps),
        lr=float(args.lr),
        grad_clip_norm=float(args.grad_clip_norm),
        train_chunk_steps=args.train_chunk_steps,
    )
    cfg = dict(cfg)
    cfg["eval"] = dict(cfg.get("eval", {}))
    cfg["eval"]["selector"] = "prior"
    cfg["eval"]["sticky_logit_bonus"] = 0.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_checkpoint = out_dir / "vq_continuous_residual_latest.pt"
    torch.save(
        {
            "model_type": "vq_sigreg",
            "model": model.state_dict(),
            "cfg": cfg,
            "obs_dim": int(model.obs_dim),
            "chunk_dim": int(model.chunk_dim),
            "source_checkpoint": str(args.checkpoint),
            "continuous_action_residual": {
                "near_goal_dist_px": float(args.near_goal_dist_px),
                "near_goal_angle_rad": float(args.near_goal_angle_rad),
                "residual_scale": float(args.residual_scale),
                "gate_scale_px": float(args.gate_scale_px),
                "gate_angle_rad": float(args.gate_angle_rad),
                "residual_steps": int(args.residual_steps),
                "num_near_goal_samples": int(len(subset)),
            },
        },
        out_checkpoint,
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "out_checkpoint": str(out_checkpoint),
        "num_train_samples": int(len(dataset)),
        "num_near_goal_samples": int(len(subset)),
        "near_goal_fraction": float(len(subset) / max(1, len(dataset))),
        "near_goal_dist_px": float(args.near_goal_dist_px),
        "near_goal_angle_rad": float(args.near_goal_angle_rad),
        "residual_scale": float(args.residual_scale),
        "residual_steps": int(args.residual_steps),
        "logs": logs,
    }
    (out_dir / "continuous_residual_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
