#!/usr/bin/env python
"""Train a conditional DDPM (Diffusion Policy) on the Push-T chunk dataset.

Uses the EXACT same data pipeline as the VQ-SIGReg prior (same zarr, horizon,
obs_steps, normalization, max_samples) so the only difference at eval is the
generator. This is the apples-to-apples DP-on-harness baseline.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import cycle
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from vq_sigreg.diffusion_policy import DiffusionPolicy
from vq_sigreg.pusht_data import make_openloop_datasets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="vq_sigreg/configs/pusht_openloop.yaml")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--diffusion-steps", type=int, default=100)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = yaml.safe_load(Path(args.config).read_text())
    # DP-specific model block (data block reused verbatim for fairness).
    cfg["model"] = {
        "diffusion_steps": args.diffusion_steps,
        "hidden_dim": args.hidden_dim,
        "t_embed": 128,
        "depth": args.depth,
        "beta_schedule": "cosine",
    }
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, source = make_openloop_datasets(cfg, args.seed)
    print(f"data source={source}  train={len(train_ds)}  val={len(val_ds)}")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=512, shuffle=False)

    model = DiffusionPolicy(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DP params: {n_params/1e6:.3f}M  obs_dim={model.obs_dim}  chunk_dim={model.chunk_dim}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema_model = DiffusionPolicy(cfg).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad_(False)

    def ema_update():
        with torch.no_grad():
            for ep, mp in zip(ema_model.parameters(), model.parameters()):
                ep.mul_(args.ema).add_(mp, alpha=1 - args.ema)
            for eb, mb in zip(ema_model.buffers(), model.buffers()):
                eb.copy_(mb)

    @torch.no_grad()
    def val_loss() -> float:
        model.eval()
        tot, nb = 0.0, 0
        for batch in val_dl:
            obs = batch["obs"].to(device)
            chunk = batch["chunk"].to(device)
            tot += float(model.loss(chunk, obs))
            nb += 1
        model.train()
        return tot / max(nb, 1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = []
    t0 = time.time()
    model.train()
    it = cycle(train_dl)
    run_loss = 0.0
    for step in range(1, args.steps + 1):
        batch = next(it)
        obs = batch["obs"].to(device)
        chunk = batch["chunk"].to(device)
        loss = model.loss(chunk, obs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema_update()
        run_loss += float(loss)
        if step % args.log_every == 0 or step == 1:
            tr = run_loss / (args.log_every if step != 1 else 1)
            run_loss = 0.0
            vl = val_loss()
            logs.append({"step": step, "train_loss": tr, "val_loss": vl, "elapsed_s": time.time() - t0})
            print(f"step {step:>6}  train_eps_mse={tr:.5f}  val_eps_mse={vl:.5f}  ({time.time()-t0:.0f}s)", flush=True)

    ckpt = {
        "model_state": model.state_dict(),
        "ema_state": ema_model.state_dict(),
        "cfg": cfg,
        "kind": "diffusion_policy",
        "steps": args.steps,
    }
    torch.save(ckpt, out_dir / "dp_latest.pt")
    summary = {
        "steps": args.steps,
        "params_M": n_params / 1e6,
        "final_train_loss": logs[-1]["train_loss"],
        "final_val_loss": logs[-1]["val_loss"],
        "data_source": source,
        "config": cfg,
        "logs": logs,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved -> {out_dir/'dp_latest.pt'}")
    print("DP_TRAINING_DONE")


if __name__ == "__main__":
    main()
