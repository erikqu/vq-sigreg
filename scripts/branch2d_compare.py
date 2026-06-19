#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from vq_sigreg.data import BranchConfig, branch_centers, sample_dataset, true_neg_log_density
from vq_sigreg.models import LeJEPA2D, VQSigReg2D
from vq_sigreg.sigreg import sigreg_epps_pulley


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_tensor(data: dict[str, np.ndarray], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(data["x"], dtype=torch.float32, device=device),
        torch.as_tensor(data["y"], dtype=torch.float32, device=device),
    )


def _batches(x: torch.Tensor, y: torch.Tensor, batch_size: int):
    idx = torch.randint(0, x.shape[0], (batch_size,), device=x.device)
    return x[idx], y[idx]


def train_lejepa(
    train_data: dict[str, np.ndarray],
    cfg: dict,
    device: torch.device,
) -> tuple[LeJEPA2D, list[dict[str, float]]]:
    model = LeJEPA2D(
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        embedding_dim=int(cfg["model"]["embedding_dim"]),
    ).to(device)
    x_train, y_train = _to_tensor(train_data, device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]))
    logs: list[dict[str, float]] = []

    for step in range(1, int(cfg["train"]["steps"]) + 1):
        xb, yb = _batches(x_train, y_train, int(cfg["train"]["batch_size"]))
        out = model(xb, yb)
        pred = F.mse_loss(out["shat"], out["sy"].detach())
        readout = F.mse_loss(model.decoder(out["sy"]), yb)
        pred_readout = F.mse_loss(out["yhat"], yb)
        sigreg = sigreg_epps_pulley(torch.cat([out["sx"], out["sy"]], dim=0), step)
        loss = (
            pred
            + float(cfg["train"]["lambda_readout"]) * readout
            + float(cfg["train"].get("lambda_pred_readout", 1.0)) * pred_readout
            + float(cfg["train"]["lambda_sigreg"]) * sigreg
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip_norm"]))
        opt.step()
        if step == 1 or step % 200 == 0 or step == int(cfg["train"]["steps"]):
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
    return model, logs


def train_vq_sigreg(
    train_data: dict[str, np.ndarray],
    cfg: dict,
    device: torch.device,
) -> tuple[VQSigReg2D, list[dict[str, float]]]:
    model = VQSigReg2D(
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        embedding_dim=int(cfg["model"]["embedding_dim"]),
        codebook_size=int(cfg["model"]["codebook_size"]),
    ).to(device)
    x_train, y_train = _to_tensor(train_data, device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]))
    logs: list[dict[str, float]] = []
    temp = float(cfg["train"]["select_temp"])
    codebook_size = int(cfg["model"]["codebook_size"])
    lambda_code_entropy = float(cfg["train"].get("lambda_code_entropy", 0.01))

    for step in range(1, int(cfg["train"]["steps"]) + 1):
        xb, yb = _batches(x_train, y_train, int(cfg["train"]["batch_size"]))
        out = model(xb, yb, select_temp=temp)
        dist = (out["shat"] - out["sy"].detach()[:, None, :]).square().sum(dim=-1)
        softmin = -torch.logsumexp(-temp * dist, dim=-1) / temp
        resp = torch.softmax(-temp * dist, dim=-1)
        usage = resp.mean(dim=0)
        usage_kl = (usage * (usage * codebook_size + 1e-12).log()).sum()
        pred = softmin.mean()
        readout = F.mse_loss(model.decoder(out["sy"]), yb)
        pred_readout = (out["yhat"] - yb[:, None, :]).square().sum(dim=-1).min(dim=-1).values.mean()
        sigreg = sigreg_epps_pulley(torch.cat([out["sx"], out["sy"]], dim=0), step)
        loss = (
            pred
            + float(cfg["train"]["lambda_readout"]) * readout
            + float(cfg["train"].get("lambda_pred_readout", 1.0)) * pred_readout
            + float(cfg["train"]["lambda_sigreg"]) * sigreg
            + lambda_code_entropy * usage_kl
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip_norm"]))
        opt.step()
        if step == 1 or step % 200 == 0 or step == int(cfg["train"]["steps"]):
            logs.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "pred": float(pred.detach().cpu()),
                    "readout": float(readout.detach().cpu()),
                    "pred_readout": float(pred_readout.detach().cpu()),
                    "sigreg": float(sigreg.detach().cpu()),
                    "usage_kl": float(usage_kl.detach().cpu()),
                    "code_perplexity": float(out["code_perplexity"].detach().cpu()),
                }
            )
    return model, logs


def _branch_metrics(
    x: np.ndarray,
    preds: np.ndarray,
    cfg: BranchConfig,
    tau: float,
) -> dict[str, float]:
    centers = branch_centers(x.reshape(-1), cfg)
    dists = np.linalg.norm(preds[None, :, :, :] - centers[:, :, None, :], axis=-1)
    nearest = dists.min(axis=0)
    nearest_branch = dists.argmin(axis=0)
    close = nearest <= tau
    covered = []
    for i in range(preds.shape[0]):
        covered.append(len(set(nearest_branch[i, close[i]].tolist())))
    nll = true_neg_log_density(
        np.repeat(x.reshape(-1), preds.shape[1]),
        preds.reshape(-1, 2),
        cfg,
    )
    return {
        "on_manifold_rate": float(close.mean()),
        "mean_nearest_branch_dist": float(nearest.mean()),
        "mean_neg_log_density": float(nll.mean()),
        "branches_covered_per_context": float(np.mean(covered)),
    }


@torch.no_grad()
def evaluate(
    lejepa: LeJEPA2D,
    vq: VQSigReg2D,
    test_data: dict[str, np.ndarray],
    branch_cfg: BranchConfig,
    cfg: dict,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    x_test, y_test = _to_tensor(test_data, device)
    tau = float(cfg["eval"]["tau_mult"]) * float(branch_cfg.thickness)

    lejepa_pred = lejepa.predict_y(x_test).cpu().numpy()[:, None, :]
    vq_pred = vq.predict_all_y(x_test).cpu().numpy()
    metrics = {
        "lejepa": _branch_metrics(test_data["x"], lejepa_pred, branch_cfg, tau),
        "vq_sigreg": _branch_metrics(test_data["x"], vq_pred, branch_cfg, tau),
    }

    out = vq(x_test, y_test, select_temp=float(cfg["train"]["select_temp"]))
    winners = out["winner"].cpu().numpy()
    hist = np.bincount(winners, minlength=int(cfg["model"]["codebook_size"])).astype(np.float64)
    probs = hist / max(1.0, hist.sum())
    perplexity = np.exp(-(probs * np.log(probs + 1e-12)).sum())
    metrics["vq_sigreg"].update(
        {
            "code_usage_perplexity": float(perplexity),
            "code_usage_active": float((hist > 0).sum()),
            "rate_bits": float(math.log2(int(cfg["model"]["codebook_size"]))),
        }
    )
    metrics["lejepa"].update({"code_usage_perplexity": 1.0, "code_usage_active": 1.0, "rate_bits": 0.0})
    return metrics


def plot_predictions(
    lejepa: LeJEPA2D,
    vq: VQSigReg2D,
    train_data: dict[str, np.ndarray],
    branch_cfg: BranchConfig,
    cfg: dict,
    device: torch.device,
    out_path: Path,
) -> None:
    x_grid = np.linspace(-branch_cfg.x_range, branch_cfg.x_range, 240, dtype=np.float32)[:, None]
    xt = torch.as_tensor(x_grid, dtype=torch.float32, device=device)
    with torch.no_grad():
        le = lejepa.predict_y(xt).cpu().numpy()
        vq_pred = vq.predict_all_y(xt).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for ax, title in zip(axes, ["LeJEPA deterministic prediction", "VQ-SIGReg code predictions"]):
        ax.scatter(train_data["y"][:, 0], train_data["y"][:, 1], s=2, c="0.8", alpha=0.45, label="data")
        ax.set_title(title)
        ax.set_xlabel("y[0]")
        ax.set_ylabel("y[1]")
    axes[0].plot(le[:, 0], le[:, 1], color="tab:red", lw=2.0, label="prediction")
    colors = plt.cm.tab10(np.linspace(0, 1, vq_pred.shape[1]))
    for k in range(vq_pred.shape[1]):
        axes[1].plot(vq_pred[:, k, 0], vq_pred[:, k, 1], color=colors[k], lw=2.0, label=f"code {k}")
    for ax in axes:
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle(f"{branch_cfg.n_branches}-branch conditional benchmark")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_losses(logs: dict[str, list[dict[str, float]]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for name, rows in logs.items():
        steps = [r["step"] for r in rows]
        axes[0].plot(steps, [r["loss"] for r in rows], label=name)
        axes[1].plot(steps, [r["pred"] for r in rows], label=name)
        if name == "vq_sigreg":
            axes[2].plot(steps, [r["code_perplexity"] for r in rows], label=name)
    axes[0].set_title("train loss")
    axes[1].set_title("prediction loss")
    axes[2].set_title("VQ code perplexity")
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_branch(branches: int, cfg: dict, device: torch.device, out_dir: Path) -> dict:
    branch_cfg = BranchConfig(
        n_branches=branches,
        separation=float(cfg["data"]["separation"]),
        thickness=float(cfg["data"]["thickness"]),
        wiggle=float(cfg["data"]["wiggle"]),
    )
    train_data = sample_dataset(int(cfg["data"]["n_train"]), branch_cfg, seed=int(cfg["seed"]) + branches)
    test_data = sample_dataset(int(cfg["data"]["n_test"]), branch_cfg, seed=int(cfg["seed"]) + 100 + branches)
    lejepa, lejepa_logs = train_lejepa(train_data, cfg, device)
    vq, vq_logs = train_vq_sigreg(train_data, cfg, device)
    metrics = evaluate(lejepa, vq, test_data, branch_cfg, cfg, device)
    plot_predictions(lejepa, vq, train_data, branch_cfg, cfg, device, out_dir / f"branch{branches}_predictions.png")
    plot_losses({"lejepa": lejepa_logs, "vq_sigreg": vq_logs}, out_dir / f"branch{branches}_training.png")
    return {
        "branches": branches,
        "metrics": metrics,
        "logs": {"lejepa": lejepa_logs, "vq_sigreg": vq_logs},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="vq_sigreg/configs/branch2d.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.steps is not None:
        cfg["train"]["steps"] = int(args.steps)
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir
    device_name = args.device or cfg.get("device", "cpu")
    device = torch.device(device_name)
    _set_seed(int(cfg["seed"]))

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "config": cfg,
        "runs": [run_branch(int(b), cfg, device, out_dir) for b in cfg["data"]["branches"]],
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({"summary": str(summary_path), "runs": results["runs"]}, indent=2))


if __name__ == "__main__":
    main()
