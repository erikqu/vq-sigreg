#!/usr/bin/env python
"""Populate artifacts/vq_sigreg/pusht_gallery with closed-loop Push-T rollouts.

Renders single-episode GIFs and start/mid/end stills for VQ-SIGReg and the
in-repo Diffusion Policy. Neutral gallery output - no paired comparison framing.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio

REPO = Path(__file__).resolve().parents[3]
GALLERY = REPO / "vq_sigreg" / "assets" / "gallery"

DEFAULT_SEEDS = [10_000, 10_005, 10_012, 10_018, 10_027, 10_042, 10_046]


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)


def save_vq_stills(gif_path: Path, out_dir: Path, prefix: str, seed: int) -> list[str]:
    start = mid = end = None
    count = 0
    with imageio.get_reader(gif_path) as reader:
        for frame in reader:
            if count == 0:
                start = frame
            mid = frame
            end = frame
            count += 1
    if count == 0:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for frame, name in ((start, "start"), (mid, "mid"), (end, "end")):
        path = out_dir / f"{prefix}_seed{seed}_{name}.png"
        imageio.imwrite(path, frame)
        saved.append(str(path.relative_to(REPO)))
    return saved


def copy_existing(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"copied {src.relative_to(REPO)} -> {dst.relative_to(REPO)}")


def add_existing_assets(manifest: dict, root: Path, policy: str) -> None:
    for path in sorted(root.glob("*")):
        if path.suffix.lower() not in {".gif", ".png", ".json"}:
            continue
        rel = str(path.relative_to(REPO))
        if any(asset.get("path") == rel for asset in manifest["assets"]):
            continue
        manifest["assets"].append({"kind": path.suffix.lstrip("."), "policy": policy, "path": rel})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    ap.add_argument("--skip-render", action="store_true", help="Only copy existing assets and write manifest.")
    args = ap.parse_args()

    gallery = GALLERY
    vq_dir = gallery / "vq_sigreg"
    dp_dir = gallery / "diffusion"
    stills_vq = gallery / "stills" / "vq_sigreg"
    stills_dp = gallery / "stills" / "diffusion"
    for d in (vq_dir, dp_dir, stills_vq, stills_dp):
        d.mkdir(parents=True, exist_ok=True)

    vq_ckpt = REPO / "outputs/vq_sigreg_pusht_k16_continuous_residual_s003_g48_a035/vq_continuous_residual_latest.pt"
    dp_ckpt = REPO / "outputs/vq_sigreg_pusht_dp/dp_latest.pt"
    prior_ckpt = REPO / "outputs/vq_sigreg_pusht_k16_prior/vq_sigreg_latest.pt"

    manifest: dict = {"seeds": [int(s) for s in args.seeds], "assets": []}

    if not args.skip_render:
        for seed in args.seeds:
            out = vq_dir / f"recovery_seed{seed}.gif"
            run([
                sys.executable, "vq_sigreg/scripts/render_pusht_gif.py",
                "--checkpoint", str(vq_ckpt),
                "--device", args.device,
                "--episodes", "1", "--start-seed", str(seed),
                "--replan-every", "16",
                "--selector", "recovery_prior",
                "--recovery-push-px", "200",
                "--recovery-max-uses", "1",
                "--recovery-goal-progress-px", "-2",
                "--keep", "1", "--every", "1",
                "--out", str(out),
                "--json-out", str(vq_dir / f"recovery_seed{seed}.json"),
                "--quiet",
            ])
            manifest["assets"].append({"kind": "gif", "policy": "vq_sigreg_recovery", "seed": seed, "path": str(out.relative_to(REPO))})
            for p in save_vq_stills(out, stills_vq, "vq_recovery", seed):
                manifest["assets"].append({"kind": "still", "policy": "vq_sigreg_recovery", "seed": seed, "path": p})

            dp_out = dp_dir / f"seed{seed}.gif"
            run([
                sys.executable, "vq_sigreg/scripts/diagnostics/render_dp_rollout.py",
                "--checkpoint", str(dp_ckpt),
                "--device", args.device,
                "--seeds", str(seed),
                "--replan-every", "16",
                "--sample-steps", "25",
                "--single-out-dir", str(dp_dir),
                "--stills-dir", str(stills_dp),
                "--every", "1", "--no-gif",
            ])
            # render_dp writes seed{seed}.gif when single-out-dir set
            if dp_out.exists():
                manifest["assets"].append({"kind": "gif", "policy": "diffusion", "seed": seed, "path": str(dp_out.relative_to(REPO))})

    # Curated rollouts already in outputs/
    copies = [
        (REPO / "outputs/vq_sigreg_pusht_k16_prior/vq_prior_rollout.gif", vq_dir / "prior_top3_grid.gif"),
        (REPO / "outputs/vq_sigreg_pusht_k16_prior/lejepa_rollout.gif", vq_dir / "lejepa_top3_grid.gif"),
        (REPO / "outputs/vq_sigreg_pusht_k16/vq_rollout.gif", vq_dir / "k16_early_top3_grid.gif"),
        (REPO / "artifacts/vq_sigreg/latest_closed_loop_gifs/prior_h16_top4.gif", vq_dir / "prior_h16_top4_grid.gif"),
        (REPO / "artifacts/vq_sigreg/latest_closed_loop_gifs/conservative_residual_h16_top4.gif", vq_dir / "conservative_residual_h16_top4_grid.gif"),
        (REPO / "artifacts/vq_sigreg/latest_closed_loop_gifs/residual_recovery_h16_top4.gif", vq_dir / "residual_recovery_h16_top4_grid.gif"),
        (REPO / "artifacts/vq_sigreg/latest_closed_loop_gifs/recovery_seed10027.gif", vq_dir / "recovery_seed10027_highlight.gif"),
        (REPO / "outputs/vq_sigreg_pusht_k16_prior/failure_seed_10046_h16.gif", vq_dir / "failure_seed10046.gif"),
        (REPO / "outputs/vq_sigreg_pusht_k16_prior/failure_seed_10027_h16.gif", vq_dir / "failure_seed10027.gif"),
        (REPO / "outputs/_approval/dead_10046.gif", vq_dir / "dead_seed10046.gif"),
        (REPO / "outputs/_approval/dead_10018.gif", vq_dir / "dead_seed10018.gif"),
        (REPO / "outputs/_approval/dead_10042.gif", vq_dir / "dead_seed10042.gif"),
        (REPO / "outputs/vq_sigreg_pusht_k16_prior/training.png", gallery / "training_curves.png"),
        (REPO / "outputs/vq_sigreg_pusht_k16_prior/trajectories.png", gallery / "openloop_trajectories.png"),
        (REPO / "artifacts/vq_sigreg/pusht_openloop_official_lejepa/trajectories.png", gallery / "official_lejepa_trajectories.png"),
        (REPO / "artifacts/vq_sigreg/pusht_openloop_official_lejepa/training.png", gallery / "official_lejepa_training.png"),
    ]
    for src, dst in copies:
        copy_existing(src, dst)
        if dst.exists():
            manifest["assets"].append({"kind": dst.suffix.lstrip("."), "policy": "archive", "path": str(dst.relative_to(REPO))})

    add_existing_assets(manifest, vq_dir, "vq_sigreg")
    add_existing_assets(manifest, dp_dir, "diffusion")
    add_existing_assets(manifest, stills_vq, "vq_sigreg")
    add_existing_assets(manifest, stills_dp, "diffusion")

    # One-shot prior grid from the anchor checkpoint
    if not args.skip_render and prior_ckpt.exists():
        prior_grid = vq_dir / "prior_anchor_top3_grid.gif"
        if not prior_grid.exists():
            run([
                sys.executable, "vq_sigreg/scripts/render_pusht_gif.py",
                "--checkpoint", str(prior_ckpt),
                "--device", args.device,
                "--episodes", "8", "--start-seed", "10000",
                "--replan-every", "16", "--keep", "3",
                "--out", str(prior_grid),
                "--quiet",
            ])
        manifest["assets"].append({"kind": "gif", "policy": "vq_sigreg_prior", "path": str(prior_grid.relative_to(REPO))})

    # DP multi-episode grid
    if not args.skip_render and dp_ckpt.exists():
        dp_grid = dp_dir / "top3_grid.gif"
        if not dp_grid.exists():
            run([
                sys.executable, "vq_sigreg/scripts/diagnostics/render_dp_rollout.py",
                "--checkpoint", str(dp_ckpt),
                "--device", args.device,
                "--episodes", "8", "--start-seed", "10000",
                "--replan-every", "16", "--keep", "3",
                "--sample-steps", "25",
                "--out", str(dp_grid),
            ])
        manifest["assets"].append({"kind": "gif", "policy": "diffusion", "path": str(dp_grid.relative_to(REPO))})

    (gallery / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"gallery ready -> {gallery.relative_to(REPO)}")


if __name__ == "__main__":
    main()
