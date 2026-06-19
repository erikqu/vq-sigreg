#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_row(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    k = int(summary["config"]["model"]["codebook_size"])
    return {
        "codebook_size": k,
        "source": summary["source"],
        "lejepa_impl": summary.get("lejepa_impl", ""),
        "lejepa_mined_dist_px": float(summary["metrics"]["lejepa"]["mined_nearest_demo_dist_px"]),
        "vq_mined_dist_px": float(summary["metrics"]["vq_sigreg"]["mined_nearest_demo_dist_px"]),
        "vq_demo_routes_covered": float(summary["metrics"]["vq_sigreg"]["mined_demo_routes_covered"]),
        "vq_route_spread_px": float(summary["metrics"]["vq_sigreg"]["mined_vq_route_spread_px"]),
        "vq_all_context_dist_px": float(summary["metrics"]["vq_sigreg"]["mean_nearest_demo_dist_px"]),
        "vq_fork_spread_px": float(summary["metrics"]["vq_sigreg"]["fork_spread_px"]),
        "mined_contexts": float(summary["metrics"]["selection"]["mined_contexts"]),
        "path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Push-T VQ codebook-size sweep.")
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/vq_sigreg/pusht_code_sweep"))
    args = parser.parse_args()

    rows = sorted([load_row(p) for p in args.summaries], key=lambda r: r["codebook_size"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"runs": rows}, f, indent=2)

    ks = [r["codebook_size"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    axes[0].plot(ks, [r["vq_mined_dist_px"] for r in rows], marker="o", label="VQ-SIGReg")
    axes[0].plot(ks, [r["lejepa_mined_dist_px"] for r in rows], marker="x", label="LeJEPA baseline")
    axes[0].set_title("Mined nearest-demo dist")
    axes[0].set_ylabel("px lower is better")
    axes[1].plot(ks, [r["vq_demo_routes_covered"] for r in rows], marker="o", color="tab:green")
    axes[1].set_title("Demo routes covered")
    axes[1].set_ylabel("max 2")
    axes[2].plot(ks, [r["vq_route_spread_px"] for r in rows], marker="o", color="tab:purple")
    axes[2].set_title("VQ route spread")
    axes[2].set_ylabel("px")
    for ax in axes:
        ax.set_xlabel("codebook size K")
        ax.grid(alpha=0.25)
        ax.set_xticks(ks)
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "code_sweep.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"out": str(args.out_dir), "runs": rows}, indent=2))


if __name__ == "__main__":
    main()
