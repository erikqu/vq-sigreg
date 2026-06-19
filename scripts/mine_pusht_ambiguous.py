#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.vq_sigreg.pusht_openloop import chunks_to_pixels, mine_ambiguous_contexts  # noqa: E402
from vq_sigreg.pusht_data import make_openloop_datasets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine real Push-T validation contexts with multimodal NN futures.")
    parser.add_argument("--config", default="vq_sigreg/configs/pusht_openloop.yaml")
    parser.add_argument("--out", default="artifacts/vq_sigreg/pusht_openloop_mined/mined_contexts.json")
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _train_ds, val_ds, source = make_openloop_datasets(cfg, int(cfg["seed"]))
    if source != "real":
        raise SystemExit(f"Expected real Push-T data, got source={source!r}")

    loader = DataLoader(val_ds, batch_size=min(4096, len(val_ds)), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    horizon = int(cfg["data"]["horizon"])
    payload = {
        "obs": batch["obs"].numpy(),
        "raw_state": batch["raw_state"].numpy(),
        "demo_paths": chunks_to_pixels(batch["chunk"], horizon),
    }
    mined, nn, metrics = mine_ambiguous_contexts(payload, cfg)
    rows = []
    for i in mined[: int(args.limit)]:
        sides = payload["demo_paths"][nn[int(i), : int(cfg["eval"]["nn_k"])], :, 1]
        rows.append(
            {
                "val_index": int(i),
                "raw_state": [float(x) for x in payload["raw_state"][int(i)].tolist()],
                "neighbor_indices": [int(x) for x in nn[int(i), : int(cfg["eval"]["nn_k"])].tolist()],
                "neighbor_y_min": float(np.min(sides)),
                "neighbor_y_max": float(np.max(sides)),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"source": source, "metrics": metrics, "contexts": rows}, f, indent=2)
    print(json.dumps({"out": str(out), "source": source, "metrics": metrics, "num_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
