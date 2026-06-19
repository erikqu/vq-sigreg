from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch


def _load_official_lejepa():
    root = Path(__file__).resolve().parents[1]
    external = root / "external" / "lejepa"
    if external.exists():
        sys.path.insert(0, str(external))
    return importlib.import_module("lejepa")


class OfficialLeJEPASIGReg(torch.nn.Module):
    """Official galilai-group/lejepa SIGReg wrapper.

    The upstream repository provides the LeJEPA statistical regularizers rather
    than a task-specific Push-T predictor. This wrapper uses the documented
    Epps-Pulley univariate test with multivariate random slicing.
    """

    def __init__(
        self,
        num_slices: int = 64,
        n_points: int = 17,
        t_max: float = 3.0,
        reduction: str = "mean",
    ):
        super().__init__()
        lejepa = _load_official_lejepa()
        test = lejepa.univariate.EppsPulley(t_max=float(t_max), n_points=int(n_points))
        self.loss = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=test,
            num_slices=int(num_slices),
            reduction=reduction,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            embeddings = embeddings.flatten(0, -2)
        dtype = embeddings.dtype
        return self.loss(embeddings.float()).to(dtype)

