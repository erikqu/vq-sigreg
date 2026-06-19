from __future__ import annotations

import torch


def sigreg_epps_pulley(
    embeddings: torch.Tensor,
    global_step: int = 0,
    num_slices: int = 64,
    num_knots: int = 17,
) -> torch.Tensor:
    """Sketched Isotropic Gaussian Regularization.

    This is the LeJEPA-style marginal anti-collapse term: it pushes embeddings
    toward an isotropic Gaussian without contrastive negatives or teacher/EMA
    heuristics.
    """
    original_dtype = embeddings.dtype
    embeddings = embeddings.float()
    if embeddings.ndim != 2:
        embeddings = embeddings.flatten(0, -2)
    n, dim = embeddings.shape
    if n < 2:
        return embeddings.new_zeros(()).to(original_dtype)

    generator = torch.Generator(device=embeddings.device)
    generator.manual_seed(int(global_step))
    directions = torch.randn(dim, num_slices, generator=generator, device=embeddings.device)
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-8)
    t = torch.linspace(-5.0, 5.0, num_knots, device=embeddings.device, dtype=embeddings.dtype)

    projected = embeddings @ directions
    ecf = torch.exp(1j * projected.unsqueeze(-1) * t).mean(dim=0)
    target_cf = torch.exp(-0.5 * t.square())
    err = (ecf - target_cf).abs().square() * target_cf
    return (torch.trapz(err, t, dim=-1).mean() * n).to(original_dtype)

