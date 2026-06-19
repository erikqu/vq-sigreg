from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import nn


def mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class TransformerChunkDecoder(nn.Module):
    """Small ACT-style chunk decoder with learned action queries.

    A latent embedding conditions a short sequence of learned query tokens. The
    transformer lets action steps coordinate before a final per-step action head.
    """

    def __init__(
        self,
        embedding_dim: int,
        chunk_dim: int,
        action_dim: int = 2,
        model_dim: int | None = None,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.chunk_dim = int(chunk_dim)
        self.action_dim = int(action_dim)
        if self.chunk_dim % self.action_dim != 0:
            raise ValueError(f"chunk_dim={chunk_dim} must be divisible by action_dim={action_dim}")
        self.horizon = self.chunk_dim // self.action_dim
        self.model_dim = int(model_dim or embedding_dim)
        self.causal = bool(causal)
        if self.model_dim % int(num_heads) != 0:
            raise ValueError(f"model_dim={self.model_dim} must be divisible by num_heads={num_heads}")

        self.context_proj = nn.Linear(self.embedding_dim, self.model_dim)
        self.query_tokens = nn.Parameter(torch.randn(self.horizon, self.model_dim) / math.sqrt(self.model_dim))
        self.pos_embed = nn.Parameter(torch.randn(self.horizon + 1, self.model_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(num_heads),
            dim_feedforward=self.model_dim * int(ff_mult),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(num_layers), enable_nested_tensor=False)
        self.norm = nn.LayerNorm(self.model_dim)
        self.action_head = nn.Linear(self.model_dim, self.action_dim)

    def _causal_mask(self, device: torch.device) -> torch.Tensor | None:
        if not self.causal:
            return None
        seq_len = self.horizon + 1
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        mask[1:, 1:] = torch.triu(torch.ones(self.horizon, self.horizon, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        leading_shape = embedding.shape[:-1]
        flat = embedding.reshape(-1, self.embedding_dim)
        context = self.context_proj(flat)[:, None, :]
        queries = self.query_tokens[None, :, :].expand(flat.shape[0], -1, -1)
        tokens = torch.cat([context, queries], dim=1) + self.pos_embed[None, :, :]
        decoded = self.transformer(tokens, mask=self._causal_mask(tokens.device))
        actions = self.action_head(self.norm(decoded[:, 1:, :]))
        return actions.reshape(*leading_shape, self.chunk_dim)


class EnumeratedVQ(nn.Module):
    """Small enumerated codebook: exact conditional branches, no AR decoding."""

    def __init__(self, embedding_dim: int, codebook_size: int):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.codebook_size = int(codebook_size)
        self.codebook = nn.Parameter(torch.randn(codebook_size, embedding_dim) / math.sqrt(embedding_dim))

    def enumerate_codes(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.codebook.to(device=device, dtype=dtype)
        rate = torch.full((self.codebook_size,), math.log2(self.codebook_size), device=device, dtype=dtype)
        return z, rate


class LeJEPA2D(nn.Module):
    """Deterministic JEPA + SIGReg baseline.

    It has the LeJEPA blind spot by construction: one context embedding produces
    one predicted target embedding, so multimodal conditionals average.
    """

    def __init__(self, hidden_dim: int = 128, embedding_dim: int = 2):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.obs_encoder = mlp(1, hidden_dim, embedding_dim)
        self.target_encoder = mlp(2, hidden_dim, embedding_dim)
        self.predictor = mlp(embedding_dim, hidden_dim, embedding_dim)
        self.decoder = mlp(embedding_dim, hidden_dim, 2)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        sx = self.obs_encoder(x)
        sy = self.target_encoder(y)
        shat = self.predictor(sx)
        return {"sx": sx, "sy": sy, "shat": shat, "yhat": self.decoder(shat)}

    @torch.no_grad()
    def predict_y(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.predictor(self.obs_encoder(x)))


class VQSigReg2D(nn.Module):
    """VQ-LeJEPA: LeJEPA encoders + SIGReg, with discrete conditional branches."""

    def __init__(self, hidden_dim: int = 128, embedding_dim: int = 2, codebook_size: int = 4):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.codebook_size = int(codebook_size)
        self.obs_encoder = mlp(1, hidden_dim, embedding_dim)
        self.target_encoder = mlp(2, hidden_dim, embedding_dim)
        self.latent = EnumeratedVQ(embedding_dim, codebook_size)
        self.predictor = mlp(embedding_dim + embedding_dim, hidden_dim, embedding_dim)
        self.decoder = mlp(embedding_dim, hidden_dim, 2)

    def code_predictions(self, sx: torch.Tensor) -> dict[str, torch.Tensor]:
        z, rate = self.latent.enumerate_codes(sx.device, sx.dtype)
        batch, num_codes = sx.shape[0], z.shape[0]
        sx_rep = sx[:, None, :].expand(batch, num_codes, sx.shape[-1])
        z_rep = z[None, :, :].expand(batch, num_codes, z.shape[-1])
        shat = self.predictor(torch.cat([sx_rep, z_rep], dim=-1).reshape(batch * num_codes, -1))
        shat = shat.reshape(batch, num_codes, -1)
        return {"shat": shat, "z": z, "rate_bits": rate}

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        select_temp: float = 64.0,
    ) -> dict[str, torch.Tensor]:
        sx = self.obs_encoder(x)
        sy = self.target_encoder(y)
        codes = self.code_predictions(sx)
        dist = (codes["shat"] - sy[:, None, :]).square().sum(dim=-1)
        hard_dist, winner = dist.min(dim=-1)
        softmin = -torch.logsumexp(-float(select_temp) * dist, dim=-1) / float(select_temp)
        resp = torch.softmax(-float(select_temp) * dist, dim=-1)
        avg_resp = resp.mean(dim=0)
        code_perplexity = torch.exp(-(avg_resp * (avg_resp + 1e-12).log()).sum())
        per_example_perplexity = torch.exp(-(resp * (resp + 1e-12).log()).sum(dim=-1)).mean()
        yhat = self.decoder(codes["shat"].reshape(-1, self.embedding_dim)).reshape(x.shape[0], -1, 2)
        return {
            "sx": sx,
            "sy": sy,
            "shat": codes["shat"],
            "yhat": yhat,
            "dist": dist,
            "softmin": softmin,
            "hard_dist": hard_dist,
            "winner": winner,
            "responsibilities": resp,
            "code_perplexity": code_perplexity,
            "per_example_perplexity": per_example_perplexity,
            "rate_bits": codes["rate_bits"],
        }

    @torch.no_grad()
    def predict_all_y(self, x: torch.Tensor) -> torch.Tensor:
        sx = self.obs_encoder(x)
        shat = self.code_predictions(sx)["shat"]
        return self.decoder(shat.reshape(-1, self.embedding_dim)).reshape(x.shape[0], -1, 2)


class LeJEPAOpenLoop(nn.Module):
    """Deterministic chunk predictor for open-loop Push-T visuals."""

    def __init__(
        self,
        obs_dim: int,
        chunk_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 32,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.chunk_dim = int(chunk_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.obs_encoder = mlp(obs_dim, hidden_dim, embedding_dim)
        self.target_encoder = mlp(chunk_dim, hidden_dim, embedding_dim)
        self.predictor = mlp(embedding_dim, hidden_dim, embedding_dim)
        self.decoder = mlp(embedding_dim, hidden_dim, chunk_dim)

    def forward(self, obs: torch.Tensor, chunk: torch.Tensor) -> dict[str, torch.Tensor]:
        sx = self.obs_encoder(obs)
        sy = self.target_encoder(chunk)
        shat = self.predictor(sx)
        return {"sx": sx, "sy": sy, "shat": shat, "chunk_hat": self.decoder(shat)}

    @torch.no_grad()
    def predict_chunk(self, obs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.predictor(self.obs_encoder(obs)))


class VQSigRegOpenLoop(nn.Module):
    """Enumerated-code chunk predictor for open-loop Push-T visuals."""

    def __init__(
        self,
        obs_dim: int,
        chunk_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 32,
        codebook_size: int = 4,
        decoder_type: str = "mlp",
        action_dim: int = 2,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_dim: int | None = None,
        transformer_ff_mult: int = 4,
        transformer_dropout: float = 0.0,
        transformer_causal: bool = False,
        final_align_residual: bool = False,
        final_align_residual_scale: float = 0.12,
        final_align_gate_scale_px: float = 96.0,
        delta_action_residual: bool = False,
        delta_action_residual_scale: float = 0.12,
        delta_action_gate_scale_px: float = 96.0,
        delta_action_gate_angle_rad: float = 0.75,
        continuous_action_residual: bool = False,
        continuous_action_residual_scale: float = 0.06,
        continuous_action_gate_scale_px: float = 48.0,
        continuous_action_gate_angle_rad: float = 0.35,
        continuous_action_steps: int = 1,
        local_continuous_action_residual: bool = False,
        local_continuous_action_residual_scale: float = 0.06,
        local_continuous_action_gate_scale_px: float = 48.0,
        local_continuous_action_gate_angle_rad: float = 0.35,
        local_continuous_action_steps: int = 1,
        local_continuous_action_last_steps: int = 0,
        local_spline_action_residual: bool = False,
        local_spline_action_residual_scale: float = 0.06,
        local_spline_action_gate_scale_px: float = 48.0,
        local_spline_action_gate_angle_rad: float = 0.35,
        contact_action_residual: bool = False,
        contact_action_residual_scale: float = 0.06,
        contact_action_gate_scale_px: float = 48.0,
        contact_action_gate_angle_rad: float = 0.35,
        contact_action_steps: int = 1,
        multi_contact_action_residual: bool = False,
        multi_contact_action_residual_scale: float = 0.06,
        multi_contact_action_gate_scale_px: float = 48.0,
        multi_contact_action_gate_angle_rad: float = 0.35,
        multi_contact_action_samples: int = 4,
        flow_action_residual: bool = False,
        flow_action_residual_scale: float = 0.06,
        flow_action_gate_scale_px: float = 48.0,
        flow_action_gate_angle_rad: float = 0.35,
        flow_action_steps: int = 1,
        hierarchical_flow_decoder: bool = False,
        hierarchical_flow_steps: int = 2,
        hierarchical_flow_noise_scale: float = 0.0,
        hierarchical_flow_samples: int = 1,
        hierarchical_flow_relative: bool = False,
        hierarchical_flow_no_z: bool = False,
        fine_vq_residual: bool = False,
        fine_vq_codebook_size: int = 8,
        fine_vq_residual_scale: float = 0.06,
        fine_vq_gate_scale_px: float = 48.0,
        fine_vq_gate_angle_rad: float = 0.35,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.chunk_dim = int(chunk_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.codebook_size = int(codebook_size)
        self.decoder_type = str(decoder_type)
        self.action_dim = int(action_dim)
        if self.chunk_dim % self.action_dim != 0:
            raise ValueError(f"chunk_dim={chunk_dim} must be divisible by action_dim={action_dim}")
        self.horizon = self.chunk_dim // self.action_dim
        self.final_align_residual = bool(final_align_residual)
        self.final_align_residual_scale = float(final_align_residual_scale)
        self.final_align_gate_scale_px = float(final_align_gate_scale_px)
        self.delta_action_residual = bool(delta_action_residual)
        self.delta_action_residual_scale = float(delta_action_residual_scale)
        self.delta_action_gate_scale_px = float(delta_action_gate_scale_px)
        self.delta_action_gate_angle_rad = float(delta_action_gate_angle_rad)
        self.continuous_action_residual = bool(continuous_action_residual)
        self.continuous_action_residual_scale = float(continuous_action_residual_scale)
        self.continuous_action_gate_scale_px = float(continuous_action_gate_scale_px)
        self.continuous_action_gate_angle_rad = float(continuous_action_gate_angle_rad)
        self.continuous_action_steps = max(1, int(continuous_action_steps))
        self.local_continuous_action_residual = bool(local_continuous_action_residual)
        self.local_continuous_action_residual_scale = float(local_continuous_action_residual_scale)
        self.local_continuous_action_gate_scale_px = float(local_continuous_action_gate_scale_px)
        self.local_continuous_action_gate_angle_rad = float(local_continuous_action_gate_angle_rad)
        self.local_continuous_action_steps = max(1, int(local_continuous_action_steps))
        self.local_continuous_action_last_steps = max(0, int(local_continuous_action_last_steps))
        self.local_spline_action_residual = bool(local_spline_action_residual)
        self.local_spline_action_residual_scale = float(local_spline_action_residual_scale)
        self.local_spline_action_gate_scale_px = float(local_spline_action_gate_scale_px)
        self.local_spline_action_gate_angle_rad = float(local_spline_action_gate_angle_rad)
        self.contact_action_residual = bool(contact_action_residual)
        self.contact_action_residual_scale = float(contact_action_residual_scale)
        self.contact_action_gate_scale_px = float(contact_action_gate_scale_px)
        self.contact_action_gate_angle_rad = float(contact_action_gate_angle_rad)
        self.contact_action_steps = max(1, int(contact_action_steps))
        self.contact_feature_dim = 10
        self.multi_contact_action_residual = bool(multi_contact_action_residual)
        self.multi_contact_action_residual_scale = float(multi_contact_action_residual_scale)
        self.multi_contact_action_gate_scale_px = float(multi_contact_action_gate_scale_px)
        self.multi_contact_action_gate_angle_rad = float(multi_contact_action_gate_angle_rad)
        self.multi_contact_action_samples = max(1, int(multi_contact_action_samples))
        self.flow_action_residual = bool(flow_action_residual)
        self.flow_action_residual_scale = float(flow_action_residual_scale)
        self.flow_action_gate_scale_px = float(flow_action_gate_scale_px)
        self.flow_action_gate_angle_rad = float(flow_action_gate_angle_rad)
        self.flow_action_steps = max(1, int(flow_action_steps))
        self.hierarchical_flow_decoder = bool(hierarchical_flow_decoder)
        self.hierarchical_flow_steps = max(1, int(hierarchical_flow_steps))
        self.hierarchical_flow_noise_scale = float(hierarchical_flow_noise_scale)
        self.hierarchical_flow_samples = max(1, int(hierarchical_flow_samples))
        self.hierarchical_flow_relative = bool(hierarchical_flow_relative)
        self.hierarchical_flow_no_z = bool(hierarchical_flow_no_z)
        self.fine_vq_residual = bool(fine_vq_residual)
        self.fine_vq_codebook_size = int(fine_vq_codebook_size)
        self.fine_vq_residual_scale = float(fine_vq_residual_scale)
        self.fine_vq_gate_scale_px = float(fine_vq_gate_scale_px)
        self.fine_vq_gate_angle_rad = float(fine_vq_gate_angle_rad)
        self.obs_encoder = mlp(obs_dim, hidden_dim, embedding_dim)
        self.target_encoder = mlp(chunk_dim, hidden_dim, embedding_dim)
        self.latent = EnumeratedVQ(embedding_dim, codebook_size)
        self.predictor = mlp(embedding_dim + embedding_dim, hidden_dim, embedding_dim)
        if self.decoder_type == "transformer":
            self.decoder = TransformerChunkDecoder(
                embedding_dim=embedding_dim,
                chunk_dim=chunk_dim,
                action_dim=action_dim,
                model_dim=transformer_dim,
                num_layers=transformer_layers,
                num_heads=transformer_heads,
                ff_mult=transformer_ff_mult,
                dropout=transformer_dropout,
                causal=transformer_causal,
            )
        elif self.decoder_type == "mlp":
            self.decoder = mlp(embedding_dim, hidden_dim, chunk_dim)
        else:
            raise ValueError(f"unknown decoder_type={decoder_type!r}")
        self.prior_head = mlp(embedding_dim, hidden_dim, codebook_size)
        self.reranker_head = mlp(embedding_dim + embedding_dim + chunk_dim, hidden_dim, 1)
        if self.final_align_residual:
            self.final_align_head = mlp(embedding_dim + embedding_dim, hidden_dim, chunk_dim)
            nn.init.zeros_(self.final_align_head[-1].weight)
            nn.init.zeros_(self.final_align_head[-1].bias)
        else:
            self.final_align_head = None
        if self.delta_action_residual:
            self.delta_action_head = mlp(obs_dim + self.action_dim + 1, hidden_dim, self.action_dim)
            nn.init.zeros_(self.delta_action_head[-1].weight)
            nn.init.zeros_(self.delta_action_head[-1].bias)
        else:
            self.delta_action_head = None
        if self.continuous_action_residual:
            self.continuous_action_head = mlp(obs_dim + chunk_dim + chunk_dim + 1, hidden_dim, chunk_dim)
            nn.init.zeros_(self.continuous_action_head[-1].weight)
            nn.init.zeros_(self.continuous_action_head[-1].bias)
        else:
            self.continuous_action_head = None
        if self.local_continuous_action_residual:
            self.local_continuous_action_head = mlp(obs_dim + chunk_dim + chunk_dim + 1, hidden_dim, chunk_dim)
            nn.init.zeros_(self.local_continuous_action_head[-1].weight)
            nn.init.zeros_(self.local_continuous_action_head[-1].bias)
        else:
            self.local_continuous_action_head = None
        if self.local_spline_action_residual:
            self.local_spline_action_head = mlp(obs_dim + chunk_dim, hidden_dim, 3 * self.action_dim)
            nn.init.zeros_(self.local_spline_action_head[-1].weight)
            nn.init.zeros_(self.local_spline_action_head[-1].bias)
        else:
            self.local_spline_action_head = None
        if self.contact_action_residual:
            self.contact_action_head = mlp(
                obs_dim + self.contact_feature_dim + chunk_dim + chunk_dim + 1,
                hidden_dim,
                chunk_dim,
            )
            nn.init.zeros_(self.contact_action_head[-1].weight)
            nn.init.zeros_(self.contact_action_head[-1].bias)
        else:
            self.contact_action_head = None
        if self.multi_contact_action_residual:
            self.multi_contact_action_head = mlp(
                obs_dim + self.contact_feature_dim + chunk_dim,
                hidden_dim,
                self.multi_contact_action_samples * chunk_dim,
            )
            nn.init.zeros_(self.multi_contact_action_head[-1].weight)
            nn.init.zeros_(self.multi_contact_action_head[-1].bias)
        else:
            self.multi_contact_action_head = None
        if self.flow_action_residual:
            self.flow_action_head = mlp(
                obs_dim + self.contact_feature_dim + chunk_dim + chunk_dim + 1,
                hidden_dim,
                chunk_dim,
            )
            nn.init.zeros_(self.flow_action_head[-1].weight)
            nn.init.zeros_(self.flow_action_head[-1].bias)
        else:
            self.flow_action_head = None
        if self.hierarchical_flow_decoder:
            self.hierarchical_flow_head = mlp(
                obs_dim + self.contact_feature_dim + embedding_dim + embedding_dim + embedding_dim + chunk_dim + 1,
                hidden_dim,
                chunk_dim,
            )
            nn.init.zeros_(self.hierarchical_flow_head[-1].weight)
            nn.init.zeros_(self.hierarchical_flow_head[-1].bias)
        else:
            self.hierarchical_flow_head = None
        if self.fine_vq_residual:
            self.fine_vq_latent = EnumeratedVQ(embedding_dim, self.fine_vq_codebook_size)
            self.fine_vq_head = mlp(obs_dim + chunk_dim + embedding_dim, hidden_dim, chunk_dim)
            self.fine_vq_prior_head = mlp(obs_dim + chunk_dim, hidden_dim, self.fine_vq_codebook_size)
        else:
            self.fine_vq_latent = None
            self.fine_vq_head = None
            self.fine_vq_prior_head = None

    def code_predictions(self, sx: torch.Tensor) -> dict[str, torch.Tensor]:
        z, rate = self.latent.enumerate_codes(sx.device, sx.dtype)
        batch, num_codes = sx.shape[0], z.shape[0]
        sx_rep = sx[:, None, :].expand(batch, num_codes, sx.shape[-1])
        z_rep = z[None, :, :].expand(batch, num_codes, z.shape[-1])
        shat = self.predictor(torch.cat([sx_rep, z_rep], dim=-1).reshape(batch * num_codes, -1))
        shat = shat.reshape(batch, num_codes, -1)
        return {"shat": shat, "z": z, "rate_bits": rate}

    def _final_alignment_gate(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_dim % 6 != 0:
            return torch.ones(obs.shape[0], device=obs.device, dtype=obs.dtype)
        obs_steps = self.obs_dim // 6
        block_xy = obs.reshape(obs.shape[0], obs_steps, 6)[:, -1, 2:4]
        dist = torch.linalg.norm(block_xy, dim=-1)
        scale = max(self.final_align_gate_scale_px / 256.0, 1e-6)
        return torch.exp(-((dist / scale) ** 2))

    def _apply_final_alignment_residual(
        self,
        obs: torch.Tensor,
        sx: torch.Tensor,
        shat: torch.Tensor,
        chunk_hat: torch.Tensor,
    ) -> torch.Tensor:
        if self.final_align_head is None:
            return chunk_hat
        sx_rep = sx[:, None, :].expand(-1, self.codebook_size, -1)
        residual_input = torch.cat([sx_rep, shat], dim=-1)
        residual = self.final_align_head(residual_input.reshape(-1, residual_input.shape[-1]))
        residual = torch.tanh(residual).reshape(obs.shape[0], self.codebook_size, self.chunk_dim)
        gate = self._final_alignment_gate(obs).reshape(obs.shape[0], 1, 1)
        return chunk_hat + gate * self.final_align_residual_scale * residual

    def _delta_action_gate(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_dim % 6 != 0:
            return torch.ones(obs.shape[0], device=obs.device, dtype=obs.dtype)
        obs_steps = self.obs_dim // 6
        latest = obs.reshape(obs.shape[0], obs_steps, 6)[:, -1]
        block_xy = latest[:, 2:4]
        dist = torch.linalg.norm(block_xy, dim=-1)
        scale = max(self.delta_action_gate_scale_px / 256.0, 1e-6)
        xy_gate = torch.exp(-((dist / scale) ** 2))
        goal = torch.tensor([math.sqrt(0.5), math.sqrt(0.5)], device=obs.device, dtype=obs.dtype)
        angle_vec = F.normalize(latest[:, 4:6], dim=-1)
        angle_err = torch.atan2(angle_vec[:, 1] * goal[0] - angle_vec[:, 0] * goal[1], (angle_vec * goal).sum(dim=-1))
        angle_scale = max(self.delta_action_gate_angle_rad, 1e-6)
        angle_gate = torch.exp(-((angle_err / angle_scale) ** 2))
        return xy_gate * angle_gate

    def _continuous_action_gate(self, obs: torch.Tensor) -> torch.Tensor:
        old_scale = self.delta_action_gate_scale_px
        old_angle = self.delta_action_gate_angle_rad
        self.delta_action_gate_scale_px = self.continuous_action_gate_scale_px
        self.delta_action_gate_angle_rad = self.continuous_action_gate_angle_rad
        try:
            return self._delta_action_gate(obs)
        finally:
            self.delta_action_gate_scale_px = old_scale
            self.delta_action_gate_angle_rad = old_angle

    def _fine_vq_gate(self, obs: torch.Tensor) -> torch.Tensor:
        old_scale = self.delta_action_gate_scale_px
        old_angle = self.delta_action_gate_angle_rad
        self.delta_action_gate_scale_px = self.fine_vq_gate_scale_px
        self.delta_action_gate_angle_rad = self.fine_vq_gate_angle_rad
        try:
            return self._delta_action_gate(obs)
        finally:
            self.delta_action_gate_scale_px = old_scale
            self.delta_action_gate_angle_rad = old_angle

    def _local_continuous_action_gate(self, obs: torch.Tensor) -> torch.Tensor:
        old_scale = self.delta_action_gate_scale_px
        old_angle = self.delta_action_gate_angle_rad
        self.delta_action_gate_scale_px = self.local_continuous_action_gate_scale_px
        self.delta_action_gate_angle_rad = self.local_continuous_action_gate_angle_rad
        try:
            return self._delta_action_gate(obs)
        finally:
            self.delta_action_gate_scale_px = old_scale
            self.delta_action_gate_angle_rad = old_angle

    def _local_spline_action_gate(self, obs: torch.Tensor) -> torch.Tensor:
        old_scale = self.delta_action_gate_scale_px
        old_angle = self.delta_action_gate_angle_rad
        self.delta_action_gate_scale_px = self.local_spline_action_gate_scale_px
        self.delta_action_gate_angle_rad = self.local_spline_action_gate_angle_rad
        try:
            return self._delta_action_gate(obs)
        finally:
            self.delta_action_gate_scale_px = old_scale
            self.delta_action_gate_angle_rad = old_angle

    def _contact_action_gate(self, obs: torch.Tensor) -> torch.Tensor:
        old_scale = self.delta_action_gate_scale_px
        old_angle = self.delta_action_gate_angle_rad
        self.delta_action_gate_scale_px = self.contact_action_gate_scale_px
        self.delta_action_gate_angle_rad = self.contact_action_gate_angle_rad
        try:
            return self._delta_action_gate(obs)
        finally:
            self.delta_action_gate_scale_px = old_scale
            self.delta_action_gate_angle_rad = old_angle

    def _multi_contact_action_gate(self, obs: torch.Tensor) -> torch.Tensor:
        old_scale = self.delta_action_gate_scale_px
        old_angle = self.delta_action_gate_angle_rad
        self.delta_action_gate_scale_px = self.multi_contact_action_gate_scale_px
        self.delta_action_gate_angle_rad = self.multi_contact_action_gate_angle_rad
        try:
            return self._delta_action_gate(obs)
        finally:
            self.delta_action_gate_scale_px = old_scale
            self.delta_action_gate_angle_rad = old_angle

    def _flow_action_gate(self, obs: torch.Tensor) -> torch.Tensor:
        old_scale = self.delta_action_gate_scale_px
        old_angle = self.delta_action_gate_angle_rad
        self.delta_action_gate_scale_px = self.flow_action_gate_scale_px
        self.delta_action_gate_angle_rad = self.flow_action_gate_angle_rad
        try:
            return self._delta_action_gate(obs)
        finally:
            self.delta_action_gate_scale_px = old_scale
            self.delta_action_gate_angle_rad = old_angle

    def contact_features(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_dim % 6 != 0:
            return torch.zeros(obs.shape[0], self.contact_feature_dim, device=obs.device, dtype=obs.dtype)
        latest = obs.reshape(obs.shape[0], self.obs_dim // 6, 6)[:, -1]
        agent_xy = latest[:, 0:2]
        block_xy = latest[:, 2:4]
        angle_vec = F.normalize(latest[:, 4:6], dim=-1)
        block_to_goal = -block_xy
        pusher_to_block = block_xy - agent_xy
        goal_dir = F.normalize(block_to_goal, dim=-1)
        contact_dir = F.normalize(pusher_to_block, dim=-1)
        goal_angle = torch.tensor([math.sqrt(0.5), math.sqrt(0.5)], device=obs.device, dtype=obs.dtype)
        angle_err = torch.atan2(
            angle_vec[:, 1] * goal_angle[0] - angle_vec[:, 0] * goal_angle[1],
            (angle_vec * goal_angle).sum(dim=-1),
        )
        dist_goal = torch.linalg.norm(block_to_goal, dim=-1, keepdim=True)
        dist_contact = torch.linalg.norm(pusher_to_block, dim=-1, keepdim=True)
        contact_dot = (goal_dir * contact_dir).sum(dim=-1, keepdim=True)
        contact_cross = goal_dir[:, 0:1] * contact_dir[:, 1:2] - goal_dir[:, 1:2] * contact_dir[:, 0:1]
        return torch.cat(
            [
                block_to_goal,
                pusher_to_block,
                torch.sin(angle_err).unsqueeze(-1),
                torch.cos(angle_err).unsqueeze(-1),
                dist_goal,
                dist_contact,
                contact_dot,
                contact_cross,
            ],
            dim=-1,
        )

    def apply_delta_action_residual(self, obs: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
        if self.delta_action_head is None:
            return chunks
        leading_shape = chunks.shape[:-1]
        if leading_shape[0] != obs.shape[0]:
            raise ValueError("chunks batch dimension must match obs batch dimension")
        actions = chunks.reshape(*leading_shape, self.horizon, self.action_dim)
        obs_view = obs.reshape(obs.shape[0], *([1] * (actions.ndim - 2)), self.obs_dim)
        obs_rep = obs_view.expand(*actions.shape[:-1], self.obs_dim)
        t = torch.linspace(0.0, 1.0, self.horizon, device=chunks.device, dtype=chunks.dtype)
        t_shape = [1] * (actions.ndim - 2) + [self.horizon, 1]
        t_rep = t.reshape(*t_shape).expand(*actions.shape[:-1], 1)
        residual_input = torch.cat([obs_rep, actions, t_rep], dim=-1)
        residual = self.delta_action_head(residual_input.reshape(-1, residual_input.shape[-1]))
        residual = torch.tanh(residual).reshape_as(actions)
        gate_shape = [obs.shape[0]] + [1] * (actions.ndim - 1)
        gate = self._delta_action_gate(obs).reshape(*gate_shape)
        corrected = actions + gate * self.delta_action_residual_scale * residual
        return corrected.reshape(*leading_shape, self.chunk_dim)

    def apply_continuous_action_residual(self, obs: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
        if self.continuous_action_head is None:
            return chunks
        leading_shape = chunks.shape[:-1]
        if leading_shape[0] != obs.shape[0]:
            raise ValueError("chunks batch dimension must match obs batch dimension")
        obs_view = obs.reshape(obs.shape[0], *([1] * (chunks.ndim - 2)), self.obs_dim)
        obs_rep = obs_view.expand(*leading_shape, self.obs_dim)
        residual = torch.zeros_like(chunks)
        for step in reversed(range(self.continuous_action_steps)):
            denom = max(1, self.continuous_action_steps - 1)
            t = torch.full((*leading_shape, 1), float(step) / float(denom), device=chunks.device, dtype=chunks.dtype)
            residual_input = torch.cat([obs_rep, chunks, residual, t], dim=-1)
            residual = torch.tanh(
                self.continuous_action_head(residual_input.reshape(-1, residual_input.shape[-1]))
            ).reshape_as(chunks)
        gate_shape = [obs.shape[0]] + [1] * (chunks.ndim - 1)
        gate = self._continuous_action_gate(obs).reshape(*gate_shape)
        return chunks + gate * self.continuous_action_residual_scale * residual

    def _local_to_world_residual(self, obs: torch.Tensor, local_residual: torch.Tensor) -> torch.Tensor:
        if self.obs_dim % 6 != 0:
            return local_residual
        leading_shape = local_residual.shape[:-1]
        actions = local_residual.reshape(*leading_shape, self.horizon, self.action_dim)
        latest = obs.reshape(obs.shape[0], self.obs_dim // 6, 6)[:, -1]
        block_xy = latest[:, 2:4]
        to_goal = -block_xy
        norm = torch.linalg.norm(to_goal, dim=-1, keepdim=True)
        angle_vec = F.normalize(latest[:, 4:6], dim=-1)
        forward = torch.where(norm > 1e-6, to_goal / norm.clamp_min(1e-6), angle_vec)
        side = torch.stack([-forward[:, 1], forward[:, 0]], dim=-1)
        basis_shape = [obs.shape[0]] + [1] * (actions.ndim - 2) + [self.action_dim]
        forward = forward.reshape(*basis_shape)
        side = side.reshape(*basis_shape)
        world = actions[..., 0:1] * forward + actions[..., 1:2] * side
        if self.local_continuous_action_last_steps > 0:
            last_steps = min(self.local_continuous_action_last_steps, self.horizon)
            mask = torch.zeros(self.horizon, device=local_residual.device, dtype=local_residual.dtype)
            mask[-last_steps:] = 1.0
            mask_shape = [1] * (actions.ndim - 2) + [self.horizon, 1]
            world = world * mask.reshape(*mask_shape)
        return world.reshape_as(local_residual)

    def apply_local_continuous_action_residual(self, obs: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
        if self.local_continuous_action_head is None:
            return chunks
        leading_shape = chunks.shape[:-1]
        if leading_shape[0] != obs.shape[0]:
            raise ValueError("chunks batch dimension must match obs batch dimension")
        obs_view = obs.reshape(obs.shape[0], *([1] * (chunks.ndim - 2)), self.obs_dim)
        obs_rep = obs_view.expand(*leading_shape, self.obs_dim)
        local_residual = torch.zeros_like(chunks)
        for step in reversed(range(self.local_continuous_action_steps)):
            denom = max(1, self.local_continuous_action_steps - 1)
            t = torch.full((*leading_shape, 1), float(step) / float(denom), device=chunks.device, dtype=chunks.dtype)
            residual_input = torch.cat([obs_rep, chunks, local_residual, t], dim=-1)
            local_residual = torch.tanh(
                self.local_continuous_action_head(residual_input.reshape(-1, residual_input.shape[-1]))
            ).reshape_as(chunks)
        world_residual = self._local_to_world_residual(obs, local_residual)
        gate_shape = [obs.shape[0]] + [1] * (chunks.ndim - 1)
        gate = self._local_continuous_action_gate(obs).reshape(*gate_shape)
        return chunks + gate * self.local_continuous_action_residual_scale * world_residual

    def _bezier_residual_from_controls(self, controls: torch.Tensor) -> torch.Tensor:
        leading_shape = controls.shape[:-1]
        control_points = controls.reshape(*leading_shape, 3, self.action_dim)
        t = torch.linspace(0.0, 1.0, self.horizon, device=controls.device, dtype=controls.dtype)
        one_minus_t = 1.0 - t
        # P0 is hardcoded to zero, so the b0 * P0 term is omitted. At t=0
        # every remaining basis coefficient is zero, guaranteeing no teleport.
        basis = torch.stack(
            [
                3.0 * one_minus_t.square() * t,
                3.0 * one_minus_t * t.square(),
                t.pow(3),
            ],
            dim=-1,
        )
        basis_shape = [1] * len(leading_shape) + [self.horizon, 3, 1]
        controls_shape = [*leading_shape, 1, 3, self.action_dim]
        curve = (basis.reshape(*basis_shape) * control_points.reshape(*controls_shape)).sum(dim=-2)
        return curve.reshape(*leading_shape, self.chunk_dim)

    def apply_local_spline_action_residual(self, obs: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
        if self.local_spline_action_head is None:
            return chunks
        leading_shape = chunks.shape[:-1]
        if leading_shape[0] != obs.shape[0]:
            raise ValueError("chunks batch dimension must match obs batch dimension")
        obs_view = obs.reshape(obs.shape[0], *([1] * (chunks.ndim - 2)), self.obs_dim)
        obs_rep = obs_view.expand(*leading_shape, self.obs_dim)
        head_input = torch.cat([obs_rep, chunks], dim=-1)
        controls = torch.tanh(self.local_spline_action_head(head_input.reshape(-1, head_input.shape[-1])))
        controls = controls.reshape(*leading_shape, 3 * self.action_dim)
        local_residual = self._bezier_residual_from_controls(controls)
        world_residual = self._local_to_world_residual(obs, local_residual)
        gate_shape = [obs.shape[0]] + [1] * (chunks.ndim - 1)
        gate = self._local_spline_action_gate(obs).reshape(*gate_shape)
        return chunks + gate * self.local_spline_action_residual_scale * world_residual

    def apply_contact_action_residual(self, obs: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
        if self.contact_action_head is None:
            return chunks
        leading_shape = chunks.shape[:-1]
        if leading_shape[0] != obs.shape[0]:
            raise ValueError("chunks batch dimension must match obs batch dimension")
        obs_view = obs.reshape(obs.shape[0], *([1] * (chunks.ndim - 2)), self.obs_dim)
        obs_rep = obs_view.expand(*leading_shape, self.obs_dim)
        contact = self.contact_features(obs)
        contact_view = contact.reshape(obs.shape[0], *([1] * (chunks.ndim - 2)), self.contact_feature_dim)
        contact_rep = contact_view.expand(*leading_shape, self.contact_feature_dim)
        residual = torch.zeros_like(chunks)
        for step in reversed(range(self.contact_action_steps)):
            denom = max(1, self.contact_action_steps - 1)
            t = torch.full((*leading_shape, 1), float(step) / float(denom), device=chunks.device, dtype=chunks.dtype)
            residual_input = torch.cat([obs_rep, contact_rep, chunks, residual, t], dim=-1)
            residual = torch.tanh(
                self.contact_action_head(residual_input.reshape(-1, residual_input.shape[-1]))
            ).reshape_as(chunks)
        gate_shape = [obs.shape[0]] + [1] * (chunks.ndim - 1)
        gate = self._contact_action_gate(obs).reshape(*gate_shape)
        return chunks + gate * self.contact_action_residual_scale * residual

    def multi_contact_candidate_chunks(self, obs: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
        if self.multi_contact_action_head is None:
            return chunks.unsqueeze(-2)
        leading_shape = chunks.shape[:-1]
        if leading_shape[0] != obs.shape[0]:
            raise ValueError("chunks batch dimension must match obs batch dimension")
        obs_view = obs.reshape(obs.shape[0], *([1] * (chunks.ndim - 2)), self.obs_dim)
        obs_rep = obs_view.expand(*leading_shape, self.obs_dim)
        contact = self.contact_features(obs)
        contact_view = contact.reshape(obs.shape[0], *([1] * (chunks.ndim - 2)), self.contact_feature_dim)
        contact_rep = contact_view.expand(*leading_shape, self.contact_feature_dim)
        residual_input = torch.cat([obs_rep, contact_rep, chunks], dim=-1)
        residual = torch.tanh(
            self.multi_contact_action_head(residual_input.reshape(-1, residual_input.shape[-1]))
        )
        residual = residual.reshape(*leading_shape, self.multi_contact_action_samples, self.chunk_dim)
        gate_shape = [obs.shape[0]] + [1] * (chunks.ndim - 1) + [1]
        gate = self._multi_contact_action_gate(obs).reshape(*gate_shape)
        return chunks.unsqueeze(-2) + gate * self.multi_contact_action_residual_scale * residual

    def _multi_contact_scores(self, obs: torch.Tensor, candidates: torch.Tensor, base_chunks: torch.Tensor) -> torch.Tensor:
        if self.obs_dim % 6 != 0:
            return (candidates - base_chunks.unsqueeze(-2)).square().mean(dim=-1)
        latest = obs.reshape(obs.shape[0], self.obs_dim // 6, 6)[:, -1]
        block_xy = latest[:, 2:4]
        goal_dir = F.normalize(-block_xy, dim=-1)
        behind = block_xy - goal_dir * (35.0 / 256.0)
        through = block_xy + goal_dir * (70.0 / 256.0)
        cand = candidates.reshape(*candidates.shape[:-1], self.horizon, self.action_dim)
        start = cand[..., 0, :]
        end = cand[..., -1, :]
        smooth = (cand[..., 1:, :] - cand[..., :-1, :]).square().mean(dim=(-1, -2))
        residual_mag = (candidates - base_chunks.unsqueeze(-2)).square().mean(dim=-1)
        prefix = [obs.shape[0]] + [1] * (candidates.ndim - 3)
        behind = behind.reshape(*prefix, 1, 2)
        through = through.reshape(*prefix, 1, 2)
        start_score = (start - behind).square().sum(dim=-1)
        end_score = (end - through).square().sum(dim=-1)
        return end_score + 0.25 * start_score + 0.1 * smooth + 0.05 * residual_mag

    def apply_multi_contact_action_residual(self, obs: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
        if self.multi_contact_action_head is None:
            return chunks
        candidates = self.multi_contact_candidate_chunks(obs, chunks)
        scores = self._multi_contact_scores(obs, candidates, chunks)
        selected = scores.argmin(dim=-1)
        gather_idx = selected.unsqueeze(-1).unsqueeze(-1).expand(*selected.shape, 1, self.chunk_dim)
        return candidates.gather(dim=-2, index=gather_idx).squeeze(-2)

    def flow_velocity(self, obs: torch.Tensor, chunks: torch.Tensor, residual: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.flow_action_head is None:
            return torch.zeros_like(residual)
        leading_shape = chunks.shape[:-1]
        obs_view = obs.reshape(obs.shape[0], *([1] * (chunks.ndim - 2)), self.obs_dim)
        obs_rep = obs_view.expand(*leading_shape, self.obs_dim)
        contact = self.contact_features(obs)
        contact_view = contact.reshape(obs.shape[0], *([1] * (chunks.ndim - 2)), self.contact_feature_dim)
        contact_rep = contact_view.expand(*leading_shape, self.contact_feature_dim)
        if t.shape[-1] != 1:
            raise ValueError("t must have a final singleton dimension")
        flow_input = torch.cat([obs_rep, contact_rep, chunks, residual, t.expand(*leading_shape, 1)], dim=-1)
        return self.flow_action_head(flow_input.reshape(-1, flow_input.shape[-1])).reshape_as(residual)

    def apply_flow_action_residual(self, obs: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
        if self.flow_action_head is None:
            return chunks
        residual = torch.zeros_like(chunks)
        leading_shape = chunks.shape[:-1]
        steps = max(1, int(self.flow_action_steps))
        for step in range(steps):
            t = torch.full((*leading_shape, 1), float(step) / float(steps), device=chunks.device, dtype=chunks.dtype)
            residual = (residual + self.flow_velocity(obs, chunks, residual, t) / float(steps)).clamp(-1.0, 1.0)
        gate_shape = [obs.shape[0]] + [1] * (chunks.ndim - 1)
        gate = self._flow_action_gate(obs).reshape(*gate_shape)
        return chunks + gate * self.flow_action_residual_scale * residual

    def hierarchical_flow_velocity(
        self,
        obs: torch.Tensor,
        sx: torch.Tensor,
        z: torch.Tensor,
        shat: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if self.hierarchical_flow_head is None:
            raise RuntimeError("hierarchical_flow_decoder is not enabled.")
        leading_shape = shat.shape[:-1]
        if leading_shape[0] != obs.shape[0]:
            raise ValueError("shat batch dimension must match obs batch dimension")
        obs_view = obs.reshape(obs.shape[0], *([1] * (shat.ndim - 2)), self.obs_dim)
        obs_rep = obs_view.expand(*leading_shape, self.obs_dim)
        sx_view = sx.reshape(obs.shape[0], *([1] * (shat.ndim - 2)), self.embedding_dim)
        sx_rep = sx_view.expand(*leading_shape, self.embedding_dim)
        if self.hierarchical_flow_no_z:
            z_rep = torch.zeros(*leading_shape, self.embedding_dim, device=shat.device, dtype=shat.dtype)
            shat_input = torch.zeros_like(shat)
        else:
            if z.ndim == 2:
                z_view = z.reshape(*([1] * (shat.ndim - 2)), z.shape[0], self.embedding_dim)
                z_rep = z_view.expand(*leading_shape, self.embedding_dim)
            else:
                z_rep = z.expand(*leading_shape, self.embedding_dim)
            shat_input = shat
        contact = self.contact_features(obs)
        contact_view = contact.reshape(obs.shape[0], *([1] * (shat.ndim - 2)), self.contact_feature_dim)
        contact_rep = contact_view.expand(*leading_shape, self.contact_feature_dim)
        flow_input = torch.cat(
            [obs_rep, contact_rep, sx_rep, z_rep, shat_input, xt, t.expand(*leading_shape, 1)],
            dim=-1,
        )
        return self.hierarchical_flow_head(flow_input.reshape(-1, flow_input.shape[-1])).reshape_as(xt)

    def hierarchical_flow_anchor(self, obs: torch.Tensor, leading_shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
        """Current end-effector pose, broadcast across the horizon and chunk layout.

        Used when the flow predicts *relative* action deltas: the integrated
        deltas are added back to this anchor to recover absolute targets.
        """
        leading_shape = tuple(leading_shape)
        if self.obs_dim % 6 != 0:
            return torch.zeros(*leading_shape, self.chunk_dim, device=obs.device, dtype=obs.dtype)
        agent_xy = obs.reshape(obs.shape[0], self.obs_dim // 6, 6)[:, -1, 0:2]  # (B, 2)
        anchor = agent_xy.reshape(obs.shape[0], *([1] * (len(leading_shape) - 1)), self.action_dim)
        anchor = anchor.expand(*leading_shape, self.action_dim)
        anchor = anchor.unsqueeze(-2).expand(*leading_shape, self.horizon, self.action_dim)
        return anchor.reshape(*leading_shape, self.chunk_dim)

    def hierarchical_flow_decode(
        self,
        obs: torch.Tensor,
        sx: torch.Tensor,
        z: torch.Tensor,
        shat: torch.Tensor,
        init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.hierarchical_flow_head is None:
            raise RuntimeError("hierarchical_flow_decoder is not enabled.")
        xt = torch.zeros(*shat.shape[:-1], self.chunk_dim, device=shat.device, dtype=shat.dtype) if init is None else init
        steps = max(1, int(self.hierarchical_flow_steps))
        for step in range(steps):
            t = torch.full((*shat.shape[:-1], 1), float(step) / float(steps), device=shat.device, dtype=shat.dtype)
            xt = (xt + self.hierarchical_flow_velocity(obs, sx, z, shat, xt, t) / float(steps)).clamp(-1.0, 1.0)
        if self.hierarchical_flow_relative:
            xt = xt + self.hierarchical_flow_anchor(obs, xt.shape[:-1])
        return xt

    def hierarchical_flow_decode_select(
        self,
        obs: torch.Tensor,
        sx: torch.Tensor,
        z: torch.Tensor,
        shat: torch.Tensor,
    ) -> torch.Tensor:
        """Generative inference for the flow micro-controller.

        When ``hierarchical_flow_noise_scale > 0`` the integration starts from
        sampled noise (true generative sampling) instead of zeros. With
        ``hierarchical_flow_samples > 1`` we draw several trajectories per code
        and keep, per code, the one whose re-encoding is most self-consistent
        with ``shat`` (cycle-consistency selection, no simulation required).
        """
        noise_scale = float(self.hierarchical_flow_noise_scale)
        num_samples = max(1, int(self.hierarchical_flow_samples))
        if noise_scale <= 0.0 and num_samples <= 1:
            return self.hierarchical_flow_decode(obs, sx, z, shat)
        samples = []
        for _ in range(num_samples):
            if noise_scale > 0.0:
                init = noise_scale * torch.randn(
                    *shat.shape[:-1], self.chunk_dim, device=shat.device, dtype=shat.dtype
                )
            else:
                init = None
            samples.append(self.hierarchical_flow_decode(obs, sx, z, shat, init=init))
        if num_samples <= 1:
            return samples[0]
        stacked = torch.stack(samples, dim=0)  # [S, ..., chunk_dim]
        reencoded = self.target_encoder(stacked.reshape(-1, self.chunk_dim))
        reencoded = reencoded.reshape(*stacked.shape[:-1], self.embedding_dim)
        cycle_gap = (reencoded - shat.unsqueeze(0)).square().sum(dim=-1)  # [S, ...]
        best = cycle_gap.argmin(dim=0)  # [...]
        gather_idx = best.unsqueeze(0).unsqueeze(-1).expand(1, *best.shape, self.chunk_dim)
        return stacked.gather(0, gather_idx).squeeze(0)

    def fine_vq_candidate_chunks(self, obs: torch.Tensor, chunks: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.fine_vq_head is None or self.fine_vq_latent is None or self.fine_vq_prior_head is None:
            raise RuntimeError("fine_vq_residual is not enabled.")
        leading_shape = chunks.shape[:-1]
        if leading_shape[0] != obs.shape[0]:
            raise ValueError("chunks batch dimension must match obs batch dimension")
        z, _ = self.fine_vq_latent.enumerate_codes(chunks.device, chunks.dtype)
        obs_view = obs.reshape(obs.shape[0], *([1] * (chunks.ndim - 2)), self.obs_dim)
        obs_rep = obs_view.expand(*leading_shape, self.obs_dim)
        fine_logits = self.fine_vq_prior_head(torch.cat([obs_rep, chunks], dim=-1).reshape(-1, self.obs_dim + self.chunk_dim))
        fine_logits = fine_logits.reshape(*leading_shape, self.fine_vq_codebook_size)
        obs_fine = obs_rep.unsqueeze(-2).expand(*leading_shape, self.fine_vq_codebook_size, self.obs_dim)
        chunks_fine = chunks.unsqueeze(-2).expand(*leading_shape, self.fine_vq_codebook_size, self.chunk_dim)
        z_view = z.reshape(*([1] * len(leading_shape)), self.fine_vq_codebook_size, self.embedding_dim)
        z_rep = z_view.expand(*leading_shape, self.fine_vq_codebook_size, self.embedding_dim)
        residual_input = torch.cat([obs_fine, chunks_fine, z_rep], dim=-1)
        residual = torch.tanh(
            self.fine_vq_head(residual_input.reshape(-1, residual_input.shape[-1]))
        ).reshape(*leading_shape, self.fine_vq_codebook_size, self.chunk_dim)
        gate_shape = [obs.shape[0]] + [1] * (chunks.ndim - 1)
        gate = self._fine_vq_gate(obs).reshape(*gate_shape)
        fine_chunks = chunks.unsqueeze(-2) + gate.unsqueeze(-1) * self.fine_vq_residual_scale * residual
        return {
            "fine_chunks": fine_chunks,
            "fine_logits": fine_logits,
            "fine_residual": residual,
        }

    def apply_fine_vq_residual(self, obs: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
        if self.fine_vq_head is None:
            return chunks
        out = self.fine_vq_candidate_chunks(obs, chunks)
        selected = out["fine_logits"].argmax(dim=-1)
        gather_idx = selected.unsqueeze(-1).unsqueeze(-1).expand(*selected.shape, 1, self.chunk_dim)
        return out["fine_chunks"].gather(dim=-2, index=gather_idx).squeeze(-2)

    def forward(
        self,
        obs: torch.Tensor,
        chunk: torch.Tensor,
        select_temp: float = 64.0,
    ) -> dict[str, torch.Tensor]:
        sx = self.obs_encoder(obs)
        sy = self.target_encoder(chunk)
        codes = self.code_predictions(sx)
        dist = (codes["shat"] - sy.detach()[:, None, :]).square().sum(dim=-1)
        resp = torch.softmax(-float(select_temp) * dist, dim=-1)
        usage = resp.mean(dim=0)
        code_perplexity = torch.exp(-(usage * (usage + 1e-12).log()).sum())
        if self.hierarchical_flow_head is not None:
            chunk_hat = self.hierarchical_flow_decode_select(obs, sx, codes["z"], codes["shat"])
        else:
            shat_flat = codes["shat"].reshape(-1, self.embedding_dim)
            chunk_hat = self.decoder(shat_flat).reshape(obs.shape[0], -1, self.chunk_dim)
            chunk_hat = self._apply_final_alignment_residual(obs, sx, codes["shat"], chunk_hat)
            chunk_hat = self.apply_delta_action_residual(obs, chunk_hat)
            chunk_hat = self.apply_continuous_action_residual(obs, chunk_hat)
            chunk_hat = self.apply_local_continuous_action_residual(obs, chunk_hat)
            chunk_hat = self.apply_local_spline_action_residual(obs, chunk_hat)
            chunk_hat = self.apply_contact_action_residual(obs, chunk_hat)
            chunk_hat = self.apply_multi_contact_action_residual(obs, chunk_hat)
            chunk_hat = self.apply_flow_action_residual(obs, chunk_hat)
            chunk_hat = self.apply_fine_vq_residual(obs, chunk_hat)
        prior_logits = self.prior_head(sx)
        sx_rep = sx[:, None, :].expand(-1, self.codebook_size, -1)
        reranker_input = torch.cat([sx_rep, codes["shat"], chunk_hat], dim=-1)
        reranker_input = reranker_input.detach()
        reranker_logits = self.reranker_head(reranker_input.reshape(-1, reranker_input.shape[-1]))
        reranker_logits = reranker_logits.reshape(obs.shape[0], self.codebook_size)
        return {
            "sx": sx,
            "sy": sy,
            "shat": codes["shat"],
            "chunk_hat": chunk_hat,
            "dist": dist,
            "softmin": -torch.logsumexp(-float(select_temp) * dist, dim=-1) / float(select_temp),
            "winner": dist.argmin(dim=-1),
            "responsibilities": resp,
            "code_perplexity": code_perplexity,
            "rate_bits": codes["rate_bits"],
            "prior_logits": prior_logits,
            "reranker_logits": reranker_logits,
        }

    def cycle_consistency_logits(self, shat: torch.Tensor, chunk_hat: torch.Tensor) -> torch.Tensor:
        """Test-time selection surrogate: reuse the training winner rule with the
        true future replaced by the model's own decoded reconstruction.

        Training picks ``argmin_k ||shat_k - E_y(a*)||``. Here ``a*`` is unknown, so
        we substitute ``E_y(Dec(shat_k))`` and score how self-consistent each code
        is. Larger logit = smaller cycle gap. No new parameters, no retraining.
        """
        batch, num_codes = shat.shape[0], shat.shape[1]
        reencoded = self.target_encoder(chunk_hat.reshape(batch * num_codes, self.chunk_dim))
        reencoded = reencoded.reshape(batch, num_codes, self.embedding_dim)
        cycle_gap = (shat - reencoded).square().sum(dim=-1)
        return -cycle_gap

    def candidate_outputs(self, obs: torch.Tensor, apply_fine_vq: bool = True) -> dict[str, torch.Tensor]:
        sx = self.obs_encoder(obs)
        codes = self.code_predictions(sx)
        shat = codes["shat"]
        if self.hierarchical_flow_head is not None:
            chunk_hat = self.hierarchical_flow_decode_select(obs, sx, codes["z"], shat)
        else:
            chunk_hat = self.decoder(shat.reshape(-1, self.embedding_dim)).reshape(obs.shape[0], -1, self.chunk_dim)
            chunk_hat = self._apply_final_alignment_residual(obs, sx, shat, chunk_hat)
            chunk_hat = self.apply_delta_action_residual(obs, chunk_hat)
            chunk_hat = self.apply_continuous_action_residual(obs, chunk_hat)
            chunk_hat = self.apply_local_continuous_action_residual(obs, chunk_hat)
            chunk_hat = self.apply_local_spline_action_residual(obs, chunk_hat)
            chunk_hat = self.apply_contact_action_residual(obs, chunk_hat)
            chunk_hat = self.apply_multi_contact_action_residual(obs, chunk_hat)
            chunk_hat = self.apply_flow_action_residual(obs, chunk_hat)
            if apply_fine_vq:
                chunk_hat = self.apply_fine_vq_residual(obs, chunk_hat)
        sx_rep = sx[:, None, :].expand(-1, self.codebook_size, -1)
        reranker_input = torch.cat([sx_rep, shat, chunk_hat], dim=-1)
        reranker_logits = self.reranker_head(reranker_input.reshape(-1, reranker_input.shape[-1]))
        reranker_logits = reranker_logits.reshape(obs.shape[0], self.codebook_size)
        return {
            "sx": sx,
            "shat": shat,
            "chunk_hat": chunk_hat,
            "prior_logits": self.prior_head(sx),
            "reranker_logits": reranker_logits,
            "cycle_logits": self.cycle_consistency_logits(shat, chunk_hat),
        }

    @torch.no_grad()
    def predict_all_chunks(self, obs: torch.Tensor) -> torch.Tensor:
        return self.candidate_outputs(obs)["chunk_hat"]

    @torch.no_grad()
    def prior_logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.prior_head(self.obs_encoder(obs))


def lejepa_loss(out: dict[str, torch.Tensor], y: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred = F.mse_loss(out["shat"], out["sy"].detach())
    readout = F.mse_loss(out["decoder_y"] if "decoder_y" in out else out["yhat"], y)
    return pred + readout, {"pred": pred, "readout": readout}

